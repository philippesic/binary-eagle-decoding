"""Authenticated completed-corpus adapter; model/data loading remains in pinned source."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT = '6f1444b86dd01862da878c2d5d2434a1d9165c29'
NATIVE_COMMIT = '9e2c7a90051e738751aab7d7bd7c2d8201fb76e3'
COMMON = ('target_gguf', 'candidate_d_gguf', 'base_draft_gguf', 'absolute_d2t', 'model_snapshot_manifest')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink() and path.is_file(),
            'Expected canonical regular file: ' + str(path))
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


class Files:
    """Only this process's successful byte hashes can initialize stat reuse."""
    def __init__(self):
        self.verified = {}

    def check(self, path, expected):
        path = Path(path)
        before = identity(path)
        key = (str(path), expected, before)
        if key not in self.verified:
            require(sha256(path) == expected, 'Artifact SHA differs: ' + str(path))
            require(identity(path) == before, 'Artifact changed while hashing: ' + str(path))
            self.verified[key] = True
        return path

    def text(self, path, expected):
        path = self.check(path, expected)
        before = identity(path)
        result = path.read_text()
        require(identity(path) == before, 'Artifact changed while parsing: ' + str(path))
        return result

    def read(self, path, expected):
        return json.loads(self.text(path, expected))


def source_api(checkout):
    checkout = Path(checkout).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True, timeout=15).strip()
    require(git('rev-parse', 'HEAD') == SOURCE_COMMIT, 'Only pinned source6f is supported')
    require(git('status', '--porcelain', '--untracked-files=normal') == '', 'Pinned source is dirty')
    require(git('ls-tree', 'HEAD', 'third_party/llama.cpp').split()[2] == NATIVE_COMMIT,
            'Pinned source native gitlink differs')
    native = subprocess.check_output(['git', '-C', str(checkout / 'third_party/llama.cpp'),
                                      'rev-parse', 'HEAD'], text=True, timeout=15).strip()
    require(native == NATIVE_COMMIT, 'Pinned native checkout differs')
    for name, module in list(sys.modules.items()):
        if name.startswith(('w1a1_eagle', 'w1ax_', 'train_continuous_w1ax', 'check_qat_optimization_readiness')):
            file = getattr(module, '__file__', None)
            require(file is not None and Path(file).resolve().is_relative_to(checkout),
                    'Mixed source import: ' + name)
    sys.path[:0] = [str(checkout / 'scripts'), str(checkout / 'src')]
    launcher = importlib.import_module('train_continuous_w1ax')
    require(Path(launcher.__file__).resolve() == checkout / 'scripts/train_continuous_w1ax.py',
            'Launcher import escaped pinned source')
    return launcher


def authenticate(api, spec, new_run, prepared_run, ready_sha):
    files = Files()
    prepared_run = Path(prepared_run).resolve()
    ready = files.read(prepared_run / 'preparation-ready.json', ready_sha)
    binding, train_index, development = api.prepared_corpus_inputs(
        spec, Path(new_run).resolve(), prepared_run, ready_sha)
    # Unchanged helper authenticates all source/checkpoint/smoke/coverage metadata.
    # Independently stabilize its artifact bytes before any lazy cache use.
    for path, digest in binding['artifacts'].items():
        files.check(Path(path), digest)
    source = ready['teacher_coverage']['source']
    require(api.prepared_digest(source) == binding['source_sha256'], 'Prepared full source differs')
    index = files.read(train_index, source['execution_manifest_sha256'])
    require(index['split'] == 'train' and index['training_eligible'] is True,
            'Prepared adapter requires full TRAIN index')
    require(len(index['shards']) == len(source['shards']), 'Complete source shard inventory differs')
    records, ids = [], set()
    for ordinal, (record, source_record) in enumerate(zip(index['shards'], source['shards'])):
        require(record['ordinal'] == source_record['ordinal'] == ordinal
                and record['provider_manifest_sha256'] == source_record['provider_manifest_sha256'],
                'Prepared shard/source membership differs')
        child = files.read(record['provider_manifest'], record['provider_manifest_sha256'])
        require(child['split'] == 'train' and child['training_eligible'] is True
                and child['sha256']['capture_manifest'] == source_record['capture_manifest_sha256']
                and all(child['sha256'][k] == source['common_source_sha256'][k] for k in COMMON),
                'Prepared child changes frozen source/split')
        prompt_text = files.text(child['paths']['prompts'], child['sha256']['prompts'])
        child_ids = [json.loads(line)['id'] for line in prompt_text.split('\n') if line.strip()]
        require(len(child_ids) == child['prompt_count'] == source_record['prompt_count']
                and len(set(child_ids)) == len(child_ids) and not ids.intersection(child_ids),
                'Prepared prompt coverage has duplicates/count mismatch')
        ids.update(child_ids)
        records.append((record, child, frozenset(child_ids)))
    require(len(ids) == source['prompt_count'] == ready['teacher_coverage']['unique_train_prompts'],
            'Authenticated full prompt coverage differs')
    require(ready['teacher_coverage']['unique_supervised_rows'] >= spec['coverage']['min_unique_supervised_rows']
            and len(ids) >= spec['coverage']['min_unique_train_prompts'], 'Completed coverage below required floor')
    return {'binding': binding, 'ready': ready, 'source': source, 'records': records,
            'ids': frozenset(ids), 'files': files, 'development': development}


class PreparedProvider:
    """Full authenticated corpus identity; instantiate/check payloads only when used."""
    def __init__(self, authenticated, config, child_factory, selected_shard=0):
        self.auth = authenticated
        self._config, self._factory = config, child_factory
        require(type(selected_shard) is int and 0 <= selected_shard < len(self.auth['records']),
                'Selected TRAIN shard is absent')
        self.selected_shard = selected_shard
        self.source_metadata = copy.deepcopy(self.auth['source'])
        self.allowed_prompt_ids = set(self.auth['ids'])
        self.training_eligible = True
        self.full_body_qat_eligible = True
        self.readiness_scope = 'full_body_qat'
        self.split = self.data_split = 'train'
        self.total_rounds = self.source_metadata['round_count']
        self.supervised_rows = self.auth['ready']['teacher_coverage']['unique_supervised_rows']
        self._first = None
        self.candidate_d = None

    def _child(self, ordinal):
        record, spec, ids = self.auth['records'][ordinal]
        path = self.auth['files'].check(record['provider_manifest'], record['provider_manifest_sha256'])
        child = self._factory(self._config, path)
        require(child.training_eligible is True and child.full_body_qat_eligible is True
                and child.data_split == 'train' and set(child.allowed_prompt_ids) == set(ids)
                and all(child.hashes[k] == self.source_metadata['common_source_sha256'][k] for k in COMMON)
                and child.capture_id == spec['capture_id'], 'Lazy child membership/eligibility/source differs')
        require(child.total_rounds == self.source_metadata['shards'][ordinal]['round_count'],
                'Lazy child round inventory differs from completed source')
        if self._first is not None:
            require(tuple(child.d2t_offsets) == tuple(self._first.d2t_offsets)
                    and child.target_vocab_size == self._first.target_vocab_size
                    and child.draft_vocab_size == self._first.draft_vocab_size,
                    'Lazy child changes model/map contract')
        return child

    def first(self):
        if self._first is None:
            self._first = self._child(self.selected_shard)
        return self._first

    def __getattr__(self, name):
        if name in ('target_vocab_size', 'draft_vocab_size', 'max_depth', 'd2t_offsets', 'base_gguf_sha256'):
            return getattr(self.first(), name)
        raise AttributeError(name)

    def load_models_cpu(self):
        result = self.first().load_models_cpu()
        self.candidate_d = self.first().candidate_d
        return result

    def make_step_adapter(self, model):
        return self.first().make_step_adapter(model)

    def bounded_rounds(self):
        for batch in self.first().rounds():
            yield dataclasses.replace(batch, shard_ordinal=self.selected_shard)

    def rounds(self):
        for ordinal in range(len(self.auth['records'])):
            child = self.first() if ordinal == self.selected_shard else self._child(ordinal)
            for batch in child.rounds():
                yield dataclasses.replace(batch, shard_ordinal=ordinal)
            if child is not self._first:
                del child


def create_provider(config, manifest_path):
    """Existing gate CLI factory, using the ORIGINAL full TRAIN index as manifest.

    The extra authorization locator pins this adapter independently. It does not
    rewrite/rename a child's source or grant GPU/data eligibility by itself.
    """
    locator = os.environ.get('QAT_PREPARED_BINDING_PATH')
    digest = os.environ.get('QAT_PREPARED_BINDING_SHA256')
    require(bool(locator) and bool(digest), 'Pinned prepared provider binding environment required')
    record = Files().read(locator, digest)
    from train_prepared_continuous_w1ax import code_identity
    require(record.get('schema') == 'prepared_provider_binding_v1'
            and record.get('orchestration_sha256') == code_identity(), 'Prepared adapter code binding differs')
    api = source_api(record['source_checkout'])
    files = Files()
    config_path = files.check(record['config']['path'], record['config']['sha256'])
    spec, continuous = api.load_config(config_path)
    if record.get('stages_manifest') is not None:
        stage = record['stages_manifest']
        spec['stages'] = files.read(stage['path'], stage['sha256'])
    auth = authenticate(api, spec, record['validation_run_dir'], record['prepared_run_dir'],
                        record['prepared_ready_sha256'])
    require(Path(manifest_path).resolve() == Path(auth['binding']['provider_manifest']),
            'Gate must request authenticated ORIGINAL full provider index')
    require(dataclasses.asdict(config) == dataclasses.asdict(continuous.qat(config.contract.activation_bits)),
            'Gate changes declared prepared recipe')
    child_factory = importlib.import_module('w1ax_capture_provider').NativeCaptureProvider
    provider = PreparedProvider(auth, config, child_factory, record['selected_shard'])
    # Native/model gates use selected TRAIN cases while source identity stays FULL.
    class BoundedProvider:
        def __getattr__(self, name):
            return getattr(provider, name)
        def rounds(self):
            return provider.bounded_rounds()
    return BoundedProvider()
