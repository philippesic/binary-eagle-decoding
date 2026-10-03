#!/usr/bin/env python3
"""New-run checkpoint-zero handoff using unchanged source6f; fail-closed optimizer gates."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import uuid

from prepared_continuous_provider import (
    COMMON, Files, NATIVE_COMMIT, SOURCE_COMMIT, PreparedProvider,
    authenticate, identity, require, sha256, source_api,
)

HERE = Path(__file__).resolve().parent


def code_identity():
    return {name: sha256(HERE / name) for name in
            ('train_prepared_continuous_w1ax.py', 'prepared_continuous_provider.py')}


def checkpoint_inventory(auth):
    checkpoint = Path(auth['ready']['checkpoint']['path'])
    files = auth['files']
    manifest_path = checkpoint.parent / 'manifest.json'
    expected = auth['binding']['artifacts'][str(manifest_path)]
    manifest = files.read(manifest_path, expected)
    require(all(type(manifest[k]) is int and manifest[k] == 0 for k in ('step', 'epoch', 'cursor')),
            'Only genuine zero checkpoint may initialize new run')
    inventory = {'resume.pt': auth['ready']['checkpoint']['sha256'], 'manifest.json': expected}
    for lane in ('A8', 'A1'):
        require(set(manifest['exports'][lane]) == {'joint.npz', 'joint.json'}, 'Paired export inventory differs')
        for name, digest in manifest['exports'][lane].items():
            inventory[lane + '/' + name] = digest
    actual = {str(path.relative_to(checkpoint.parent)) for path in checkpoint.parent.rglob('*') if path.is_file()}
    require(actual == set(inventory), 'Checkpoint publication has unknown/missing files')
    require(not any(path.is_symlink() for path in checkpoint.parent.rglob('*')), 'Checkpoint publication has symlink')
    return checkpoint.parent, inventory


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_zero(auth, run_dir):
    origin, inventory = checkpoint_inventory(auth)
    parent = Path(run_dir) / 'checkpoints'
    parent.mkdir(exist_ok=False)
    destination = parent / origin.name
    temporary = parent / ('.' + origin.name + '.copy-' + uuid.uuid4().hex)
    temporary.mkdir()
    for name, digest in inventory.items():
        source = auth['files'].check(origin / name, digest)
        before = identity(source)
        output = temporary / name
        output.parent.mkdir(parents=True, exist_ok=True)
        actual = hashlib.sha256()
        with source.open('rb') as reader, output.open('xb') as writer:
            for block in iter(lambda: reader.read(8 * 1024**2), b''):
                actual.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        require(actual.hexdigest() == digest and identity(source) == before, 'Zero checkpoint changed during copy')
    for folder in sorted((p for p in temporary.rglob('*') if p.is_dir()), reverse=True):
        fsync_dir(folder)
    fsync_dir(temporary)
    os.rename(temporary, destination)
    fsync_dir(parent)
    return {'origin': str(origin), 'publication': str(destination), 'sha256': inventory,
            'original_latest_copied': False, 'atomic_pair_copy': True}


def enforce_gates(api, args, auth, config, provider, lanes):
    files = Files()
    admission = files.read(args.admission, args.admission_sha256)
    require(admission.get('schema') == 'prepared_continuous_launch_admission_v1'
            and admission.get('orchestration_sha256') == code_identity()
            and admission.get('source_commit') == SOURCE_COMMIT and admission.get('native_commit') == NATIVE_COMMIT
            and admission.get('prepared_ready_sha256') == auth['binding']['prepared_ready_sha256']
            and admission.get('source_sha256') == auth['binding']['source_sha256']
            and admission.get('origin_zero_checkpoint_sha256') == auth['ready']['checkpoint']['sha256']
            and admission.get('run_dir') == str(Path(args.run_dir).resolve()),
            'Specific current launch admission absent')
    now = time.time()
    require(type(admission.get('granted_unix')) in (int, float)
            and type(admission.get('expires_unix')) in (int, float)
            and admission['granted_unix'] <= now < admission['expires_unix'] <= admission['granted_unix'] + 300
            and admission.get('sole_gpu_owner') is True and admission.get('rtx5080_pause_requested') is False,
            'Fresh exclusive unpaused admission absent')
    runtime_module = importlib.import_module('w1a1_eagle.continuous_runtime')
    readiness = importlib.import_module('w1a1_eagle.qat_readiness')
    runtime = runtime_module.training_runtime_identity(config.device)
    require(runtime == auth['ready']['training_runtime'], 'Original zero checkpoint runtime differs')
    native_tool = importlib.import_module('check_qat_optimization_readiness')
    deployments = {lane.name: native_tool.runtime_api().deployment_state_sha256(lane.linears) for lane in lanes}
    torch = importlib.import_module('torch')
    properties = torch.cuda.get_device_properties(config.device)
    hardware = {'device_type': 'cuda', 'name': properties.name,
                'compute_capability': [int(properties.major), int(properties.minor)],
                'total_memory_bytes': int(properties.total_memory)}
    locator = admission.get('optimization_readiness')
    require(isinstance(locator, dict) and set(locator) == {'path', 'sha256'}, 'Missing actual model/native readiness')
    receipt = files.read(locator['path'], locator['sha256'])
    validation_config = dataclasses.replace(config, optimization_readiness=locator)
    readiness.validate_optimization_readiness(validation_config,
        source_sha256=auth['binding']['source_sha256'], runtime_identity=runtime,
        native_commit=NATIVE_COMMIT, backend='cuda', hardware=hardware)
    require(receipt.get('deployment_state_sha256') == deployments and receipt.get('optimizer_updates') == 0
            and receipt.get('timing_repeats') == 5, 'Actual restored state/five-repeat timing gate differs')
    require(not os.path.lexists(Path(args.run_dir) / 'CANCEL'), 'New run cancelled')
    return {'admission_sha256': args.admission_sha256, 'runtime': runtime,
            'deployment_state_sha256': deployments, 'orchestration_sha256': code_identity()}


def prepare(args):
    api = source_api(args.source_checkout)
    spec, config = api.load_config(args.config)
    if args.stages_manifest:
        spec['stages'] = json.loads(Path(args.stages_manifest).read_text())
    require(isinstance(spec.get('stages'), dict), 'Pinned stages required')
    run_dir, old = Path(args.run_dir).resolve(), Path(args.prepared_run_dir).resolve()
    require(run_dir != old and run_dir not in old.parents and old not in run_dir.parents,
            'New run overlaps immutable preparation')
    require(not run_dir.exists(), 'New run must not exist')
    auth = authenticate(api, spec, run_dir, old, args.prepared_ready_sha256)
    old_spec, old_config = api.load_config(old / 'resolved_config.json')
    immutable = importlib.import_module('w1a1_eagle.continuous_qat').immutable_config
    require(immutable(dataclasses.asdict(config)) == immutable(dataclasses.asdict(old_config)),
            'Only existing four run caps may differ from zero checkpoint config')
    return api, spec, config, auth


def start(args):
    require(args.allow_cuda and args.admission and args.admission_sha256,
            'Actual start requires CUDA permission and exact complete gate admission')
    # Do not load any model when an admission or required gate artifact is absent.
    initial = Files().read(args.admission, args.admission_sha256)
    for name in ('optimization_readiness',):
        record = initial.get(name)
        require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Missing gate ' + name)
        Files().check(record['path'], record['sha256'])
    api, spec, config, auth = prepare(args)
    torch = importlib.import_module('torch')
    require(torch.cuda.is_available(), 'Actual CUDA hardware required')
    frozen = auth['ready']['training_runtime']['cuda_math']
    torch.set_float32_matmul_precision(frozen['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = frozen['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = frozen['cudnn_allow_tf32']
    require(api.training_runtime_identity(config.device) == auth['ready']['training_runtime'],
            'Pinned source/runtime/precision changed')
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    run_lock = api.lock(run / '.owner.lock')
    gpu_lock = api.lock(Path.home() / '.cache/binary-eagle-decoding/cuda-0.owner.lock')
    try:
        api.atomic_json(run / 'resolved_config.json', spec)
        api.atomic_json(run / 'prepared-corpus.json', auth['binding'])
        api.atomic_json(run / 'orchestration.json', {'source_commit': SOURCE_COMMIT,
            'native_commit': NATIVE_COMMIT, 'orchestration_sha256': code_identity(), 'optimizer_updates': 0})
        child = importlib.import_module('w1ax_capture_provider').NativeCaptureProvider
        provider = PreparedProvider(auth, config.qat(8), child, args.selected_shard)
        publication = copy_zero(auth, run)
        api.atomic_json(run / 'zero-checkpoint-origin.json', publication)
        lanes = api.build_lanes(provider, config, run)
        stages = importlib.import_module('w1ax_continuous_stages')
        development = auth['development']
        def evaluator(checkpoint, _):
            return stages.evaluate_development(development, Path(checkpoint['path']).parent, run)
        trainer = api.ContinuousTrainer(provider, lanes, config, run, development_evaluator=evaluator)
        trainer.resume()
        api.require_zero_optimizer_progress(trainer)
        require(trainer.checkpoint['sha256'] == auth['ready']['checkpoint']['sha256']
                and Path(trainer.checkpoint['path']).resolve().is_relative_to(run)
                and trainer.source == auth['binding']['source_sha256']
                and trainer.runtime_identity == auth['ready']['training_runtime']
                and all(type(value) is int and value == 0
                        for value in (trainer.step, trainer.epoch, trainer.cursor, trainer.tokens))
                and not trainer.unique_prompts and not trainer.unique_rows,
                'Actual restored zero checkpoint state/source/runtime differs')
        view = type('BoundedPreparedView', (), {'rounds': lambda self: provider.bounded_rounds(),
                    '__getattr__': lambda self, name: getattr(provider, name)})()
        tool = importlib.import_module('check_qat_optimization_readiness')
        rounds = tool.selected_rounds(view, tool.runtime_api(), config)
        restored = {lane.name: tool.deterministic_state_sha256(lane.linears) for lane in lanes}
        api.atomic_json(run / 'save-resume.json', {
            'schema': 'prepared_checkpoint_actual_restore_v1',
            'original_saved_checkpoint_sha256': auth['ready']['checkpoint']['sha256'],
            'current_restored_checkpoint': trainer.checkpoint,
            'source_sha256': trainer.source, 'training_runtime': trainer.runtime_identity,
            'actual_restored_state_sha256': restored, 'counters': {'step': trainer.step,
                'epoch': trainer.epoch, 'cursor': trainer.cursor, 'tokens': trainer.tokens},
            'original_save_authenticated': True, 'actual_resume_completed': True, 'optimizer_updates': 0})
        for batch, _ in rounds[:2]:
            trainer.smoke(batch)  # Actual paired backward; never set smoke_passed ourselves.
        api.require_zero_optimizer_progress(trainer)
        require(restored == {lane.name: tool.deterministic_state_sha256(lane.linears) for lane in lanes},
                'Current smoke mutated restored checkpoint weights')
        evidence = enforce_gates(api, args, auth, config, provider, lanes)
        api.atomic_json(run / 'launch-gates.json', evidence)
        trainer.run()
    finally:
        gpu_lock.close()
        run_lock.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--start', action='store_true')
    p.add_argument('--source-checkout', type=Path)
    p.add_argument('--prepared-run-dir', type=Path)
    p.add_argument('--prepared-ready-sha256')
    p.add_argument('--config', type=Path)
    p.add_argument('--stages-manifest', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--selected-shard', type=int, default=0)
    p.add_argument('--allow-cuda', action='store_true')
    p.add_argument('--emit-provider-binding', type=Path)
    p.add_argument('--admission', type=Path)
    p.add_argument('--admission-sha256')
    return p


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not args.start and not args.validate:
        print(json.dumps({'schema': 'prepared_continuous_startup_plan_v1', 'source_commit': SOURCE_COMMIT,
            'native_commit': NATIVE_COMMIT, 'orchestration_sha256': code_identity(),
            'optimizer_allowed': False, 'model_loaded': False,
            'required_gates': ['full-source actual native/model/backward/memory/five-repeat timing receipt',
                               'authenticated actual saved checkpoint plus unchanged trainer resume and current smoke',
                               'Q4_0 development comparison retained as training supervision'],
            'new_entrypoint_identity': True, 'original_source_edited': False}, sort_keys=True))
        return
    for name in ('source_checkout', 'prepared_run_dir', 'prepared_ready_sha256', 'config', 'run_dir'):
        if getattr(args, name) is None:
            cli.error('Missing --' + name.replace('_', '-'))
    if args.validate:
        _, _, _, auth = prepare(args)
        if args.emit_provider_binding is not None:
            require(args.emit_provider_binding.is_absolute()
                    and args.emit_provider_binding == args.emit_provider_binding.resolve(), 'New canonical binding output required')
            require(not args.emit_provider_binding.resolve().is_relative_to(args.prepared_run_dir.resolve()), 'No original run writes')
            record = {'schema': 'prepared_provider_binding_v1', 'source_checkout': str(args.source_checkout.resolve()),
                      'prepared_run_dir': str(args.prepared_run_dir.resolve()),
                      'prepared_ready_sha256': args.prepared_ready_sha256,
                      'validation_run_dir': str(args.run_dir.resolve()), 'selected_shard': args.selected_shard,
                      'orchestration_sha256': code_identity(),
                      'config': {'path': str(args.config.resolve()), 'sha256': sha256(args.config)}}
            if args.stages_manifest is not None:
                record['stages_manifest'] = {'path': str(args.stages_manifest.resolve()), 'sha256': sha256(args.stages_manifest)}
            with args.emit_provider_binding.open('x') as stream:
                json.dump(record, stream, sort_keys=True, indent=2); stream.write('\n')
                stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({'schema': 'prepared_continuous_authentication_v1',
            'source_sha256': auth['binding']['source_sha256'], 'prompt_count': len(auth['ids']),
            'coverage_rows': auth['ready']['teacher_coverage']['unique_supervised_rows'],
            'origin_checkpoint': auth['ready']['checkpoint'], 'orchestration_sha256': code_identity(),
            'payloads_loaded': False, 'model_loaded': False, 'optimizer_allowed': False}, sort_keys=True))
    else:
        start(args)


if __name__ == '__main__':
    main()
