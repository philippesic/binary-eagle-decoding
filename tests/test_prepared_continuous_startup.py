"""Tiny synthetic local metadata/checkpoint bytes; no model/data/CUDA execution."""
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepared_continuous_provider as adapter
import train_prepared_continuous_w1ax as launch
import check_prepared_continuous_readiness as measured


@dataclasses.dataclass
class Batch:
    token: int
    shard_ordinal: int | None = None


class StartupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve(); self.prep = self.root / 'prepared'; self.prep.mkdir()
        self.source = {'factory': 'original-frozen-provider', 'prompt_count': 2, 'round_count': 2,
                       'common_source_sha256': {name: 'a'*64 for name in adapter.COMMON}, 'shards': []}
        self.records = []; self.artifacts = {}
        for ordinal in range(2):
            prompt = self.prep / f'prompts-{ordinal}.jsonl'; prompt.write_text(json.dumps({'id':f'train-{ordinal}'})+'\n')
            child = self.prep / f'child-{ordinal}.json'
            spec = {'split': 'train', 'training_eligible': True, 'prompt_count': 1,
                    'capture_id': f'capture-{ordinal}', 'paths': {'prompts': str(prompt)},
                    'sha256': {**self.source['common_source_sha256'], 'prompts': adapter.sha256(prompt),
                               'capture_manifest': str(ordinal)*64}}
            child.write_text(json.dumps(spec))
            record = {'ordinal': ordinal, 'provider_manifest': str(child), 'provider_manifest_sha256': adapter.sha256(child)}
            self.records.append(record)
            self.source['shards'].append({'ordinal': ordinal, 'provider_manifest_sha256': record['provider_manifest_sha256'],
                                          'capture_manifest_sha256': spec['sha256']['capture_manifest'], 'prompt_count': 1,
                                          'round_count':1})
            self.artifacts.update({str(child): adapter.sha256(child), str(prompt): adapter.sha256(prompt)})
        self.index = self.prep / 'train-providers.json'
        self.index.write_text(json.dumps({'split': 'train', 'training_eligible': True, 'shards': self.records}))
        self.source['execution_manifest_sha256'] = adapter.sha256(self.index)
        cp = self.prep / 'checkpoints/step-000000000000-e000000-r000000000000'; cp.mkdir(parents=True)
        (cp/'resume.pt').write_bytes(b'synthetic-zero-state')
        exports = {}
        for lane in ('A8','A1'):
            (cp/lane).mkdir(); exports[lane] = {}
            for name in ('joint.npz','joint.json'):
                (cp/lane/name).write_bytes((lane+name).encode()); exports[lane][name] = adapter.sha256(cp/lane/name)
        manifest = {'step':0,'epoch':0,'cursor':0,'exports':exports}
        (cp/'manifest.json').write_text(json.dumps(manifest))
        checkpoint = {'path':str(cp/'resume.pt'),'sha256':adapter.sha256(cp/'resume.pt'),'step':0}
        self.ready = {'teacher_coverage': {'source':self.source,'unique_train_prompts':2,'unique_supervised_rows':20},
                      'checkpoint':checkpoint}
        self.ready_path=self.prep/'preparation-ready.json'; self.ready_path.write_text(json.dumps(self.ready))
        self.ready_sha=adapter.sha256(self.ready_path)
        for file in cp.rglob('*'):
            if file.is_file():self.artifacts[str(file)]=adapter.sha256(file)
        self.artifacts[str(self.index)] = adapter.sha256(self.index)
        self.binding={'source_sha256':self.digest(self.source),'prepared_ready_sha256':self.ready_sha,
                      'provider_manifest':str(self.index),'artifacts':self.artifacts}
        self.api=SimpleNamespace(prepared_digest=self.digest,
             prepared_corpus_inputs=lambda *args:(self.binding,self.index,{'split':'development'}))
        self.spec={'coverage':{'min_unique_train_prompts':2,'min_unique_supervised_rows':20}}
        self.auth=adapter.authenticate(self.api,self.spec,self.root/'new',self.prep,self.ready_sha)
        self.calls=[]
        def factory(config,path):
            spec=json.loads(Path(path).read_text());ordinal=int(Path(path).stem.split('-')[-1]);self.calls.append(ordinal)
            return SimpleNamespace(training_eligible=True,full_body_qat_eligible=True,data_split='train',
                allowed_prompt_ids={f'train-{ordinal}'},hashes=spec['sha256'],capture_id=spec['capture_id'],
                d2t_offsets=(0,1),target_vocab_size=2,draft_vocab_size=2,max_depth=5,base_gguf_sha256='a'*64,
                total_rounds=1,
                rounds=lambda:iter([Batch(ordinal)]),load_models_cpu=lambda:(_ for _ in ()).throw(AssertionError('Model load')))
        self.factory=factory

    @staticmethod
    def digest(value):
        import hashlib
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def test_lazy_provider_authenticates_full_source_without_loading_children(self):
        provider=adapter.PreparedProvider(self.auth,object(),self.factory)
        self.assertEqual(self.calls,[]);self.assertEqual(provider.source_metadata,self.source)
        self.assertEqual(provider.allowed_prompt_ids,{'train-0','train-1'})
        self.assertEqual(list(provider.bounded_rounds()),[Batch(0,0)])
        self.assertEqual(self.calls,[0]);self.assertEqual(list(provider.rounds()),[Batch(0,0),Batch(1,1)])
        self.assertEqual(self.calls,[0,1])

    def test_selected_child_remains_full_authenticated_source(self):
        provider=adapter.PreparedProvider(self.auth,object(),self.factory,1)
        self.assertEqual(list(provider.bounded_rounds()),[Batch(1,1)])
        self.assertEqual(provider.source_metadata,self.source);self.assertEqual(self.calls,[1])

    def test_out_of_range_case_refuses(self):
        with self.assertRaises(ValueError): adapter.PreparedProvider(self.auth,object(),self.factory,2)

    def test_changed_child_bytes_refuse_before_factory(self):
        Path(self.records[0]['provider_manifest']).write_text('{}')
        provider=adapter.PreparedProvider(self.auth,object(),self.factory)
        with self.assertRaisesRegex(ValueError,'SHA differs'):list(provider.bounded_rounds())
        self.assertEqual(self.calls,[])

    def test_child_membership_mismatch_is_not_full_ready_flag(self):
        def wrong(config,path):
            child=self.factory(config,path);child.allowed_prompt_ids={'unrelated'};return child
        provider=adapter.PreparedProvider(self.auth,object(),wrong)
        with self.assertRaisesRegex(ValueError,'membership'):list(provider.bounded_rounds())

    def test_ready_digest_or_full_source_mutation_refuses(self):
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            adapter.authenticate(self.api,self.spec,self.root/'new',self.prep,'0'*64)
        self.binding['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'full source'):
            adapter.authenticate(self.api,self.spec,self.root/'new',self.prep,self.ready_sha)

    def test_atomic_copy_preserves_all_bytes_and_does_not_copy_latest(self):
        new=self.root/'new';new.mkdir();(self.prep/'latest.json').write_text('{"old_path":true}')
        report=launch.copy_zero(self.auth,new);dest=Path(report['publication'])
        for relative,digest in report['sha256'].items():self.assertEqual(adapter.sha256(dest/relative),digest)
        self.assertFalse((new/'latest.json').exists());self.assertTrue((self.prep/'latest.json').exists())
        self.assertFalse(any(path.is_symlink() for path in dest.rglob('*')))
        with self.assertRaises(FileExistsError):launch.copy_zero(self.auth,new)

    def test_checkpoint_mutation_and_unknown_inventory_refuse(self):
        cp=Path(self.ready['checkpoint']['path']).parent
        (cp/'extra').write_bytes(b'not-owned')
        with self.assertRaisesRegex(ValueError,'unknown'):launch.checkpoint_inventory(self.auth)
        (cp/'extra').unlink();(cp/'resume.pt').write_bytes(b'changed')
        new=self.root/'new';new.mkdir()
        with self.assertRaisesRegex(ValueError,'SHA differs'):launch.copy_zero(self.auth,new)

    def test_regular_file_cache_detects_same_length_restored_mtime_and_symlink(self):
        path=self.root/'file';path.write_bytes(b'old');files=adapter.Files();digest=adapter.sha256(path)
        files.check(path,digest);before=path.stat();path.write_bytes(b'new')
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'SHA differs'):files.check(path,digest)
        target=self.root/'target';target.write_bytes(b'old');path.unlink();path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'canonical'):files.check(path,digest)

    def test_missing_actual_gates_refuses_before_source_or_model(self):
        args=SimpleNamespace(allow_cuda=True,admission=None,admission_sha256=None)
        with patch.object(launch,'source_api',side_effect=AssertionError('Unexpected source/model load')):
            with self.assertRaisesRegex(ValueError,'complete gate admission'):launch.start(args)

    def test_plan_is_stdlib_only_and_never_imports_torch(self):
        command=[sys.executable,'-c',
          "import runpy,sys;sys.path.insert(0,'scripts');sys.argv=['script','--plan'];runpy.run_path('scripts/train_prepared_continuous_w1ax.py',run_name='__main__');assert 'torch' not in sys.modules"]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout);self.assertFalse(report['optimizer_allowed']);self.assertFalse(report['model_loaded'])

    def test_start_without_existing_model_native_receipt_refuses_before_import(self):
        path=self.root/'admission.json';path.write_text(json.dumps({'sole_gpu_owner':True}))
        args=SimpleNamespace(allow_cuda=True,admission=path,admission_sha256=adapter.sha256(path))
        with patch.object(launch,'prepare',side_effect=AssertionError('No model/source work')):
            with self.assertRaisesRegex(ValueError,'Missing gate optimization_readiness'):
                launch.start(args)

    def admission(self):
        config=self.root/'config.json';config.write_text('{}')
        receipt=self.root/'measured.json';receipt.write_text('{}')
        locator={'path':str(receipt),'sha256':adapter.sha256(receipt)}
        source=self.root/'source';(source/'scripts').mkdir(parents=True)
        producer=source/'scripts/check_qat_optimization_readiness.py';producer.write_text('# synthetic producer')
        evidence=self.root/'orchestration.json'
        evidence.write_text(json.dumps({'schema':'prepared_measurement_orchestration_v1',
            'orchestration_sha256':launch.code_identity(),'prepared_ready_sha256':self.ready_sha,
            'receipt':locator,'producer':'readiness','source_commit':adapter.SOURCE_COMMIT,
            'native_commit':adapter.NATIVE_COMMIT,'optimizer_updates':0,'receipt_bytes_modified':False,
            'producer_sha256':adapter.sha256(producer)}))
        args=SimpleNamespace(config=config,stages_manifest=None,selected_shard=0,
            source_checkout=source,prepared_run_dir=self.prep,
            prepared_ready_sha256=self.ready_sha,run_dir=self.root/'new',
            admission=self.root/'admission.json')
        record={'schema':'prepared_continuous_launch_admission_v1',
            'orchestration_sha256':launch.code_identity(),'source_commit':adapter.SOURCE_COMMIT,
            'native_commit':adapter.NATIVE_COMMIT,'prepared_ready_sha256':self.ready_sha,
            'inputs':launch.input_pins(args),'run_dir':str(args.run_dir),
            'granted_unix':100,'expires_unix':200,'sole_gpu_owner':True,
            'rtx5080_pause_requested':False,'optimization_readiness':locator,
            'measured_orchestration':{'path':str(evidence),'sha256':adapter.sha256(evidence)}}
        def publish():
            args.admission.write_text(json.dumps(record));args.admission_sha256=adapter.sha256(args.admission)
        publish()
        return args,record,publish

    def test_admission_expires_and_cancel_is_rechecked(self):
        args,record,publish=self.admission()
        with patch.object(launch.time,'time',return_value=150):launch.launch_admission(args)
        with patch.object(launch.time,'time',return_value=201):
            with self.assertRaisesRegex(ValueError,'Fresh exclusive'):launch.launch_admission(args)
        args.run_dir.mkdir();(args.run_dir/'CANCEL').touch()
        with patch.object(launch.time,'time',return_value=150):
            with self.assertRaisesRegex(ValueError,'cancelled'):launch.launch_admission(args)

    def test_changed_config_and_shard_refuse_same_admission(self):
        args,record,publish=self.admission()
        args.selected_shard=1
        with patch.object(launch.time,'time',return_value=150):
            with self.assertRaisesRegex(ValueError,'Specific current'):launch.launch_admission(args)
        args.selected_shard=0;args.config.write_text('{"changed_budget":true}')
        with patch.object(launch.time,'time',return_value=150):
            with self.assertRaisesRegex(ValueError,'Specific current'):launch.launch_admission(args)

    def test_expired_admission_refuses_before_source_or_cuda(self):
        args,record,publish=self.admission();args.allow_cuda=True
        with patch.object(launch.time,'time',return_value=201), \
             patch.object(launch,'prepare',side_effect=AssertionError('Source/CUDA work')):
            with self.assertRaisesRegex(ValueError,'Fresh exclusive'):launch.start(args)

    def test_receipt_sidecar_mismatch_refuses(self):
        args,record,publish=self.admission()
        record['optimization_readiness']['sha256']='0'*64;publish()
        with patch.object(launch.time,'time',return_value=150):
            with self.assertRaisesRegex(ValueError,'SHA differs'):launch.launch_admission(args)

    def test_prepared_environment_preserves_actual_hardware_validation_and_math(self):
        expected={'cuda_math':{'float32_matmul_precision':'highest',
                 'matmul_allow_tf32':False,'cudnn_allow_tf32':True}}
        calls=[]
        torch=SimpleNamespace(set_float32_matmul_precision=lambda x:calls.append(x),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)),
                                     cudnn=SimpleNamespace(allow_tf32=False)))
        api=SimpleNamespace(torch=torch,training_runtime_identity=lambda device:expected)
        def original(api,config,spec):
            calls.append('hardware-validated');return {'actual':'cuda'}
        environment=measured.prepared_environment(expected,original)
        self.assertEqual(environment(api,SimpleNamespace(device='cuda:0'),{}),{'actual':'cuda'})
        self.assertEqual(calls,['hardware-validated','highest'])
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32);self.assertTrue(torch.backends.cudnn.allow_tf32)
        api.training_runtime_identity=lambda device:{'different':'runtime'}
        with self.assertRaisesRegex(ValueError,'runtime differs'):
            environment(api,SimpleNamespace(device='cuda:0'),{})


if __name__=='__main__':unittest.main()
