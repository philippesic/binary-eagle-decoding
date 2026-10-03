#!/usr/bin/env python3
"""Pinned orchestration of source6f measurements under authenticated prepared math.

Pass producer arguments after --. Native actions are bootstrap, execute-decision,
and collect. Run each decision through this wrapper; the collector's generated
unwrapped subprocess recipe is retained as evidence but is not dispatched here.
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from prepared_continuous_provider import Files, NATIVE_COMMIT, SOURCE_COMMIT, require, sha256, source_api
from train_prepared_continuous_w1ax import code_identity, prepare


def binding_inputs(path, digest):
    files = Files()
    record = files.read(path, digest)
    require(record.get('schema') == 'prepared_provider_binding_v1'
            and record.get('orchestration_sha256') == code_identity(), 'Prepared wrapper code binding differs')
    # This MUST precede any checker, collector, provider or Torch import.
    api = source_api(record['source_checkout'])
    config_path = files.check(record['config']['path'], record['config']['sha256'])
    stage = record.get('stages_manifest')
    stage_path = files.check(stage['path'], stage['sha256']) if stage else None
    measurement = record.get('measurement_config')
    require(isinstance(measurement, dict) and set(measurement) == {'path', 'sha256'},
            'Pinned separate measurement config required')
    measurement_path = files.check(measurement['path'], measurement['sha256'])
    original, _ = api.load_config(config_path)
    measured = files.read(measurement_path, measurement['sha256'])
    expected = dict(original)
    # Native actor declaration belongs to measurement orchestration, not frozen
    # capture/provider declarations or the copied checkpoint configuration.
    expected['native'] = {'expected_commit': NATIVE_COMMIT}
    require(measured == expected, 'Measurement config changes prepared declarations or recipe')
    args = argparse.Namespace(source_checkout=Path(record['source_checkout']), config=config_path,
                              stages_manifest=stage_path,
                              prepared_run_dir=Path(record['prepared_run_dir']),
                              prepared_ready_sha256=record['prepared_ready_sha256'],
                              selected_shard=record['selected_shard'],
                              run_dir=Path(record['validation_run_dir']))
    _, _, config, auth = prepare(args)
    return api, config, auth, record, measurement_path


def prepared_environment(expected_runtime, original_environment):
    def environment(api, config, spec):
        # Retain the producer's actual CUDA availability/configured-hardware
        # validation, then apply the authenticated math BEFORE any measurement.
        hardware = original_environment(api, config, spec)
        frozen = expected_runtime['cuda_math']
        api.torch.set_float32_matmul_precision(frozen['float32_matmul_precision'])
        api.torch.backends.cuda.matmul.allow_tf32 = frozen['matmul_allow_tf32']
        api.torch.backends.cudnn.allow_tf32 = frozen['cudnn_allow_tf32']
        require(api.training_runtime_identity(config.device) == expected_runtime,
                'Measurement runtime differs from actual prepared checkpoint')
        return hardware
    return environment


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--binding', required=True, type=Path)
    cli.add_argument('--binding-sha256', required=True)
    cli.add_argument('--producer', required=True, choices=('readiness', 'native'))
    cli.add_argument('producer_args', nargs=argparse.REMAINDER)
    args = cli.parse_args(argv)
    forwarded = args.producer_args[1:] if args.producer_args[:1] == ['--'] else args.producer_args
    api, config, auth, record, measured_path = binding_inputs(args.binding, args.binding_sha256)
    common = importlib.import_module('check_qat_optimization_readiness')
    require(Path(common.__file__).resolve() == Path(record['source_checkout']).resolve() / 'scripts/check_qat_optimization_readiness.py',
            'Readiness producer escaped pinned source')
    common.cuda_environment = prepared_environment(auth['ready']['training_runtime'], common.cuda_environment)
    producer = common if args.producer == 'readiness' else importlib.import_module('collect_qat_native_evidence')
    require(Path(producer.__file__).resolve().is_relative_to(Path(record['source_checkout']).resolve()),
            'Measurement producer escaped pinned source')
    producer_args = producer.parser().parse_args(forwarded)
    require(producer_args.allow_cuda and producer_args.output is not None,
            'Prepared measurement wrapper requires an explicit actual producer action')
    require(not getattr(producer_args, 'curriculum_config', None), 'Prepared reference recipe has no curriculum')
    require(not getattr(producer_args, 'run_native', False),
            'Run each native decision through this wrapper; unwrapped child recipe is not admitted')
    if args.producer == 'native':
        require(producer_args.bootstrap or producer_args.collect or producer_args.execute_decision,
                'Explicit native bootstrap, decision or collection required')
    configured = getattr(producer_args, 'config', None)
    if configured is not None:
        require(Path(configured).resolve() == measured_path, 'Producer measurement config differs')
    elif args.producer == 'native':
        context_path = producer_args.bootstrap_context
        require(context_path is not None, 'Native action needs authenticated bootstrap context')
        context = common.read_json(context_path)
        require(context.get('config') == record['measurement_config']
                and context.get('training_runtime_base') == auth['ready']['training_runtime'],
                'Native bootstrap changes prepared runtime/config')
    else:
        raise ValueError('Readiness requires pinned measurement config')
    if args.producer == 'readiness' or getattr(producer_args, 'collect', False):
        require(producer_args.provider == 'prepared_continuous_provider:create_provider'
                and Path(producer_args.provider_manifest).resolve() == Path(auth['binding']['provider_manifest']),
                'Measurement must use authenticated full prepared provider')
    os.environ['QAT_PREPARED_BINDING_PATH'] = str(args.binding.resolve())
    os.environ['QAT_PREPARED_BINDING_SHA256'] = args.binding_sha256
    # The decision action bypasses recipe_context; initialize its authentic math
    # and verify the full runtime explicitly before unchanged run_gate executes.
    if getattr(producer_args, 'execute_decision', None):
        common.cuda_environment(common.runtime_api(), config, common.read_json(measured_path))
    result = producer.main(forwarded)
    require(result == 0, 'Actual producer failed; preserve its failure artifact')
    output = producer_args.output.resolve()
    receipt_path = output / ('readiness.json' if args.producer == 'readiness' else 'result.json')
    receipt = {'path': str(receipt_path), 'sha256': sha256(receipt_path)}
    evidence = {'schema': 'prepared_measurement_orchestration_v1', 'producer': args.producer,
                'producer_sha256': sha256(producer.__file__), 'producer_args': forwarded,
                'orchestration_sha256': code_identity(), 'source_commit': SOURCE_COMMIT,
                'native_commit': NATIVE_COMMIT,
                'prepared_ready_sha256': record['prepared_ready_sha256'],
                'provider_binding': {'path': str(args.binding.resolve()), 'sha256': args.binding_sha256},
                'measurement_config': record['measurement_config'],
                'source_sha256': auth['binding']['source_sha256'],
                'training_runtime': auth['ready']['training_runtime'], 'receipt': receipt,
                'optimizer_updates': 0, 'receipt_bytes_modified': False}
    common.write_new(output / 'prepared-orchestration.json', evidence)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
