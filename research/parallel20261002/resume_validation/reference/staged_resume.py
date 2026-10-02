"""Owned CPU research prototype; source-bound, never installed in live jobs.

stage_resume takes exclusive ownership of an independently deserialized payload.
The caller must not mutate it after staging. Core tensors back candidate weights;
the module graph is copied as metadata, not as a second model allocation. CUDA
adoption needs a device-transfer admission check and a fresh runtime identity.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from w1a1_eagle.continuous_qat import restore_rng, sha256
from w1a1_eagle.qat_curriculum import CurriculumState
from w1a1_eagle.qat_curriculum_runner import SCHEMA
from w1a1_eagle.recurrent_qat import joint_parameter_families

CONTROL = Path('/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json')


def check_control():
    d = json.loads(CONTROL.read_text())
    if (d.get('research_stop') or d.get('monitor_interrupt') or d.get('reset_observed')
            or d['last_reset_unix'] != 1791049896 or time.time() >= 1791049896
            or 100 - d['last_weekly_used_percent'] <= 1):
        raise SystemExit('Research control requires checkpoint and stop')


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_option(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_option(a, b) for a, b in zip(left, right))
    return left == right


def named_layout(linears, optimizer):
    """Actual joint constructor order, including sorted binary recipe families."""
    families = joint_parameter_families(linears)
    names = {}
    for path, module in linears.items():
        for name, parameter in module.named_parameters():
            names.setdefault(id(parameter), f'{path}.{name}')
    expected_families = [name for name, values in families.items() if values]
    if [g.get('family') for g in optimizer.param_groups] != expected_families:
        raise ValueError('joint optimizer family order differs')
    result, seen = [], set()
    for group in optimizer.param_groups:
        family = group['family']
        params = group['params']
        if {id(p) for p in params} != {id(p) for p in families[family]}:
            raise ValueError('joint optimizer ownership differs')
        ordered = families[family]
        if family in ('sign', 'scale') and getattr(optimizer, '_binary_recipe', None) is not None:
            attribute = 'latent_sign' if family == 'sign' else 'scale_offset'
            ordered = [getattr(linears[path], attribute) for path in sorted(linears)]
        if [id(p) for p in params] != [id(p) for p in ordered]:
            raise ValueError('joint optimizer within-family order differs')
        for p in params:
            if id(p) in seen:
                raise ValueError('aliased joint optimizer ownership')
            seen.add(id(p))
            result.append({'name': names[id(p)], 'family': family,
                           'shape': list(p.shape), 'dtype': str(p.dtype)})
    return result


def validate_optimizer_state(linears, optimizer, saved, base_lrs, state,
                             model_phase, warmup_updates, participation=None):
    layout = named_layout(linears, optimizer)
    if type(optimizer) not in (torch.optim.AdamW, torch.optim.SGD):
        raise ValueError('unsupported actual joint optimizer')
    if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('optimizer checkpoint inventory differs')
    groups = saved['param_groups']
    expected = optimizer.state_dict()['param_groups']
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError('optimizer family inventory differs')
    if (not isinstance(base_lrs, list) or len(base_lrs) != len(expected)
            or any(not _same_option(rate, group['lr']) for rate, group in zip(base_lrs, expected))):
        raise ValueError('checkpoint optimizer base rates differ')
    updates = state.phases[model_phase]['updates']
    factor = min(1.0, state.global_updates / max(1, warmup_updates)) if updates else 1.0
    for actual, wanted, base in zip(groups, expected, base_lrs):
        if not isinstance(actual, dict) or set(actual) != set(wanted):
            raise ValueError('optimizer group inventory differs')
        ids = actual['params']
        if (not isinstance(ids, list) or any(type(i) is not int for i in ids)
                or ids != wanted['params']):
            raise ValueError('optimizer owned IDs/order differ')
        if any(not _same_option(actual[k], wanted[k]) for k in wanted if k not in ('params', 'lr')):
            raise ValueError('immutable optimizer options differ')
        if not _finite_number(actual['lr']) or actual['lr'] != base * factor:
            raise ValueError('optimizer effective warmup LR differs')
    params = [p for g in optimizer.param_groups for p in g['params']]
    ids = [i for g in groups for i in g['params']]
    entries = saved['state']
    if (not isinstance(entries, dict) or any(type(i) is not int for i in entries)
            or set(entries) - set(ids)):
        raise ValueError('optimizer contains unowned state')
    if not updates and entries:
        raise ValueError('fresh phase must have empty optimizer state')
    by_id = dict(zip(ids, params))
    group_by_id = {i: g for g in groups for i in g['params']}
    for i, values in entries.items():
        group = group_by_id[i]
        required = ({'step', 'exp_avg', 'exp_avg_sq'} if type(optimizer) is torch.optim.AdamW
                    else {'momentum_buffer'} if group['momentum'] else set())
        if not isinstance(values, dict) or set(values) != required or not required:
            raise ValueError('optimizer moment inventory differs')
        for key, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.device.type != 'cpu' or value.requires_grad
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('optimizer moments must be detached finite dense CPU tensors')
            if key == 'step':
                # Matches Adam._init_group/_get_scalar_dtype in admitted Torch.
                scalar_dtype = (torch.float64 if torch.get_default_dtype() == torch.float64
                                else torch.float32)
                if (value.shape != torch.Size([]) or value.dtype != scalar_dtype
                        or float(value) < 0 or float(value) != int(float(value))
                        or float(value) > updates):
                    raise ValueError('invalid optimizer per-parameter step')
            elif value.shape != by_id[i].shape or value.dtype != by_id[i].dtype:
                raise ValueError('optimizer moment shape/dtype differs')
            if key == 'exp_avg_sq' and bool((value < 0).any()):
                raise ValueError('negative optimizer second moment')
    # Optional future save receipt. Absent state is otherwise legal with grad=None.
    if participation is not None:
        if (not isinstance(participation, dict)
                or set(participation) != {x['name'] for x in layout}
                or any(type(n) is not int or not 0 <= n <= updates
                       for n in participation.values())):
            raise ValueError('invalid participation receipt')
        for i, item in zip(ids, layout):
            n = participation[item['name']]
            has_state = type(optimizer) is torch.optim.AdamW or group_by_id[i]['momentum'] > 0
            if has_state and ((n > 0) != (i in entries)):
                raise ValueError('moment presence differs from participation receipt')
            if type(optimizer) is torch.optim.AdamW and n and int(entries[i]['step']) != n:
                raise ValueError('Adam step differs from participation receipt')
    return layout


def _module_shell(module, memo):
    """Copy nn.Module containers; keep tensor storage shared until rebound."""
    if id(module) in memo:
        return memo[id(module)]
    result = copy.copy(module)
    memo[id(module)] = result
    result._parameters = dict(module._parameters)
    result._buffers = dict(module._buffers)
    result._modules = {k: None if v is None else _module_shell(v, memo)
                       for k, v in module._modules.items()}
    return result


def _validate_core(runner, model):
    if not isinstance(model, dict) or set(model) != {
            'linears', 'activation_bank', 'fusion_correction', 'affine_bank'}:
        raise ValueError('model checkpoint inventory differs')
    core = runner._core_states()
    if not isinstance(model['linears'], dict) or set(model['linears']) != set(core):
        raise ValueError('checkpoint projection inventory differs')
    live_storages = {t.untyped_storage().data_ptr() for t in runner.drafter.state_dict().values()
                     if isinstance(t, torch.Tensor)}
    for path, wanted in core.items():
        values = model['linears'][path]
        if not isinstance(values, dict) or set(values) != set(wanted):
            raise ValueError('checkpoint core inventory differs')
        for key, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.device.type != 'cpu' or value.requires_grad
                    or value.shape != wanted[key].shape or value.dtype != wanted[key].dtype
                    or not bool(torch.isfinite(value).all())
                    or value.untyped_storage().data_ptr() in live_storages):
                raise ValueError('invalid or live-aliased core checkpoint tensor')
            if key in ('initial_scale', 'frozen_bias') and not torch.equal(value, wanted[key]):
                raise ValueError('checkpoint changed frozen binary operands')
            if key == 'latent_sign' and bool((value.abs() > 1).any()):
                raise ValueError('checkpoint latent outside projected interval')
        if bool((values['initial_scale'] + values['scale_offset'] < 0).any()):
            raise ValueError('checkpoint scales outside projected bounds')


def _validate_rng(value):
    if not isinstance(value, dict) or set(value) != {'python', 'numpy', 'torch'}:
        raise ValueError('CPU RNG inventory differs')
    try:
        random.Random().setstate(value['python'])
        np.random.RandomState().set_state(value['numpy'])
        t = value['torch']
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.uint8
                or t.device.type != 'cpu' or t.layout != torch.strided or t.ndim != 1):
            raise ValueError('invalid Torch RNG tensor')
        torch.Generator(device='cpu').set_state(t)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError('invalid checkpoint RNG') from error


@dataclass
class StagedResume:
    live: object
    candidate: object
    payload: dict
    pointer: dict
    committed: bool = False

    def commit(self):
        """No loader/validation/allocation remains in the publication section.

        Single-thread/process resume contract. Not a crash-atomic multi-object
        swap; process failure still recovers from the unchanged atomic file.
        """
        if self.committed:
            raise ValueError('staged resume already committed')
        fields = ('state', 'model_phase', 'qat', 'provider', 'drafter', 'linears', 'bank',
                  'adapter', 'optimizer', 'base_lrs', 'cursor', 'epoch', 'occupancy_seconds',
                  'smoke_seconds', 'transition_seconds', 'unique_rows', 'unique_prompts', 'rng')
        for field in fields:
            setattr(self.live, field, getattr(self.candidate, field))
        restore_rng(self.live.rng, self.live.device)
        self.live.last_checkpoint = self.pointer
        self.live._accounted_at = self.candidate._accounted_at
        self.committed = True


def stage_resume(runner, payload, pointer):
    check_control()
    if torch.device(runner.device).type != 'cpu':
        raise ValueError('research transactional prototype is CPU-only')
    if payload.get('schema') != SCHEMA or payload.get('contract') != runner.contract:
        raise ValueError('resume changes config/source/math/runtime contract')
    candidate = copy.copy(runner)
    candidate.state = CurriculumState(runner.curriculum, data_contract=runner.source,
                                     model_contract=runner.contract)
    candidate.state.load_state_dict(payload['state'])
    phase = payload['model_phase']
    if (type(phase) is not int or not 0 <= phase < len(runner.curriculum.stages)
            or phase not in (candidate.state.phase_index, candidate.state.phase_index - 1)
            or pointer.get('schema') != SCHEMA or pointer.get('model_phase') != phase
            or type(pointer.get('model_phase')) is not int
            or type(pointer.get('global_updates')) is not int
            or pointer.get('global_updates') != candidate.state.global_updates
            or len(candidate.state.transitions) != phase):
        raise ValueError('checkpoint model/phase/transition ancestry differs')
    _validate_core(runner, payload['model'])
    candidate.drafter = _module_shell(runner.drafter, {})
    candidate.linears = {p: candidate.drafter.get_submodule(p) for p in runner.linears}
    for path, module in candidate.linears.items():
        for key, value in payload['model']['linears'][path].items():
            if key in module._parameters:
                module._parameters[key] = torch.nn.Parameter(value, requires_grad=True)
            else:
                module._buffers[key] = value
        for name in ('fusion_correction', 'affine_binary'):
            if name in module._modules:
                module._modules[name] = copy.deepcopy(module._modules[name])
    # Rebind the registered canonical alias BEFORE NativeStepAdapter freezes
    # unapproved parameters. A shell of the old bank still points at live
    # midpoint objects through its Python mapping and would freeze them.
    if hasattr(candidate.drafter, 'qat_affine_bank'):
        candidate.drafter.qat_affine_bank = candidate._affine_bank()
    candidate._bind_phase(phase)
    candidate.model_phase = phase
    candidate._load_model(payload['model'])
    validate_optimizer_state(candidate.linears, candidate.optimizer, payload['optimizer'],
                             payload['base_lrs'], candidate.state, phase,
                             runner.config.warmup_updates, payload.get('participation'))
    candidate.optimizer.load_state_dict(payload['optimizer'])
    for name in ('cursor', 'epoch'):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError('invalid checkpoint provider cursor')
        setattr(candidate, name, payload[name])
    for name in ('occupancy_seconds', 'smoke_seconds', 'transition_seconds'):
        if not _finite_number(payload[name]) or payload[name] < 0:
            raise ValueError('invalid checkpoint timing')
        setattr(candidate, name, payload[name])
    durable = pointer.get('occupancy_seconds')
    if not _finite_number(durable) or durable < candidate.occupancy_seconds:
        raise ValueError('invalid checkpoint residency accounting')
    for name in ('unique_rows', 'unique_prompts'):
        values = payload[name]
        if (not isinstance(values, list) or any(type(x) is not str for x in values)
                or len(set(values)) != len(values)):
            raise ValueError('invalid checkpoint unique inventory')
        setattr(candidate, name, set(values))
    _validate_rng(payload['rng'])
    candidate.rng = payload['rng']
    stamp = runner.clock()
    reconstruction_tail = 0 if runner._accounted_at is None else stamp - runner._accounted_at
    if not _finite_number(reconstruction_tail) or reconstruction_tail < 0:
        raise ValueError('invalid synchronized reconstruction clock')
    candidate.occupancy_seconds = durable + runner.occupancy_seconds + reconstruction_tail
    if candidate.occupancy_seconds > sum(s.gpu_seconds for s in runner.curriculum.stages):
        raise ValueError('resume reconstruction exceeds total residency budget')
    candidate._accounted_at = stamp
    return StagedResume(runner, candidate, payload, pointer)


def resume_staged(runner):
    """Pointer/hash checks precede the side-effect-free CPU staging entry point."""
    check_control()
    status_path = runner.run_dir / 'status.json'
    if status_path.is_file() and json.loads(status_path.read_text()).get('budget_failed') is True:
        raise ValueError('budget-overrun run requires an explicit new experiment budget')
    pointer = json.loads((runner.run_dir / 'latest.json').read_text())
    if (pointer.get('schema') != SCHEMA or type(pointer.get('file')) is not str
            or Path(pointer['file']).name != pointer['file']):
        raise ValueError('invalid atomic checkpoint pointer')
    path = runner.run_dir / 'checkpoints' / pointer['file']
    if sha256(path) != pointer.get('sha256'):
        raise ValueError('checkpoint payload hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    staged = stage_resume(runner, payload, pointer)
    staged.commit()
    return staged
