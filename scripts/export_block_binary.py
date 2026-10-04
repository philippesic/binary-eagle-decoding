#!/usr/bin/env python3
"""Atomic FFN15(+fusion) DSpark/DFlash row-W1Ax checkpoint export.

Manifest v1: schema_version, family(dspark/dflash), profile(ffn15/ffn15_fusion),
base_gguf_sha256, checkpoint_sha256, activation_bits(1/8), projections mapping
GGUF base (without .weight) -> {checkpoint_name, shape:[rows,K]}. NPZ contains
exactly F32 <checkpoint_name>.latent[rows,K] and .scale[rows]>=0. Fixed native
quantizers only. No dense selected shadows, head borrowing or row permutation.
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Type
from export_recurrent_binary import sha256, raw_hash, pack, same_tensor

PREFIX = 'dflash.w1ax.'
FFN_BASES = {f'blk.{i}.ffn_{part}' for i in range(5) for part in ('gate', 'up', 'down')}

def check_manifest(manifest: dict, base_hash: str, checkpoint_hash: str) -> dict:
    required = {'schema_version', 'family', 'profile', 'base_gguf_sha256',
                'checkpoint_sha256', 'activation_bits', 'projections'}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError('block manifest requires exact v1 fields')
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('unsupported block manifest version')
    if manifest['family'] not in ('dspark','dflash') or manifest['profile'] not in ('ffn15','ffn15_fusion'):
        raise ValueError('unsupported family/profile')
    if type(manifest['activation_bits']) is not int or manifest['activation_bits'] not in (1,8):
        raise ValueError('block activation bits must be 1 or 8')
    if manifest['base_gguf_sha256'] != base_hash or manifest['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('source/checkpoint SHA256 mismatch')
    expected = FFN_BASES | ({'fc'} if manifest['profile'] == 'ffn15_fusion' else set())
    entries = manifest['projections']
    if not isinstance(entries,dict) or set(entries) != expected:
        raise ValueError('block projection coverage must be exact fifteen FFN plus optional fusion')
    names = []
    for base,item in entries.items():
        if not isinstance(item,dict) or set(item) != {'checkpoint_name','shape'}:
            raise ValueError('invalid projection fields')
        shape = item['shape']
        if not isinstance(shape,list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('invalid projection shape')
        if not isinstance(item['checkpoint_name'],str) or not item['checkpoint_name']:
            raise ValueError('invalid checkpoint name')
        names.append(item['checkpoint_name'])
    if len(set(names)) != len(names):
        raise ValueError('duplicate checkpoint names')
    return entries

def export_model(base_path: Path, checkpoint: Path, manifest_path: Path, output: Path) -> dict:
    base_path,checkpoint,manifest_path,output = map(Path,(base_path,checkpoint,manifest_path,output))
    if output.exists(): raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    base_hash,checkpoint_hash = sha256(base_path),sha256(checkpoint)
    entries = check_manifest(manifest,base_hash,checkpoint_hash)
    reader = GGUFReader(base_path)
    if reader.fields['general.architecture'].contents() != 'dflash':
        raise ValueError('base architecture must be dflash')
    if any(key.startswith(PREFIX) for key in reader.fields):
        raise ValueError('base must be original dense block GGUF')
    for key, expected in {'dflash.block_count':5,'dflash.block_size':7,'dflash.sample_from_anchor':True}.items():
        if key not in reader.fields or reader.fields[key].contents() != expected:
            raise ValueError(f'unsupported block layout metadata: {key}')
    tensors = {t.name:t for t in reader.tensors}
    if len(tensors) != len(reader.tensors) or any('.w1a1_' in n or '.w1ax_' in n for n in tensors):
        raise ValueError('duplicate/prepacked base tensors')
    if not {'token_embd.weight','output.weight','fc.weight'} <= set(tensors) or 'd2t' in tensors:
        raise ValueError('private full vocabulary embedding/head required')
    has_markov = 'markov_w1.weight' in tensors
    if has_markov != (manifest['family'] == 'dspark'):
        raise ValueError('family differs from original Markov inventory')
    arrays = {}
    required = {item['checkpoint_name']+s for item in entries.values() for s in ('.latent','.scale')}
    with np.load(checkpoint,allow_pickle=False) as archive:
        if len(archive.files) != len(required) or set(archive.files) != required:
            raise ValueError('checkpoint contains missing/extra latent-scale arrays')
        for base,item in entries.items():
            source = tensors.get(base+'.weight')
            shape = tuple(item['shape'])
            if source is None or tuple(int(n) for n in source.shape) != shape[::-1]:
                raise ValueError(f'{base}: source shape mismatch')
            name = item['checkpoint_name']
            latent,scale = archive[name+'.latent'],archive[name+'.scale']
            if latent.dtype != np.float32 or latent.shape != shape or not np.isfinite(latent).all():
                raise ValueError(f'{base}: invalid latent')
            if scale.dtype != np.float32 or scale.shape != (shape[0],) or not np.isfinite(scale).all() or (scale<0).any():
                raise ValueError(f'{base}: invalid nonnegative F32 row scales')
            arrays[base] = {'packed':pack(latent),'scale':np.ascontiguousarray(scale),
                            'latent_sha256':raw_hash(latent),'scale_sha256':raw_hash(scale)}
    preserved = {n:t for n,t in tensors.items() if n.removesuffix('.weight') not in entries}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.block-export-',dir=output.parent) as directory:
        temporary = Path(directory)/'model.gguf'
        writer = GGUFWriter(temporary,'dflash')
        for key,field in reader.fields.items():
            if not key.startswith('GGUF.') and key != 'general.architecture':
                writer.add_key_value(key,field.contents(),field.types[0],field.types[-1] if len(field.types)>1 else None)
        writer.add_uint32(PREFIX+'version',1)
        writer.add_uint32(PREFIX+'activation_bits',manifest['activation_bits'])
        for key,value in {'profile':manifest['profile'],'bit_order':'little','sign_rule':'nonnegative_is_one','scale_rule':'f32_learned_nonnegative'}.items():
            writer.add_string(PREFIX+key,value)
        writer.add_array(PREFIX+'tensors',sorted(base+'.weight' for base in entries))
        for tensor in reader.tensors:
            base = tensor.name.removesuffix('.weight')
            if base in entries:
                writer.add_tensor(base+'.w1a1_packed',arrays[base]['packed'],raw_dtype=Type.I32)
                writer.add_tensor(base+'.w1a1_scale',arrays[base]['scale'],raw_dtype=Type.F32)
            else: writer.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        reread = GGUFReader(temporary)
        observed = {t.name:t for t in reread.tensors}
        expected_names = set(preserved) | {base+s for base in entries for s in ('.w1a1_packed','.w1a1_scale')}
        if len(observed) != len(reread.tensors) or set(observed) != expected_names:
            raise ValueError('export tensor inventory mismatch')
        for name,tensor in preserved.items():
            if not same_tensor(tensor,observed[name]): raise ValueError(f'protected tensor changed: {name}')
        for base,pair in arrays.items():
            for kind,dtype in (('packed',Type.I32),('scale',Type.F32)):
                tensor = observed[base+'.w1a1_'+kind]
                if tensor.tensor_type != dtype or tensor.data.shape != pair[kind].shape or raw_hash(tensor.data) != raw_hash(pair[kind]):
                    raise ValueError(f'packed roundtrip mismatch: {base}')
        for key,field in reader.fields.items():
            if not key.startswith('GGUF.') and reread.fields[key].contents() != field.contents():
                raise ValueError(f'protected metadata changed: {key}')
        report = {'schema':'block_binary_export_v1','base_gguf':{'sha256':base_hash},
                  'checkpoint':{'sha256':checkpoint_hash},'manifest':{'sha256':sha256(manifest_path)},
                  'output':{'path':str(output),'sha256':sha256(temporary)},'family':manifest['family'],
                  'profile':manifest['profile'],'activation_bits':manifest['activation_bits'],
                  'serialization_audit_passed':True,'native_runtime_gate':'pending',
                  'projections':{base:{'shape':entries[base]['shape'],'latent_sha256':p['latent_sha256'],
                                      'scale_sha256':p['scale_sha256'],'packed_sha256':raw_hash(p['packed'])} for base,p in arrays.items()},
                  'preserved_tensors':{n:{'type':t.tensor_type.name,'raw_sha256':raw_hash(t.data)} for n,t in preserved.items()}}
        temporary.rename(output)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('base','checkpoint','manifest','output','audit'): parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    if args.audit.exists() or args.audit.resolve() in {p.resolve() for p in (args.base,args.checkpoint,args.manifest,args.output)}:
        parser.error('audit must be a distinct new file')
    report=export_model(args.base,args.checkpoint,args.manifest,args.output)
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['output']))
if __name__=='__main__': main()
