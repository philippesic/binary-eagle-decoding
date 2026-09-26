"""Audit this project's nine-linear F16 -> Q1_0 control, without inference.

Run with the pinned llama.cpp gguf-py directory on PYTHONPATH and NumPy.
Uses the public Q1_0 reference's sequential F32 absolute sum and F16 scales.
Raw JSON and model files belong in ignored results directories, not Git.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(source, candidate, quantizer):
    original = {t.name: t for t in GGUFReader(source).tensors}
    packed = {t.name: t for t in GGUFReader(candidate).tensors}
    assert set(original) == set(packed), 'Tensor inventory changed'
    records = []
    preserved = 0
    for name, tensor in original.items():
        result = packed[name]
        assert np.array_equal(tensor.shape, result.shape), name
        if tensor.tensor_type.name != 'F16':
            assert tensor.tensor_type == result.tensor_type, name
            assert np.array_equal(tensor.data, result.data), name
            preserved += 1
            continue
        assert len(tensor.shape) == 2 and result.tensor_type.name == 'Q1_0', name
        sign_errors = scale_errors = 0
        max_error = 0.0
        for first in range(0, tensor.data.shape[0], 64):
            values = tensor.data[first:first+64].astype(np.float32).reshape(-1, 128)
            blocks = result.data[first:first+64].reshape(-1, 18)
            expected = np.packbits(values >= 0, axis=1, bitorder='little')
            sign_errors += int(np.count_nonzero(expected != blocks[:, 2:]))
            sums = np.cumsum(np.abs(values), axis=1, dtype=np.float32)[:, -1]
            scales = (sums / np.float32(128)).astype(np.float16)
            observed = blocks[:, :2].copy().view('<f2').ravel()
            scale_errors += int(np.count_nonzero(observed.view('u2') != scales.view('u2')))
            max_error = max(max_error, float(np.max(np.abs(
                observed.astype(np.float32) - scales.astype(np.float32)))))
        records.append(dict(tensor=name, shape=tensor.shape.tolist(),
                            type=result.tensor_type.name, bytes=result.n_bytes,
                            sign_byte_mismatches=sign_errors,
                            scale_word_mismatches=scale_errors,
                            max_scale_error=max_error))
    return dict(
        kind='CPU export/source-row audit only; no inference or speed result',
        source=str(source), source_sha256=sha(source),
        artifact=str(candidate), artifact_sha256=sha(candidate),
        artifact_bytes=candidate.stat().st_size,
        quantizer_binary_sha256=sha(quantizer), records=records,
        preserved_nonmatrix_tensors=preserved,
        reference='F16 source -> sequential F32 abs sum/128 -> FP16 scale; sign>=0 little-endian pack',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--quantizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.candidate, args.quantizer)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    assert len(result['records']) == 9, 'Expected the nine selected drafter linears'
    assert all(r['sign_byte_mismatches'] == r['scale_word_mismatches'] == 0
               for r in result['records']), 'Source-row audit failed; inspect raw JSON'
    print('PASS: nine matrices; zero sign-byte/scale-word mismatches; nonmatrix tensors preserved')
