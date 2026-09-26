"""Read-only BF16 weight-geometry diagnostic; not an acceptance or timing test.

Uses deterministic evenly spaced output rows and standard-library safetensors
reading. Does not load a model, activations, prompts, or a GPU. Raw JSON belongs
outside Git. Groupwise ternary is an exact weight-L2 oracle, not Prism's method.
"""

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def cast(value, code):
    return struct.unpack('<' + code, struct.pack('<' + code, value))[0]


def binary_sse(magnitudes, scale_code=None):
    n = len(magnitudes)
    total = math.fsum(magnitudes)
    scale = total / n
    if scale_code:
        scale = cast(scale, scale_code)
    # Direct residuals avoid cancellation near a perfect representation.
    return math.fsum((v - scale) ** 2 for v in magnitudes), scale


def ternary_sse(magnitudes):
    ordered = sorted(magnitudes, reverse=True)
    prefix = 0.0
    best = 0.0
    count = 0
    for n, value in enumerate(ordered, 1):
        prefix += value
        explained = prefix * prefix / n
        if explained > best:
            best = explained
            count = n
    return max(0.0, math.fsum(v * v for v in magnitudes) - best), count


def analyze(path, rows_per_tensor, group_size):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    lookup = [struct.unpack('<f', struct.pack('<I', i << 16))[0]
              for i in range(65536)]
    results = []
    with path.open('rb') as f:
        header_length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_length))
        base = 8 + header_length
        for name, info in header.items():
            if name == '__metadata__' or info.get('dtype') != 'BF16':
                continue
            if len(info['shape']) != 2:
                continue
            m, k = info['shape']
            assert k % group_size == 0, (name, k)
            nrows = min(rows_per_tensor, m)
            indices = sorted({round(i * (m - 1) / max(1, nrows - 1))
                              for i in range(nrows)})
            natural = 2560 if name == 'fc.weight' or name in (
                'midlayer.self_attn.q_proj.weight',
                'midlayer.self_attn.k_proj.weight',
                'midlayer.self_attn.v_proj.weight') else None
            totals = dict(energy=0.0, row_f32_sse=0.0, group_ideal_sse=0.0,
                          group_f16_sse=0.0, ternary_ideal_sse=0.0,
                          natural_group_sse=0.0, ternary_active=0, elements=0)
            cvs = []
            gains = []
            for row in indices:
                f.seek(base + info['data_offsets'][0] + 2 * row * k)
                codes = array.array('H')
                codes.frombytes(f.read(2 * k))
                if sys.byteorder != 'little':
                    codes.byteswap()
                values = [abs(lookup[code]) for code in codes]
                assert all(math.isfinite(v) for v in values)
                energy = math.fsum(v * v for v in values)
                row_sse, _ = binary_sse(values, 'f')
                totals['energy'] += energy
                totals['row_f32_sse'] += row_sse
                totals['elements'] += k
                if natural:
                    totals['natural_group_sse'] += math.fsum(
                        binary_sse(values[start:start+natural], 'e')[0]
                        for start in range(0, k, natural))
                scales = []
                local_group_sse = 0.0
                for start in range(0, k, group_size):
                    group = values[start:start + group_size]
                    loss, scale = binary_sse(group)
                    totals['group_ideal_sse'] += loss
                    group_f16_sse = binary_sse(group, 'e')[0]
                    totals['group_f16_sse'] += group_f16_sse
                    local_group_sse += group_f16_sse
                    loss, active = ternary_sse(group)
                    totals['ternary_ideal_sse'] += loss
                    totals['ternary_active'] += active
                    scales.append(scale)
                if row_sse:
                    gains.append(1-local_group_sse/row_sse)
                mean = math.fsum(scales) / len(scales)
                if mean:
                    cvs.append(math.sqrt(math.fsum((a-mean)**2 for a in scales)
                                          / len(scales)) / mean)
            energy = totals['energy']
            row_loss = totals['row_f32_sse']
            gains.sort()
            results.append(dict(
                tensor=name, shape=[m, k], sampled_rows=len(indices),
                row_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                row_binary_nmse=totals['row_f32_sse']/energy,
                group_binary_nmse=totals['group_ideal_sse']/energy,
                group_f16_binary_nmse=totals['group_f16_sse']/energy,
                group_ternary_l2_oracle_nmse=totals['ternary_ideal_sse']/energy,
                binary_residual_reduction=1-totals['group_f16_sse']/row_loss,
                row_gain_p10_p50_p90=[gains[round(q*(len(gains)-1))]
                                      for q in (0.1, 0.5, 0.9)],
                natural_group_width=natural,
                natural_group_nmse=totals['natural_group_sse']/energy if natural else None,
                mean_within_row_scale_cv=math.fsum(cvs)/len(cvs),
                ternary_active_fraction=totals['ternary_active']/totals['elements'],
                totals=totals,
            ))
    return dict(
        method='deterministic evenly spaced rows; weight L2 only',
        model_sha256=digest, rows_per_tensor=rows_per_tensor, group_size=group_size,
        scale_contract='row F32; group binary FP16; ternary oracle unrounded optimal scale',
        exclusions='no activations, covariance, teacher logits, acceptance, training, or timing',
        tensors=results,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.rows > 0 and args.group_size > 0
    result = analyze(args.weights, args.rows, args.group_size)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for t in result['tensors']:
        print(t['tensor'], 'row_nmse', round(t['row_binary_nmse'], 5),
              'g128_nmse', round(t['group_f16_binary_nmse'], 5),
              'residual_reduction_pct', round(100*t['binary_residual_reduction'], 3),
              'ternary_oracle_nmse', round(t['group_ternary_l2_oracle_nmse'], 5))
