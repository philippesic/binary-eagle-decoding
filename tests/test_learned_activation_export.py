"""Strict learned scalar metadata and byte-preserving weight export checks."""
import copy
import json
import unittest

import numpy as np

from test_recurrent_binary_export import RecurrentBinaryExportTests
from export_recurrent_binary import (
    LEARNED_ACTIVATION_RULE, QUANTIZER_BOUNDARIES, QUANTIZER_PREFIX,
    GGUFReader, check_activation_quantizers, sha256,
)


class LearnedActivationExportTests(RecurrentBinaryExportTests):
    # Reuse the synthetic nine-projection fixture, but avoid rerunning its tests.
    def learned_manifest(self, bits=4):
        for name in self.arrays:
            if name.endswith('.scale') and self.arrays[name].ndim == 2:
                self.arrays[name] = self.arrays[name][:, 0].copy()
        self.save_checkpoint()
        return {
            'schema_version': 3, 'base_gguf_sha256': sha256(self.base),
            'checkpoint_sha256': sha256(self.checkpoint), 'scale_layout': 'row',
            'activation_bits': bits, 'activation_rule': LEARNED_ACTIVATION_RULE,
            'weight_rule': 'hard_sign_zero_positive_clipped_identity_ste',
            'qk_row_order': 'original_checkpoint',
            'export_status': 'row_w1ax_requires_native_validation', 'objective': 'hard_ce',
            'projections': {base: {'checkpoint_name': name, 'shape': list(self.shapes[base])}
                            for base, name in __import__('export_recurrent_binary').SOURCE_NAMES.items()},
            'activation_quantizers': {'version': 1, 'boundaries': {
                boundary: {'bits': bits, 'threshold_delta': float(np.float32(.25 if bits == 1 else 0)),
                           'clip_ratio': float(np.float32(1 if bits == 1 else .75))}
                for boundary in QUANTIZER_BOUNDARIES}},
        }

    def test_learned_scalar_round_trip(self):
        for bits in (1, 4, 8):
            manifest = self.learned_manifest(bits)
            self.manifest.write_text(json.dumps(manifest))
            report = self.export()
            self.assertEqual(report['activation_quantizers'], manifest['activation_quantizers'])
            reader = GGUFReader(self.output)
            self.assertEqual(reader.fields[QUANTIZER_PREFIX+'version'].contents(), 1)
            self.assertEqual(reader.fields[QUANTIZER_PREFIX+'boundaries'].contents(), list(QUANTIZER_BOUNDARIES))
            for boundary, params in manifest['activation_quantizers']['boundaries'].items():
                for key in ('threshold_delta', 'clip_ratio'):
                    self.assertEqual(reader.fields[QUANTIZER_PREFIX+boundary+'.'+key].contents(), params[key])
            self.output.unlink()

    def test_fail_closed_parameters(self):
        valid = self.learned_manifest()['activation_quantizers']
        for key, value in (('clip_ratio', float('nan')), ('clip_ratio', 0), ('clip_ratio', 1.1),
                           ('clip_ratio', 0.1), ('threshold_delta', .25), ('bits', 8),
                           ('clip_ratio', [1])):
            case = copy.deepcopy(valid)
            case['boundaries']['qkv'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_activation_quantizers(case, 4)
        for mutation in (
            lambda x: x['boundaries'].pop('head'),
            lambda x: x['boundaries'].update(extra=x['boundaries']['head']),
            lambda x: x['boundaries']['qkv'].update(extra=1),
            lambda x: x.update(version=2),
        ):
            case = copy.deepcopy(valid); mutation(case)
            with self.assertRaises(ValueError): check_activation_quantizers(case, 4)
        with self.assertRaises(ValueError): check_activation_quantizers(valid, 16)


# Inherited fixture methods are useful; inherited test cases are covered separately.
for name in list(RecurrentBinaryExportTests.__dict__):
    if name.startswith('test_') and name not in LearnedActivationExportTests.__dict__:
        setattr(LearnedActivationExportTests, name, None)

if __name__ == '__main__': unittest.main()
