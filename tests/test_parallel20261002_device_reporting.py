"""Exact CPU helper census and device routing without accelerator allocation."""

import unittest
from unittest.mock import patch

import torch

from experiments.parallel20261002.device_aware_reporting.proof import helper_census, step_proof
from w1a1_eagle import recurrent_qat as qat


class RoutedScalar:
    """Device metadata stub with a real CPU scalar payload, never accelerator memory."""

    is_cpu = False

    def __init__(self, value, device):
        self.payload = value
        self.device = torch.device(device)
        self.dtype = value.dtype
        self.detaches = 0

    def detach(self):
        self.detaches += 1
        return self

    def item(self):
        raise AssertionError("non-CPU reporting must use the grouped path")


class DeviceReportingTests(unittest.TestCase):
    def test_cpu_exact_values_types_order_without_stack_or_transfer(self):
        values = {"bool": torch.tensor(True),
                  "wide": torch.tensor(2**53 + 137, dtype=torch.int64),
                  "f16": torch.tensor(0.5, dtype=torch.float16, requires_grad=True),
                  "bf16": torch.tensor(1.25, dtype=torch.bfloat16),
                  "f32": torch.tensor(2.5, dtype=torch.float32),
                  "f64": torch.tensor(3.75 + 2**-40, dtype=torch.float64),
                  "native_bool": False, "native_int": 7, "native_float": 0.25}
        expected = {key: value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in values.items()}
        with patch.object(torch, "stack", side_effect=AssertionError("CPU stack")), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU transfer")), \
                patch.object(torch.Tensor, "detach", side_effect=AssertionError("CPU detach")):
            actual = qat._reporting_scalars(values)
        self.assertEqual(list(actual), list(values))
        self.assertEqual(actual, expected)
        for key in expected:
            self.assertIs(type(actual[key]), type(expected[key]))
        self.assertIsNone(values["f16"].grad)

    def test_mixed_device_dtype_routing_with_cpu_payload_stubs(self):
        values = {
            "cuda0_f32_a": RoutedScalar(torch.tensor(0.5), "cuda:0"),
            "cpu_count": torch.tensor(2**53 + 137, dtype=torch.int64),
            "cuda1_f32": RoutedScalar(torch.tensor(1.25), "cuda:1"),
            "cuda0_f32_b": RoutedScalar(torch.tensor(2.5), "cuda:0"),
            "cuda0_int": RoutedScalar(torch.tensor(2**53 + 139), "cuda:0"),
            "cuda0_f64": RoutedScalar(torch.tensor(3.75 + 2**-40, dtype=torch.float64),
                                      "cuda:0"),
            "mps_bool": RoutedScalar(torch.tensor(True), "mps"),
            "native": 7,
        }
        groups = []
        real_stack = torch.stack

        def stack_cpu_payloads(group):
            self.assertTrue(all(isinstance(value, RoutedScalar) for value in group))
            groups.append([(str(value.device), value.dtype) for value in group])
            return real_stack([value.payload for value in group])

        with patch.object(qat, "Tensor", (torch.Tensor, RoutedScalar)), \
                patch.object(torch, "stack", stack_cpu_payloads):
            actual = qat._reporting_scalars(values)
        expected = {name: value.payload.item() if isinstance(value, RoutedScalar)
                    else value.item() if isinstance(value, torch.Tensor) else value
                    for name, value in values.items()}
        self.assertEqual(list(actual), list(values))
        self.assertEqual(actual, expected)
        for key in expected:
            self.assertIs(type(actual[key]), type(expected[key]))
        self.assertEqual(groups, [
            [("cuda:0", torch.float32)] * 2,
            [("cuda:1", torch.float32)], [("cuda:0", torch.int64)],
            [("cuda:0", torch.float64)], [("mps", torch.bool)],
        ])
        self.assertTrue(all(value.detaches == 1 for value in values.values()
                            if isinstance(value, RoutedScalar)))

    def test_actual_cpu_dispatch_census_has_no_allocation(self):
        helper_census()

    def test_fixed_actual_step_exact_metrics_state_gradients_and_moments(self):
        step_proof(False)

    def test_learned_affine_actual_step_exact_metrics_state_gradients_and_moments(self):
        step_proof(True)

    def test_empty_and_python_only_metrics(self):
        self.assertEqual(qat._reporting_scalars({}), {})
        self.assertEqual(qat._reporting_scalars({"bool": True, "count": 2**53 + 17}),
                         {"bool": True, "count": 2**53 + 17})


if __name__ == "__main__":
    unittest.main()
