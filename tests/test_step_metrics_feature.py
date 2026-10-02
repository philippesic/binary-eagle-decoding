"""Exact actual source step and operation census checks on tiny CPU fixtures."""

import unittest

from experiments.parallel20261002.step_metrics_feature.probe import comparison


class StepMetricsFeatureTests(unittest.TestCase):
    def test_fixed_actual_source_metrics_state_and_census(self):
        comparison(False)

    def test_optional_actual_source_metrics_state_and_census(self):
        comparison(True)
