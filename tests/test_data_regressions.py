"""Small, CPU-only checks for temporal boundaries and dataset configuration."""

import unittest

import numpy as np
import pandas as pd
import torch

from data.data_pipeline import (
    add_segment_id_opp,
    divide_features_labels,
    filter_realdisp,
    interpolation_pam,
    opp_file_id,
    sliding_window_wrapper_group,
)
from data.OPPORTUNITY_data import make_loaders_OPP
from data.PAMAP2_data import make_loaders_PAM
from data.REALWORLD_data import make_loaders_RW
from data.REALDISP_data import make_loaders_REALDISP


class InterpolationBoundaryTests(unittest.TestCase):
    @staticmethod
    def frame(values, *, labels=None, groups=None, timestamps=None):
        n = len(values)
        return pd.DataFrame({
            0: values,
            "ts": np.arange(n) * 0.01 if timestamps is None else timestamps,
            1: np.ones(n, dtype=int) if labels is None else labels,
            "group_id": ["S1"] * n if groups is None else groups,
        })

    def test_interpolates_short_contiguous_chunk_and_keeps_metadata(self):
        frame = self.frame([0.0, np.nan, np.nan, 3.0])
        original = frame.copy(deep=True)
        result = interpolation_pam(frame)
        np.testing.assert_allclose(result[0], [0.0, 1.0, 2.0, 3.0])
        pd.testing.assert_frame_equal(result.drop(columns=0), frame.drop(columns=0))
        pd.testing.assert_frame_equal(frame, original)

    def test_rejects_label_change_hidden_between_matching_endpoints(self):
        result = interpolation_pam(self.frame([0.0, np.nan, 2.0], labels=[1, 2, 1]))
        self.assertEqual(result[0].tolist(), [0.0, 2.0])

    def test_rejects_subject_change_hidden_between_matching_endpoints(self):
        result = interpolation_pam(self.frame([0.0, np.nan, 2.0], groups=["S1", "S2", "S1"]))
        self.assertEqual(result[0].tolist(), [0.0, 2.0])

    def test_rejects_internal_restart_despite_positive_endpoint_delta(self):
        frame = self.frame(
            [0.0, 1.0, np.nan, np.nan, 4.0, 5.0, 6.0],
            timestamps=[0.0, 0.01, 0.02, 0.005, 0.015, 0.025, 0.035],
        )
        self.assertEqual(interpolation_pam(frame)[0].tolist(), [0.0, 1.0, 4.0, 5.0, 6.0])

    def test_rejects_internal_gap_using_resampler_contiguity_threshold(self):
        frame = self.frame(
            [0.0, 1.0, np.nan, np.nan, 4.0, 5.0],
            timestamps=[0.0, 0.01, 0.02, 0.04, 0.05, 0.06],
        )
        self.assertEqual(interpolation_pam(frame)[0].tolist(), [0.0, 1.0, 4.0, 5.0])

    def test_keeps_long_and_unbounded_chunks_uninterpolated(self):
        frame = self.frame([np.nan, 1.0, np.nan, np.nan, np.nan, 5.0, np.nan])
        self.assertEqual(interpolation_pam(frame, max_gap=2)[0].tolist(), [1.0, 5.0])

    def test_rejects_known_pam_activity_boundary(self):
        frame = self.frame(
            [0.0, 1.0, np.nan, np.nan, np.nan, 5.0, 6.0],
            labels=[3, 3, 3, 4, 4, 4, 4],
            timestamps=[0.0, 0.01, 0.02, 1447.34, 1447.35, 1447.36, 1447.37],
        )
        self.assertEqual(interpolation_pam(frame)[0].tolist(), [0.0, 1.0, 5.0, 6.0])


class DatasetBoundaryTests(unittest.TestCase):
    def test_opp_windows_cannot_cross_removed_rows(self):
        positions = np.r_[np.arange(85), np.arange(95, 200)]
        frame = pd.DataFrame({0: positions, 1: 1, "group_id": "S1-ADL1"}, index=positions)
        segmented = add_segment_id_opp(frame)
        self.assertEqual(set(opp_file_id(segmented["group_id"])), {"S1-ADL1"})
        features, labels = divide_features_labels(segmented)
        windows, window_labels = sliding_window_wrapper_group(features, labels, 90, 30)
        self.assertEqual(windows.shape, (1, 90, 1))
        np.testing.assert_array_equal(windows[0, :, 0], np.arange(95, 185))
        np.testing.assert_array_equal(window_labels, [1])

    def test_realdisp_sensor_order_and_channel_counts(self):
        raw = pd.DataFrame(np.tile(np.arange(120, dtype=float), (2, 1)))
        raw[0] = [10.0, 10.0]
        raw[1] = [0.0, 20000.0]
        raw[119] = [1, 2]
        raw["group_id"] = "S1"
        for sensors, starts in [(6, [28, 15, 2, 41, 54, 67]), (5, [28, 15, 2, 41, 54]), (3, [2, 28, 67])]:
            with self.subTest(sensors=sensors):
                selected = filter_realdisp(raw, keep_timestamp=True, keep_sensors=sensors)
                expected = np.concatenate([np.arange(start, start + 9) for start in starts])
                np.testing.assert_array_equal(selected.iloc[0, :9 * sensors].to_numpy(), expected)
                np.testing.assert_allclose(selected["ts"], [10.0, 10.02])
                np.testing.assert_array_equal(selected[9 * sensors], [1, 2])

    def test_worker_limits_and_shared_label_mapping_for_all_datasets(self):
        features = np.zeros((4, 90, 9), dtype=np.float32)
        labels = np.array([1, 2, 3, 4])
        for factory in (make_loaders_OPP, make_loaders_PAM, make_loaders_RW, make_loaders_REALDISP):
            with self.subTest(factory=factory.__name__):
                train, val, test, encoder = factory(
                    features, labels, features, labels, features, labels,
                    generator=torch.Generator().manual_seed(1),
                )
                self.assertEqual([loader.num_workers for loader in (train, val, test)], [2, 0, 0])
                self.assertTrue(train.persistent_workers)
                self.assertFalse(val.persistent_workers)
                self.assertFalse(test.persistent_workers)
                np.testing.assert_array_equal(encoder.classes_, labels)


if __name__ == "__main__":
    unittest.main()
