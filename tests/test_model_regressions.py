"""CPU contract tests for JEPA and probe models.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_model_regressions.py' -v

The forward tests use a zero-block, nonfused backbone to exercise the actual
tokenizer, predictor, decoder, losses, and EMA without CUDA-only Mamba kernels.
The seed test constructs a real Mamba2 block but does not run its CUDA forward.
These tests do not substitute for a GPU training smoke test.
"""

import math
import unittest

import torch
from torch import nn

from Mamba_blocks import _init_weights
from MambaSSL_JEPA_Model import (
    DecoderModel,
    HARMambaConfig,
    MambaDownstreamClassifier,
    MambaJEPA,
    masking_algorithm_targets,
)


def cpu_config(**overrides):
    values = dict(
        d_model=24,
        d_intermediate=32,
        n_layer=0,
        num_sensor_features=9,
        rms_norm=False,
        fused_add_norm=False,
    )
    values.update(overrides)
    return HARMambaConfig(**values)


class ModelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(812)

    def test_both_residual_outputs_receive_reference_depth_scaling(self):
        for name in ("out_proj", "fc2"):
            with self.subTest(branch=name):
                module = nn.Module()
                setattr(module, name, nn.Linear(17, 13, bias=False))
                torch.manual_seed(718)
                _init_weights(module, n_layer=24, n_residuals_per_layer=2)
                actual = getattr(module, name).weight.detach()
                expected = torch.empty_like(actual)
                torch.manual_seed(718)
                nn.init.kaiming_uniform_(expected, a=math.sqrt(5))
                expected /= math.sqrt(48)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_masks_preserve_visible_tokens_and_have_disjoint_spans(self):
        tokens = torch.randn(16, 18, 24)
        for seed in range(8):
            with self.subTest(seed=seed):
                masked, mask, blocks = masking_algorithm_targets(
                    tokens, mask_ratio=0.33, t_l=3,
                    generator=torch.Generator().manual_seed(seed),
                )
                self.assertEqual(len(blocks), 2)
                self.assertTrue(torch.equal(mask.sum(dim=1), torch.full((16,), 6)))
                self.assertEqual(torch.count_nonzero(masked[mask]).item(), 0)
                torch.testing.assert_close(masked[~mask], tokens[~mask])
                for batch in range(len(tokens)):
                    positions = torch.cat([block[batch] for block in blocks])
                    self.assertEqual(positions.unique().numel(), 6)
                    self.assertTrue(mask[batch, positions].all())
                    for block in blocks:
                        self.assertEqual(block.shape, (16, 3))
                        self.assertTrue((block[batch].diff() == 1).all())

    def test_jepa_targets_reconstruction_and_gradient_contract(self):
        for drop in (False, True):
            with self.subTest(drop=drop):
                model = MambaJEPA(
                    cpu_config(), mask_ratio=0.33, t_l=3,
                    num_heads=3, drop=drop, recon=True,
                )
                x = torch.randn(4, 90, 9)
                full_target, targets, context, mask, blocks, predictions, raw = model(x)
                self.assertEqual(full_target.shape, (4, 18, 24))
                self.assertEqual(targets.shape, (8, 3, 24))
                self.assertEqual(predictions.shape, targets.shape)
                self.assertEqual(context.shape, (4, 12 if drop else 18, 24))
                for block_number, block in enumerate(blocks):
                    for batch in range(len(x)):
                        torch.testing.assert_close(
                            targets[block_number * len(x) + batch],
                            full_target[batch, block[batch]],
                        )
                reconstruction = model.decoder(raw)
                sample_mask = mask.repeat_interleave(5, dim=1)
                self.assertEqual(reconstruction.shape, x.shape)
                self.assertEqual(sample_mask.sum().item(), 120)
                loss = (
                    0.1 * nn.SmoothL1Loss()(predictions, targets)
                    + nn.SmoothL1Loss()(reconstruction[sample_mask], x[sample_mask])
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
                self.assertFalse(targets.requires_grad)
                for branch in (model.context_encoder, model.predictor, model.decoder):
                    self.assertTrue(any(
                        p.grad is not None and torch.count_nonzero(p.grad) > 0
                        for p in branch.parameters()
                    ))

    def test_target_starts_as_frozen_copy_and_ema_updates_it(self):
        model = MambaJEPA(cpu_config(), num_heads=3, use_pe=True)
        for context, target in zip(
            model.context_encoder.parameters(), model.target_encoder.parameters()
        ):
            torch.testing.assert_close(context, target, rtol=0, atol=0)
            self.assertFalse(target.requires_grad)
        old_target = [p.detach().clone() for p in model.target_encoder.parameters()]
        old_pe = model.pe_target.clone()
        with torch.no_grad():
            for context in model.context_encoder.parameters():
                context.add_(0.7)
            model.pe.add_(0.3)
        model.update_target_encoder(0.8)
        for previous, context, target in zip(
            old_target, model.context_encoder.parameters(), model.target_encoder.parameters()
        ):
            torch.testing.assert_close(target, 0.8 * previous + 0.2 * context)
        torch.testing.assert_close(model.pe_target, 0.8 * old_pe + 0.2 * model.pe)

    def test_no_reconstruction_branch_returns_none(self):
        model = MambaJEPA(cpu_config(), num_heads=3, recon=False, drop=True)
        result = model(torch.randn(2, 90, 9))
        self.assertIsNone(model.decoder)
        self.assertIsNone(result[-1])

    def test_decoder_preserves_time_then_feature_order(self):
        model = DecoderModel(cpu_config(d_model=10, num_sensor_features=2))
        model.decoder = nn.Identity()
        patches = torch.arange(30, dtype=torch.float32).reshape(1, 3, 10)
        decoded = model(patches)
        self.assertEqual(decoded.shape, (1, 15, 2))
        for timestep in range(15):
            self.assertEqual(decoded[0, timestep].tolist(), [2 * timestep, 2 * timestep + 1])

    def test_frozen_encoder_and_separate_head_restore_exact_predictions(self):
        config = cpu_config()
        model = MambaDownstreamClassifier(config, num_classes=4)
        model.encoder.requires_grad_(False)
        encoder_state = {key: value.clone() for key, value in model.encoder.state_dict().items()}
        x = torch.randn(4, 90, 9)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=0.001
        )
        model.train()
        loss = nn.CrossEntropyLoss()(model(x), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        optimizer.step()
        for key, value in encoder_state.items():
            torch.testing.assert_close(model.encoder.state_dict()[key], value, rtol=0, atol=0)
        restored = MambaDownstreamClassifier(config, num_classes=4)
        restored.encoder.load_state_dict(encoder_state, strict=True)
        head_state = {
            key.removeprefix("classifier."): value
            for key, value in model.state_dict().items()
            if key.startswith("classifier.")
        }
        restored.classifier.load_state_dict(head_state, strict=True)
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_encoder_and_probe_seeds_are_independent(self):
        config = cpu_config(
            n_layer=1,
            ssm_cfg={"expand": 2, "layer": "Mamba2", "headdim": 8, "d_ssm": 48},
        )

        def build(encoder_seed, probe_seed):
            torch.manual_seed(encoder_seed)
            model = MambaDownstreamClassifier(config, num_classes=4)
            torch.manual_seed(probe_seed)
            for part in model.classifier.modules():
                if hasattr(part, "reset_parameters"):
                    part.reset_parameters()
            return model

        first, other_probe, other_encoder = build(42, 11), build(42, 22), build(58, 11)
        self.assertTrue(all(
            torch.equal(a, b) for a, b in zip(first.encoder.parameters(), other_probe.encoder.parameters())
        ))
        self.assertTrue(any(
            not torch.equal(a, b) for a, b in zip(first.encoder.parameters(), other_encoder.encoder.parameters())
        ))
        self.assertTrue(all(
            torch.equal(a, b) for a, b in zip(first.classifier.parameters(), other_encoder.classifier.parameters())
        ))
        self.assertTrue(any(
            not torch.equal(a, b) for a, b in zip(first.classifier.parameters(), other_probe.classifier.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
