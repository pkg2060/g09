"""CPU synthetic contracts for the independent output head; no training claim.

Run: python -m src.test_bounded_output -v
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .config import get_config
from .model import ScaledSigmoid, build_model, canonical_model_config
from .trainer import Trainer, load_trusted_checkpoint


class BoundedOutputContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model(self, margin=.1, activation='scaled_sigmoid', legacy=False):
        cfg = get_config()['model']
        cfg.update(window=10, dropout=0., ff_dim=32, dtype='float64',
                   output_activation=activation, output_margin=margin)
        if legacy:
            cfg.pop('output_margin')
        return build_model(2, cfg).eval()

    def cpu_snapshot_engine(self, model):
        """Exercise real checkpoint serialization with CPU-only synthetic state."""
        engine = Trainer.__new__(Trainer)  # CUDA-only training ctor is not executed.
        engine.model, engine.device = model, torch.device('cpu')
        engine.cfg = get_config()['train']
        engine.optimizer = torch.optim.AdamW(model.parameters(), lr=engine.cfg['lr'])
        engine.scheduler = torch.optim.lr_scheduler.StepLR(engine.optimizer, step_size=5, gamma=.9)
        engine.active_epoch = engine.batches_completed_in_active_epoch = engine.bad_epochs = 0
        engine.completed_epochs = engine.epochs_consumed = engine.best_epoch = 10
        engine.target_epochs, engine.max_epochs = 11, 30
        engine.model_id, engine.started_utc = 'cpu-contract-only', '2026-09-21T00:00:00+00:00'
        engine.best_model_state = copy.deepcopy(model.state_dict())
        engine.best_val_mae = .123
        engine.history = [{'epoch': 10, 'val_mae': .123}]
        engine.scaler_state = {'synthetic_cpu_contract_only': True}
        engine.feature_names = ['A.PV', 'B.PV']
        engine.loader_generator_states = {}
        engine.gpu_evidence = {'execution': 'CPU synthetic checkpoint contract; no GPU training'}
        return engine

    def save_snapshot(self, model, legacy=False):
        engine = self.cpu_snapshot_engine(model)
        with patch('torch.cuda.get_rng_state_all', return_value=[]):
            checkpoint = engine._checkpoint('synthetic_cpu_contract')
        if legacy:
            checkpoint['model_cfg'].pop('output_margin')
        path = self.root/'last.pt'
        torch.save(checkpoint, path)
        return path, checkpoint

    def test_exact_formula_and_analytic_gradient(self):
        for margin in (0., .1, .25):
            with self.subTest(margin=margin):
                logits = torch.tensor([-5., -1., 0., 1., 5.], dtype=torch.float64, requires_grad=True)
                head = ScaledSigmoid(margin)
                output = head(logits)
                sigmoid = torch.sigmoid(logits.detach())
                torch.testing.assert_close(output, (1 + 2*margin)*sigmoid-margin, rtol=0, atol=0)
                output.sum().backward()
                torch.testing.assert_close(logits.grad, (1 + 2*margin)*sigmoid*(1-sigmoid))
                self.assertEqual(list(head.parameters()), [])
                self.assertEqual(dict(head.state_dict()), {})

    def test_both_decoders_share_bounded_head_and_gradient(self):
        for margin in (.1, .25):
            with self.subTest(margin=margin):
                model = self.model(margin)
                x = torch.rand(3, 10, 2, dtype=torch.float64, requires_grad=True)
                uses = []
                handle = model.fcn.register_forward_hook(lambda module, args, result: uses.append(module))
                first, second = model(x)
                handle.remove()
                self.assertEqual(uses, [model.fcn, model.fcn])
                for output in (first, second):
                    self.assertEqual(tuple(output.shape), (3, 2))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue((output >= -margin).all() and (output <= 1+margin).all())
                # Decoder 2 also backpropagates through the unchanged decoder-1 focus path.
                second.square().mean().backward()
                for block in (model.decoder1, model.decoder2, model.fcn):
                    gradient = sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None)
                    self.assertGreater(gradient, 0.)
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(float(x.grad.abs().sum()), 0.)

    def test_checkpoint_roundtrip_preserves_margin_and_both_outputs(self):
        for margin in (.1, .25):
            with self.subTest(margin=margin):
                model = self.model(margin)
                x = torch.rand(4, 10, 2, dtype=torch.float64)
                with torch.inference_mode():
                    expected = model(x)
                path, _ = self.save_snapshot(model)
                checkpoint = load_trusted_checkpoint(path)
                # The identical factory path is used by pipeline.eval and ExplanationEngine.
                restored = build_model(len(checkpoint['feature_names']), checkpoint['model_cfg']).eval()
                restored.load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.assertEqual(checkpoint['model_cfg']['output_margin'], margin)
                self.assertEqual(checkpoint['source_info']['selected_output_margin'], margin)
                self.assertEqual(checkpoint['source_info']['selected_output_interval'], [-margin, 1+margin])
                with torch.inference_mode():
                    actual = restored(x)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_different_margin_resume_is_rejected_before_weights_load(self):
        path, _ = self.save_snapshot(self.model(.1))
        destination = self.cpu_snapshot_engine(self.model(.25))
        with patch.object(destination.model, 'load_state_dict', side_effect=AssertionError('must reject first')):
            with self.assertRaisesRegex(ValueError, 'architecture differs'):
                destination._restore(path)

    def test_same_margin_resume_restores_exact_state(self):
        source = self.model(.1)
        path, _ = self.save_snapshot(source)
        destination = self.cpu_snapshot_engine(self.model(.1))
        # Only the CUDA assertion is bypassed for this CPU serialization contract.
        with patch('src.trainer._assert_model_device'), patch('torch.cuda.set_rng_state_all'):
            destination._restore(path)
        self.assertEqual(destination.epochs_consumed, 10)
        self.assertEqual(destination.model.config['output_margin'], .1)
        for key, tensor in source.state_dict().items():
            torch.testing.assert_close(destination.model.state_dict()[key], tensor, rtol=0, atol=0)

    def test_legacy_activation_formula_and_missing_margin_resume(self):
        for activation, expected_head in (('sigmoid', nn.Sigmoid), ('linear', nn.Identity)):
            with self.subTest(activation=activation):
                legacy = self.model(0., activation, legacy=True)
                explicit = self.model(0., activation)
                explicit.load_state_dict(legacy.state_dict(), strict=True)
                self.assertIsInstance(legacy.fcn[1], expected_head)
                x = torch.rand(2, 10, 2, dtype=torch.float64)
                with torch.inference_mode():
                    left, right = legacy(x), explicit(x)
                    logits = explicit.fcn[0](torch.rand(2, 1, 4, dtype=torch.float64))
                    torch.testing.assert_close(explicit.fcn[1](logits), expected_head()(logits), rtol=0, atol=0)
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                path, checkpoint = self.save_snapshot(legacy, legacy=True)
                self.assertNotIn('output_margin', checkpoint['model_cfg'])
                destination = self.cpu_snapshot_engine(explicit)
                with patch('src.trainer._assert_model_device'), patch('torch.cuda.set_rng_state_all'):
                    destination._restore(path)
                self.assertEqual(destination.model.config['output_margin'], 0.)

    def test_invalid_margin_and_ambiguous_legacy_values_are_rejected(self):
        for margin in (-.1, float('nan'), float('inf'), float('-inf'), True, None, 'bad', 1e308):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                canonical_model_config({'output_activation': 'scaled_sigmoid', 'output_margin': margin})
        for activation in ('sigmoid', 'linear'):
            with self.subTest(activation=activation), self.assertRaisesRegex(ValueError, 'must be 0'):
                self.model(.1, activation)

    def test_fixed_candidate_configs_use_declared_keys_and_scratch_ten_epochs(self):
        from .pipeline import merge_dict
        directory = Path(__file__).parent/'experiment_configs'
        paths = sorted(directory.glob('scaled_sigmoid_m*.json'))
        self.assertEqual(len(paths), 2)
        margins = []
        for path in paths:
            config = merge_dict(get_config(), json.loads(path.read_text(encoding='utf-8')))
            model = build_model(2, config['model'])
            margins.append(model.config['output_margin'])
            self.assertEqual(model.config['output_activation'], 'scaled_sigmoid')
            self.assertEqual(config['train']['epochs'], 10)
            self.assertEqual(config['train']['max_epochs'], 30)
            self.assertIsNone(config['train']['resume_from'])
            self.assertTrue(config['train']['early_stopping'])
            self.assertEqual(config['deadline_utc'], '2026-09-21T22:00:00+00:00')
        self.assertEqual(margins, [.1, .25])


if __name__ == '__main__':
    if os.environ.get('EWS_AUDIT_NONCE'):
        print('EWS_AUDIT_NONCE='+os.environ['EWS_AUDIT_NONCE'], flush=True)
    unittest.main()
