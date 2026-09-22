"""GPU-only TranAD training and bounded minibatch reconstruction.

Training follows the public TranAD main.py at commit
7ffb98d0c18189cc3d9ab732b4cb0278200a0af0:
  loss = MSE(recon1,target)/epoch + (1-1/epoch)*MSE(recon2,target).
https://github.com/imperial-qore/TranAD/blob/7ffb98d0c18189cc3d9ab732b4cb0278200a0af0/main.py
This is the public implementation's evolving objective, not a claim to implement
the paper's additional MAML step or separate adversarial maximization objective.
Anomaly scoring is handled separately using the user's required MAE.

train_cfg['epochs'] is the desired cumulative epoch count, NOT extra epochs.
Resume only this project's trusted checkpoint via train_cfg['resume_from']; never
load a checkpoint obtained from an untrusted source. The user cap of 30 applies to
all epochs started by the model, including partial epochs and resumed training.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import math
import os
import random
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import RandomSampler
from torch.nn import functional as F
from .model import canonical_model_config


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """센서 복원 손실만 계산한다. Label/Inhibit를 받지 않는 순수 함수다.

    기본 MSE는 공개 TranAD 구현과 같고, 나머지는 명시적으로 기록하는
    프로젝트 실험이다. 모든 목적함수에 동일한 1/epoch, 1-1/epoch 가중 적용.
    """
    objective = str(cfg.get("objective", "mse")).lower()
    if objective == "mse":
        return F.mse_loss(prediction, target)
    if objective == "l1":
        return F.l1_loss(prediction, target)
    if objective == "huber":
        return F.huber_loss(prediction, target, delta=float(cfg.get("huber_delta", .1)))
    if objective == "mse_l1":
        weight = float(cfg.get("l1_weight", .5))
        return (1-weight)*F.mse_loss(prediction, target) + weight*F.l1_loss(prediction, target)
    raise ValueError(f"Unsupported reconstruction objective: {objective}")


class DeadlineReached(RuntimeError):
    """The externally configured hard deadline was reached between minibatches."""


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_deadline(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    deadline = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline_utc must include a UTC offset")
    return deadline.astimezone(dt.timezone.utc)


def _check_deadline(deadline: dt.datetime | None) -> None:
    if deadline is not None and _utc_now() >= deadline:
        raise DeadlineReached(f"Hard deadline reached: {deadline.isoformat()}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_scaler_state(payload: dict) -> dict:
    """Normalize old sklearn-containing and new portable checkpoint signatures.

    Early runs saved {'scaler': MinMaxScaler, 'features', 'training_rows', ...}.
    New callers supply only the fitted numeric state. Compare the same complete
    transform parameters and feature order in both forms, without re-fitting or
    modifying the scaler. This project's transform is un-clipped MinMax [0,1].
    """
    if not isinstance(payload, dict):
        raise TypeError("scaler_state must be a dictionary")
    features = list(payload["features"])
    if not features or len(set(features)) != len(features):
        raise ValueError("Scaler feature names must be nonempty and unique")
    state = {"features": features, "training_rows": int(payload["training_rows"])}
    if state["training_rows"] < 1:
        raise ValueError("Scaler training row count must be positive")
    scaler = payload.get("scaler")
    if scaler is not None:
        if tuple(scaler.feature_range) != (0, 1) or bool(scaler.clip):
            raise ValueError("Resume scaler must use un-clipped MinMax feature_range=(0,1)")
        if int(scaler.n_features_in_) != len(features):
            raise ValueError("Scaler fitted feature count does not match feature names")
    for key in ("data_min_", "data_max_", "scale_", "min_"):
        value = getattr(scaler, key) if scaler is not None else payload[key]
        array = np.asarray(value)
        if array.shape != (len(features),) or not np.isfinite(array).all():
            raise ValueError(f"Invalid fitted scaler parameter: {key}")
        state[key] = array.tolist()
    seen = getattr(scaler, "n_samples_seen_") if scaler is not None else payload["n_samples_seen_"]
    state["n_samples_seen_"] = int(seen)
    if state["n_samples_seen_"] != state["training_rows"]:
        raise ValueError("Scaler sample count must match the fitted chronological training partition")
    return state


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
    os.replace(temporary, path)


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_trusted_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load only our own checkpoint; scaler state can contain numpy objects.

    Explicit weights_only=False is required by recent PyTorch versions. This is
    intentionally not a general-purpose loader for downloaded checkpoint files.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # compatibility with older SageMaker PyTorch environments
        return torch.load(path, map_location=map_location)


def _cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU execution is required. CUDA is unavailable; CPU fallback is forbidden.")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"Unavailable CUDA device index: {index}")
    return torch.device("cuda", index)


def _assert_model_device(model: nn.Module, device: torch.device) -> None:
    if any(value.device != device for value in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError(f"Every model parameter and buffer must reside on {device}")


def _batch_to_device(batch: Any, model: nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("DataLoader must yield (window[B,L,F], target[B,F])")
    dtype = next(model.parameters()).dtype
    window, target = (item.to(device=device, dtype=dtype, non_blocking=True) for item in batch)
    if window.device != device or target.device != device:
        raise RuntimeError("CUDA batch transfer failed")
    if window.ndim != 3 or target.shape != window[:, -1, :].shape:
        raise ValueError("Bad window/target shapes")
    if not torch.isfinite(window).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite values reached the model; check preprocessing")
    # The task is current-time reconstruction, not next-step forecasting. This
    # catches a silent one-minute target shift before it contaminates evaluation.
    if not torch.equal(target, window[:, -1, :]):
        raise ValueError("Target must equal the final row of its input window")
    return window, target


class Trainer:
    """Train on complete supplied loaders, preserving best and latest states.

    All experiment knobs are read from train_cfg (normally Config.py). Required:
    epochs, max_epochs, lr, weight_decay, scheduler_step, scheduler_gamma,
    patience, min_delta, batch_size, seed, grad_clip. Optional: resume_from,
    early_stopping, progress_every_batches, progress_every_seconds. Best selection
    uses pure validation MAE; Test labels/data are never read by this class.
    """

    def __init__(self, model: nn.Module, train_cfg: dict, device: torch.device,
                 run_dir: Path, deadline_utc: str | None):
        self.cfg = copy.deepcopy(train_cfg)
        self.device = _cuda_device(device)
        self.model = model.to(self.device)
        _assert_model_device(self.model, self.device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.deadline = _parse_deadline(deadline_utc)
        self.model.deadline_utc = deadline_utc
        self.target_epochs = int(self.cfg["epochs"])
        self.max_epochs = int(self.cfg["max_epochs"])
        self.min_epochs = int(self.cfg.get("min_epochs", 10))
        if not 10 <= self.min_epochs <= self.target_epochs <= self.max_epochs <= 30:
            raise ValueError("Require 10 <= min_epochs <= epochs <= max_epochs <= user hard cap 30")
        self.cfg.setdefault("objective", "mse")
        self.cfg.setdefault("huber_delta", .1)
        self.cfg.setdefault("l1_weight", .5)
        self.cfg.setdefault("save_epoch_checkpoints", True)
        if str(self.cfg["objective"]).lower() not in {"mse", "l1", "huber", "mse_l1"}:
            raise ValueError("objective must be mse, l1, huber or mse_l1")
        if float(self.cfg["huber_delta"]) <= 0 or not 0 <= float(self.cfg["l1_weight"]) <= 1:
            raise ValueError("huber_delta must be positive and l1_weight in [0,1]")
        self.model.source_info["selected_training_objective"] = self.cfg["objective"]
        if float(self.cfg["lr"]) <= 0 or float(self.cfg["weight_decay"]) < 0:
            raise ValueError("lr must be positive and weight_decay nonnegative")
        if int(self.cfg["scheduler_step"]) < 1 or float(self.cfg["scheduler_gamma"]) <= 0:
            raise ValueError("Invalid scheduler settings")
        if int(self.cfg["patience"]) < 1 or float(self.cfg["min_delta"]) < 0:
            raise ValueError("Invalid early stopping settings")
        if float(self.cfg["grad_clip"]) < 0:
            raise ValueError("grad_clip must be nonnegative; zero disables clipping")
        seed = int(self.cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(self.cfg["lr"]),
            weight_decay=float(self.cfg["weight_decay"])
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(self.cfg["scheduler_step"]),
            gamma=float(self.cfg["scheduler_gamma"])
        )
        self.completed_epochs = 0
        self.epochs_consumed = 0
        self.active_epoch = 0
        self.batches_completed_in_active_epoch = 0
        self.best_val_mae = math.inf
        self.best_epoch = 0
        self.bad_epochs = 0
        self.history: list[dict] = []
        self.scaler_state = None
        self.feature_names = None
        self.best_model_state = None
        self.loader_generator_states = {}
        self.model_id = str(uuid.uuid4())
        self.started_utc = _utc_now().isoformat()
        props = torch.cuda.get_device_properties(self.device)
        self.gpu_evidence = {
            "device": str(self.device), "name": props.name,
            "total_memory_bytes": props.total_memory,
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "model_dtype": str(next(self.model.parameters()).dtype),
            "model_device_verified": True,
        }
        torch.cuda.reset_peak_memory_stats(self.device)
        resume = self.cfg.get("resume_from")
        if resume:
            self._restore(Path(resume))

    def _restore(self, path: Path) -> None:
        if path.name == "best.pt":
            raise ValueError("best.pt is an evaluation snapshot; resume training from last.pt or interrupted.pt")
        state = load_trusted_checkpoint(path, self.device)
        if state.get("evaluation_snapshot", False):
            raise ValueError("Cannot resume training from evaluation-only model weights")
        if (state["n_features"] != self.model.n_features or
                canonical_model_config(state["model_cfg"]) != self.model.config):
            raise ValueError("Resume model architecture differs from checkpoint")
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        # Restore optimizer moments first. Explicit Config changes then override
        # only the requested knobs, so a resumed run never silently ignores them.
        previous_cfg = state["train_cfg"]
        if float(self.cfg["lr"]) != float(previous_cfg["lr"]):
            for group in self.optimizer.param_groups:
                group["lr"] = float(self.cfg["lr"])
                group["initial_lr"] = float(self.cfg["lr"])
            self.scheduler.base_lrs = [float(self.cfg["lr"])] * len(self.optimizer.param_groups)
            self.scheduler._last_lr = [float(self.cfg["lr"])] * len(self.optimizer.param_groups)
        for group in self.optimizer.param_groups:
            group["weight_decay"] = float(self.cfg["weight_decay"])
        self.scheduler.step_size = int(self.cfg["scheduler_step"])
        self.scheduler.gamma = float(self.cfg["scheduler_gamma"])
        self.completed_epochs = int(state["completed_epochs"])
        self.epochs_consumed = int(state["epochs_consumed"])
        self.model_id = state.get("model_id", self.model_id)
        budget_path = path.parent / "epoch_budget.json"
        if budget_path.exists():
            with budget_path.open(encoding="utf-8") as stream:
                budget = json.load(stream)
            if budget["model_id"] != self.model_id:
                raise ValueError("Resume epoch budget belongs to a different model")
            # A crash/timeout can happen after an epoch started but before last.pt
            # was updated. Its consumed budget cannot be reset by loading last.pt.
            self.epochs_consumed = max(self.epochs_consumed, int(budget["epochs_consumed"]))
        self.best_val_mae = float(state["best_val_mae"])
        self.best_epoch = int(state["best_epoch"])
        self.bad_epochs = int(state["bad_epochs"])
        self.history = state["history"]
        self.scaler_state = state.get("scaler_state")
        self.feature_names = state.get("feature_names")
        self.best_model_state = state.get("best_model_state_dict")
        self.loader_generator_states = state.get("loader_generator_states", {})
        if self.epochs_consumed > self.max_epochs:
            raise ValueError("Checkpoint already exceeds the allowed cumulative epochs")
        if self.target_epochs < self.epochs_consumed:
            raise ValueError("Requested cumulative epochs is below the already consumed epoch budget")
        # Restoring RNG keeps continuation honest; the DataLoader's own generator
        # should also be seeded by the integration layer if shuffled training is used.
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
            np.random.set_state(state["numpy_rng_state"])
            random.setstate(state["python_rng_state"])
        _assert_model_device(self.model, self.device)
        print(f"RESUME {path} completed={self.completed_epochs} consumed={self.epochs_consumed}", flush=True)

    def _checkpoint(self, status: str, error: str | None = None) -> dict:
        return {
            "format_version": 2, "saved_utc": _utc_now().isoformat(), "status": status,
            "model_weights_epoch": self.active_epoch or self.completed_epochs,
            "evaluation_snapshot": False,
            "model_id": self.model_id,
            "error": error, "started_utc": self.started_utc,
            "model_state_dict": self.model.state_dict(),
            "best_model_state_dict": self.best_model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_cfg": self.model.config, "train_cfg": self.cfg,
            "n_features": self.model.n_features, "feature_names": self.feature_names,
            "scaler_state": self.scaler_state, "completed_epochs": self.completed_epochs,
            "completed_epoch_count": len(self.history),
            "epochs_consumed": self.epochs_consumed, "active_epoch": self.active_epoch,
            "batches_completed_in_active_epoch": self.batches_completed_in_active_epoch,
            "best_val_mae": self.best_val_mae, "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs, "history": self.history,
            "source_info": self.model.source_info, "gpu_evidence": self.gpu_evidence,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            "loader_generator_states": self.loader_generator_states,
        }

    def _save_history(self) -> None:
        if not self.history:
            return
        path = self.run_dir / "history.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        os.replace(temporary, path)

    def _validation(self, loader: Any) -> tuple[float, int]:
        self.model.eval()
        error_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        value_count = 0
        with torch.inference_mode():
            for batch in loader:
                _check_deadline(self.deadline)
                window, target = _batch_to_device(batch, self.model, self.device)
                _, reconstructed = self.model(window)
                if not torch.isfinite(reconstructed).all():
                    raise FloatingPointError("Validation reconstruction contains nonfinite values")
                error_sum += (reconstructed - target).abs().sum(dtype=torch.float64)
                value_count += target.numel()
        if value_count == 0:
            raise ValueError("Validation loader is empty")
        return float(error_sum.item() / value_count), value_count

    def fit(self, train_loader: Any, val_loader: Any, scaler_state=None, feature_names=None) -> dict:
        if getattr(train_loader, "drop_last", False) or getattr(val_loader, "drop_last", False):
            raise ValueError("drop_last must be False: every supplied window must be processed")
        if isinstance(getattr(val_loader, "sampler", None), RandomSampler):
            raise ValueError("Validation loader must preserve chronological order")
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Train and validation loaders must be nonempty")
        for name, loader in (("train", train_loader), ("validation", val_loader)):
            generator = getattr(loader, "generator", None)
            if generator is not None and name in self.loader_generator_states:
                generator.set_state(self.loader_generator_states[name].cpu())
            sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
            if sampler_generator is not None and name + "_sampler" in self.loader_generator_states:
                sampler_generator.set_state(self.loader_generator_states[name + "_sampler"].cpu())
        if self.feature_names is not None and feature_names is not None and list(feature_names) != list(self.feature_names):
            raise ValueError("Resume feature names/order mismatch")
        if feature_names is not None:
            self.feature_names = list(feature_names)
        if self.feature_names is not None and len(self.feature_names) != self.model.n_features:
            raise ValueError("Feature count does not match model")
        if scaler_state is not None:
            # The integration layer must reuse the exact fitted scaler on resume.
            # Compare its portable state to prevent an unnoticed new normalization.
            current_scaler_state = canonical_scaler_state(scaler_state)
            if self.scaler_state is not None:
                old = json.dumps(canonical_scaler_state(self.scaler_state), sort_keys=True)
                new = json.dumps(current_scaler_state, sort_keys=True)
                if old != new:
                    raise ValueError("Resume scaler differs from the checkpoint scaler")
            self.scaler_state = current_scaler_state
        _write_json(self.run_dir / "training_config.json", {
            "model": self.model.config, "train": self.cfg, "deadline_utc": self.model.deadline_utc,
            "features": self.feature_names, "source_info": self.model.source_info,
            "gpu_evidence": self.gpu_evidence,
        })
        # If resuming into another directory, preserve an existing validated best
        # snapshot before trying additional epochs. It remains an evaluation model.
        if self.best_model_state is not None and not (self.run_dir / "best.pt").exists():
            restored_best = self._checkpoint("restored_best_evaluation_snapshot")
            restored_best["model_state_dict"] = self.best_model_state
            restored_best["model_weights_epoch"] = self.best_epoch
            restored_best["evaluation_snapshot"] = True
            _atomic_save(restored_best, self.run_dir / "best.pt")
        start = time.monotonic()
        reason = "target_epochs_reached"
        try:
            for epoch in range(self.epochs_consumed + 1, self.target_epochs + 1):
                _check_deadline(self.deadline)
                self.active_epoch = epoch
                self.epochs_consumed = epoch
                _write_json(self.run_dir / "epoch_budget.json", {
                    "model_id": self.model_id, "epochs_consumed": self.epochs_consumed,
                    "max_epochs": self.max_epochs, "updated_utc": _utc_now().isoformat(),
                })
                self.batches_completed_in_active_epoch = 0
                self.model.train()
                epoch_start = time.monotonic()
                next_progress = epoch_start
                loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
                mae_sum = torch.zeros((), device=self.device, dtype=torch.float64)
                value_count = 0
                learning_rate = self.optimizer.param_groups[0]["lr"]
                for batch_index, batch in enumerate(train_loader, start=1):
                    _check_deadline(self.deadline)
                    window, target = _batch_to_device(batch, self.model, self.device)
                    first, second = self.model(window)
                    if "first_batch_tensors" not in self.gpu_evidence:
                        self.gpu_evidence["first_batch_tensors"] = {
                            "window_device": str(window.device), "window_dtype": str(window.dtype),
                            "window_shape": list(window.shape), "target_device": str(target.device),
                            "recon1_device": str(first.device), "recon2_device": str(second.device),
                            "reconstruction_shape": list(second.shape),
                        }
                    loss = reconstruction_loss(first, target, self.cfg) / epoch + \
                           (1 - 1 / epoch) * reconstruction_loss(second, target, self.cfg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batch_index}")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = float(self.cfg["grad_clip"])
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip, error_if_nonfinite=True)
                    self.optimizer.step()
                    loss_sum += loss.detach().to(torch.float64) * target.numel()
                    mae_sum += (second.detach() - target).abs().sum(dtype=torch.float64)
                    value_count += target.numel()
                    self.batches_completed_in_active_epoch = batch_index
                    now = time.monotonic()
                    progress_batches = int(self.cfg.get("progress_every_batches", 100))
                    progress_seconds = float(self.cfg.get("progress_every_seconds", 30))
                    if now >= next_progress or batch_index == len(train_loader) or (progress_batches > 0 and batch_index % progress_batches == 0):
                        print(f"TRAIN epoch={epoch}/{self.target_epochs} batch={batch_index}/{len(train_loader)} "
                              f"loss={loss_sum.item()/value_count:.7f} elapsed_s={now-epoch_start:.1f} "
                              f"device={self.device}", flush=True)
                        next_progress = now + progress_seconds
                val_mae, val_values = self._validation(val_loader)
                self.completed_epochs = epoch
                # Best checkpoint always means the numerically lowest validation
                # MAE. min_delta only controls patience, not which model is best.
                significant = val_mae < self.best_val_mae - float(self.cfg["min_delta"])
                is_best = val_mae < self.best_val_mae
                self.bad_epochs = 0 if significant else self.bad_epochs + 1
                if is_best:
                    self.best_val_mae = val_mae
                    self.best_epoch = epoch
                    self.best_model_state = {key: value.detach().cpu().clone()
                                             for key, value in self.model.state_dict().items()}
                self.scheduler.step()
                torch.cuda.synchronize(self.device)
                self.gpu_evidence["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                self.gpu_evidence["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
                row = {
                    "epoch": epoch, "train_loss": float(loss_sum.item()/value_count),
                    "train_mae": float(mae_sum.item()/value_count), "val_mae": val_mae,
                    "lr": learning_rate, "seconds": time.monotonic()-epoch_start,
                    "train_windows": value_count//self.model.n_features,
                    "val_windows": val_values//self.model.n_features,
                    "device": str(self.device), "gpu_name": self.gpu_evidence["name"],
                    "peak_gpu_memory_bytes": self.gpu_evidence["peak_allocated_bytes"],
                    "is_best": is_best, "utc": _utc_now().isoformat(),
                }
                self.history.append(row)
                self._save_history()
                self.loader_generator_states = {
                    name: loader.generator.get_state()
                    for name, loader in (("train", train_loader), ("validation", val_loader))
                    if getattr(loader, "generator", None) is not None
                }
                self.loader_generator_states.update({
                    name + "_sampler": loader.sampler.generator.get_state()
                    for name, loader in (("train", train_loader), ("validation", val_loader))
                    if getattr(getattr(loader, "sampler", None), "generator", None) is not None
                })
                snapshot = self._checkpoint("epoch_complete")
                _atomic_save(snapshot, self.run_dir / "last.pt")
                if self.cfg["save_epoch_checkpoints"]:
                    _atomic_save(snapshot, self.run_dir / f"epoch_{epoch:02d}.pt")
                if is_best:
                    best_snapshot = dict(snapshot, evaluation_snapshot=True, model_weights_epoch=epoch)
                    _atomic_save(best_snapshot, self.run_dir / "best.pt")
                print(f"EPOCH {epoch} train_loss={row['train_loss']:.7f} val_MAE={val_mae:.7f} "
                      f"best_epoch={self.best_epoch} seconds={row['seconds']:.1f}", flush=True)
                if (epoch >= self.min_epochs and bool(self.cfg.get("early_stopping", True))
                        and self.bad_epochs >= int(self.cfg["patience"])):
                    reason = "early_stopping"
                    break
        except DeadlineReached as exc:
            reason = "deadline_reached"
            _atomic_save(self._checkpoint(reason, str(exc)), self.run_dir / "interrupted.pt")
            print(str(exc), flush=True)
        except BaseException as exc:
            reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            try:
                _atomic_save(self._checkpoint(reason, traceback.format_exc()), self.run_dir / "error.pt")
            except BaseException as save_error:
                print(f"Emergency checkpoint failed: {save_error}", flush=True)
            self._save_history()
            raise
        self._save_history()
        result = {
            "status": reason, "completed_epochs": self.completed_epochs,
            "completed_epoch_count": len(self.history),
            "model_id": self.model_id,
            "epochs_consumed": self.epochs_consumed, "best_epoch": self.best_epoch,
            "minimum_epochs_completed": len(self.history) >= self.min_epochs,
            "best_val_mae": self.best_val_mae if math.isfinite(self.best_val_mae) else None,
            "best_checkpoint": str(self.run_dir / "best.pt") if self.best_model_state is not None else None,
            "last_checkpoint": str(self.run_dir / "last.pt") if (self.run_dir / "last.pt").exists() else None,
            "history_file": str(self.run_dir / "history.csv"), "elapsed_seconds": time.monotonic()-start,
            "gpu_evidence": self.gpu_evidence, "feature_names": self.feature_names,
            "model_cfg": self.model.config, "train_cfg": self.cfg,
            "deadline_utc": self.model.deadline_utc,
        }
        _write_json(self.run_dir / "training_summary.json", result)
        if self.best_model_state is None:
            raise RuntimeError(f"Training ended ({reason}) before one complete validation; no usable best model")
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        _assert_model_device(self.model, self.device)
        return result


def reconstruct(model: nn.Module, loader: Any, device: torch.device) -> np.ndarray:
    """Return decoder-2 predictions [N,F] in loader order using only CUDA.

    A sequential loader and drop_last=False are required for timestamp alignment.
    No full-dataset GPU tensor is created. Empty loaders return shape [0,F].
    The optimization deadline never applies here: final evaluation and report
    generation must remain available after the training search stops.
    """
    device = _cuda_device(device)
    _assert_model_device(model, device)
    if getattr(loader, "drop_last", False):
        raise ValueError("Inference drop_last must be False")
    if isinstance(getattr(loader, "sampler", None), RandomSampler):
        raise ValueError("Inference loader must be chronological, without shuffling")
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            window, _ = _batch_to_device(batch, model, device)
            _, reconstructed = model(window)
            if not torch.isfinite(reconstructed).all():
                raise FloatingPointError("Inference generated a nonfinite reconstruction")
            chunks.append(reconstructed.cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.n_features), dtype=np.float32)
