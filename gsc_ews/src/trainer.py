"""공통 DataLoader와 모델 어댑터를 연결하는 Lightning 학습·추론 설정."""

import re
from pathlib import Path
from statistics import median
from time import perf_counter

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .dataloader import evaluation_loader
from .inference import predict_batches
from .model import build_model
from .udf.quality import compute_reconstruction_quality


class EpochTimer(Callback):
    """Epoch별 학습시간을 노트북과 터미널에 즉시 출력한다."""

    def __init__(self, fold_name):
        """출력에 사용할 Fold 이름을 저장한다."""
        self.fold_name = fold_name
        self.started_at = None

    def on_train_epoch_start(self, trainer, _pl_module):
        """Epoch 학습 시작 시각을 기록한다."""
        self.started_at = perf_counter()

    def on_train_epoch_end(self, trainer, _pl_module):
        """메인 프로세스에서 Epoch 시간·손실·학습률을 출력한다."""
        if trainer.is_global_zero and self.started_at is not None:
            elapsed = perf_counter() - self.started_at
            metrics = trainer.callback_metrics

            def display(name):
                """기록되지 않았거나 유한하지 않은 지표는 N/A로 표시한다."""
                value = metrics.get(name)
                if value is None or not np.isfinite(float(value)):
                    return "N/A"
                return f"{float(value):.8g}"

            print(  # noqa: T201
                f"[{self.fold_name}] Epoch {trainer.current_epoch + 1} 실행시간: {elapsed:.2f}초 "
                f"(train_loss: {display('train_loss')}, val_loss: {display('val_loss')}, "
                f"LR: {display('learning_rate')})",
                flush=True,
            )


def build_trainer(config, *, for_prediction=False, fold_index=None, **overrides):
    """검증된 설정으로 장치·정밀도를 구성하고 학습에만 로거·체크포인트를 붙인다."""
    model_name = config["active_model"]
    model_config = config["models"][model_name]
    training = config["training"]
    run = config["run"]
    accelerator = "gpu" if run["device"] == "cuda" else run["device"]
    options = {
        "accelerator": accelerator,
        "devices": 1,
        "precision": "64-true" if model_config["dtype"] == "float64" else "32-true",
        "deterministic": bool(run["deterministic"]),
        "enable_progress_bar": bool(training["enable_progress_bar"]),
    }
    if for_prediction:
        options.update(logger=False, enable_checkpointing=False)
    else:
        checkpoint = training["checkpoint"]
        version = config.get("model_version", "v0_0_0")
        run_id = run.get("run_id") or "manual"
        artifact_path = Path(config["equipment_id"]) / model_name / version / run_id
        if not config["dataloader"]["time_series_split_enabled"]:
            fold_name = "holdout"
        elif fold_index == 0:
            fold_name = "final"
        else:
            fold_name = f"fold_{fold_index:02d}" if fold_index else "single"
        callbacks = [
            ModelCheckpoint(
                dirpath=str(Path(checkpoint["directory"]) / artifact_path / fold_name),
                filename="{epoch:03d}-{val_loss:.6f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1 if checkpoint["save_best_only"] else -1,
                save_last=True,
            )
        ]
        stopping = training["early_stopping"]
        if stopping["enabled"]:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=int(stopping["patience"]),
                    min_delta=float(stopping["min_delta"]),
                    check_finite=True,
                )
            )
        options.update(
            max_epochs=int(model_config["epochs"]),
            callbacks=callbacks,
            logger=CSVLogger(
                training["log_directory"],
                name=f"{config['equipment_id']}/{model_name}/{version}",
                version=f"{run_id}/{fold_name}",
            ),
            gradient_clip_val=float(model_config["gradient_clip_val"]),
            log_every_n_steps=int(training["log_every_n_steps"]),
        )
    options.update(overrides)
    if not for_prediction:
        options["callbacks"] = [*options.get("callbacks", []), EpochTimer(fold_name)]
    return pl.Trainer(**options)


def fit_model(bundle, config, *, ckpt_path=None, **trainer_overrides):
    """시간순 검증 점수를 수집하고 설정에 따라 전체 정상 데이터 재학습 여부를 결정한다."""
    folds = bundle.folds or [bundle]
    split_enabled = config["dataloader"]["time_series_split_enabled"]
    if ckpt_path:
        raise ValueError("TimeSeriesSplit 학습은 단일 체크포인트에서 재개할 수 없습니다")
    oof_scores, oof_segments, results, best_epochs = [], [], [], []
    segment_offset = 0
    for fold in folds:
        pl.seed_everything(int(config["run"]["seed"]), workers=True)
        model = build_model(config, fold.n_features)
        model.set_data_state(fold)
        trainer = build_trainer(config, fold_index=fold.fold_index, **trainer_overrides)
        trainer.fit(
            model,
            train_dataloaders=fold.train_loader,
            val_dataloaders=fold.val_loader,
            ckpt_path=ckpt_path,
        )
        checkpoint = trainer.checkpoint_callback
        if checkpoint and checkpoint.best_model_path:
            model = type(model).load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
        loader = evaluation_loader(fold.val_loader.dataset, config["dataloader"])
        prediction = predict_batches(model.eval(), loader, trainer=trainer)
        validation_quality = compute_reconstruction_quality(
            fold.scaler.inverse_transform(prediction["prediction"]),
            fold.scaler.inverse_transform(prediction["target"]),
            fold.feature_names,
        )["overall"]
        segments = loader.dataset.segment_ids[prediction["index"]]
        segments = segments - segments.min() + segment_offset
        segment_offset = int(segments.max()) + 1
        oof_scores.append(prediction["anomaly_score"])
        oof_segments.append(segments)
        best_score = None if checkpoint is None else checkpoint.best_model_score
        validation_loss = float(
            best_score if best_score is not None else trainer.callback_metrics.get("val_loss", float("inf"))
        )
        if not np.isfinite(validation_loss):
            validation_loss = float(prediction["anomaly_score"].mean())
        match = re.search(r"epoch=(\d+)", Path(checkpoint.best_model_path).name) if checkpoint else None
        best_epoch = int(match.group(1)) + 1 if match else max(1, int(trainer.current_epoch))
        best_epochs.append(best_epoch)
        results.append(
            {
                "strategy": "time_series_split" if split_enabled else "time_holdout",
                "fold": fold.fold_index if split_enabled else None,
                "train_rows": len(fold.train_loader.dataset),
                "validation_rows": len(fold.val_loader.dataset),
                "best_epoch": best_epoch,
                "validation_loss": validation_loss,
                "validation_reconstruction": {
                    name: float(validation_quality[name]) for name in ("MAE", "RMSE", "R2", "MAPE")
                },
            }
        )

    final_epochs = max(1, int(round(median(best_epochs))))
    losses = [result["validation_loss"] for result in results]
    trainer.oof_raw_score = np.concatenate(oof_scores)
    trainer.oof_segments = np.concatenate(oof_segments)
    trainer.cross_validation = {
        "strategy": "time_series_split" if split_enabled else "time_holdout",
        "folds": results,
        "validation_loss_mean": float(np.mean(losses)),
        "validation_loss_std": float(np.std(losses)),
        "validation_reconstruction_mean": {
            name: float(np.mean([result["validation_reconstruction"][name] for result in results]))
            for name in ("MAE", "RMSE", "R2", "MAPE")
        },
        "final_epochs": final_epochs,
    }
    if not split_enabled:
        return model.eval(), trainer

    # 여러 Fold에서 정한 Epoch 수로만 전체 정상 데이터 운영 모델을 재학습한다.
    checkpoint_cfg = config["training"]["checkpoint"]
    final_callback = ModelCheckpoint(
        dirpath=str(
            Path(checkpoint_cfg["directory"])
            / config["equipment_id"]
            / config["active_model"]
            / config["model_version"]
            / (config["run"].get("run_id") or "manual")
            / "final"
        ),
        filename="final-{epoch:03d}",
        save_top_k=1,
        save_last=True,
    )
    pl.seed_everything(int(config["run"]["seed"]), workers=True)
    model = build_model(config, bundle.n_features)
    model.set_data_state(bundle)
    trainer = build_trainer(
        config,
        fold_index=0,
        max_epochs=final_epochs,
        callbacks=[final_callback],
        **trainer_overrides,
    )
    trainer.fit(model, train_dataloaders=bundle.train_loader)
    checkpoint = trainer.checkpoint_callback
    if checkpoint and checkpoint.best_model_path:
        model = type(model).load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
    trainer.oof_raw_score = np.concatenate(oof_scores)
    trainer.oof_segments = np.concatenate(oof_segments)
    trainer.cross_validation = {
        "strategy": "time_series_split",
        "folds": results,
        "validation_loss_mean": float(np.mean(losses)),
        "validation_loss_std": float(np.std(losses)),
        "validation_reconstruction_mean": {
            name: float(np.mean([result["validation_reconstruction"][name] for result in results]))
            for name in ("MAE", "RMSE", "R2", "MAPE")
        },
        "final_epochs": final_epochs,
    }
    return model.eval(), trainer


def training_history(trainer):
    """Lightning CSVLogger가 기록한 epoch별 loss와 실제 학습률을 읽는다."""
    import pandas as pd

    if not trainer.logger:
        return None
    trainer.logger.save()
    path = Path(trainer.logger.log_dir) / "metrics.csv"
    if not path.exists():
        return None
    history = pd.read_csv(path).dropna(subset=["epoch"]).groupby("epoch").last()
    history = history.dropna(subset=["train_loss"])
    if history.empty:
        return None
    return {
        "epoch": (history.index.to_numpy(dtype=int) + 1).tolist(),
        "train_loss": history["train_loss"].tolist(),
        "val_loss": history["val_loss"].tolist() if "val_loss" in history else [None] * len(history),
        "lr": history["learning_rate"].tolist(),
    }
