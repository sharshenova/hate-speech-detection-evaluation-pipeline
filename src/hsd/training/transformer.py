"""Fine-tune a transformer sequence classifier from a YAML experiment config.

Designed to run on Apple Silicon (MPS), CUDA, or CPU. Writes the same artifact
layout as the TF-IDF baseline (metrics.json, config.resolved.json) plus
training history, curve plots, and a test confusion matrix.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsd.data.canonical import load_canonical_dataset, xy_from_frame
from hsd.evaluation.metrics import (
    apply_threshold,
    classification_metrics,
    tune_binary_threshold,
)
from hsd.utils.config import resolve_results_dir
from hsd.utils.seed import set_global_seed


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        msg = "Transformer training requires the ml extras: pip install -e '.[dev,ml]'"
        raise RuntimeError(msg) from exc


def select_device(preference: str = "auto") -> str:
    """Pick the training device.

    With ``preference="auto"`` (default): CUDA if available, then Apple MPS,
    then CPU. An explicit preference ("cuda", "mps", "cpu") is validated
    against what the machine actually supports.
    """
    import torch

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()

    if preference == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if preference == "cuda" and not cuda_ok:
        msg = "Config requests device 'cuda' but no CUDA GPU is available"
        raise RuntimeError(msg)
    if preference == "mps" and not mps_ok:
        msg = "Config requests device 'mps' but MPS is not available"
        raise RuntimeError(msg)
    if preference not in {"cuda", "mps", "cpu"}:
        msg = f"Unknown device preference: {preference!r} (use auto, cuda, mps, or cpu)"
        raise ValueError(msg)
    return preference


def _class_weights(labels: pd.Series) -> list[float]:
    """sklearn-style 'balanced' weights: n_samples / (n_classes * bincount)."""
    counts = labels.value_counts().sort_index()
    n_samples = int(counts.sum())
    n_classes = len(counts)
    return [n_samples / (n_classes * int(counts[label])) for label in sorted(counts.index)]


class _TextDataset:
    def __init__(self, encodings: dict[str, list[list[int]]], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def _resolve_model_settings(model_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Merge optional ``model.cuda`` overrides when training on CUDA."""
    resolved = dict(model_cfg)
    if device == "cuda" and isinstance(model_cfg.get("cuda"), dict):
        resolved.update(model_cfg["cuda"])
    return resolved


def _param_applies_weight_decay(param_name: str) -> bool:
    """BERT-style: no weight decay on bias and LayerNorm scale."""
    return not any(part in param_name for part in ("bias", "LayerNorm.weight"))


def _deberta_layer_id(param_name: str, num_hidden_layers: int) -> int:
    """Map a DeBERTa parameter to a depth index for LLRD (0 = embeddings, top = head)."""
    if "embeddings" in param_name:
        return 0
    match = re.search(r"\.layer\.(\d+)\.", param_name)
    if match:
        return int(match.group(1)) + 1
    # Classifier / pooler: highest LR (same as top encoder layer + 1).
    return num_hidden_layers + 1


def _build_layerwise_adamw_groups(
    model,
    *,
    learning_rate: float,
    weight_decay: float,
    decay: float,
) -> list[dict[str, Any]]:
    """Discriminative fine-tuning groups: lr = base_lr * decay**(depth_offset)."""
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 12))
    max_depth = num_hidden_layers + 1
    buckets: dict[tuple[int, float], list] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _deberta_layer_id(name, num_hidden_layers)
        layer_lr = learning_rate * (decay ** (max_depth - layer_id))
        wd = weight_decay if _param_applies_weight_decay(name) else 0.0
        key = (layer_id, wd)
        buckets.setdefault(key, []).append(param)

    groups = [
        {
            "params": params,
            "lr": learning_rate * (decay ** (max_depth - layer_id)),
            "weight_decay": wd,
        }
        for (layer_id, wd), params in sorted(buckets.items())
    ]
    if not groups:
        msg = "Layer-wise LR decay produced no optimizer parameter groups"
        raise RuntimeError(msg)
    return groups


def _layerwise_lr_summary(groups: list[dict[str, Any]]) -> tuple[float, float]:
    lrs = [float(g["lr"]) for g in groups]
    return min(lrs), max(lrs)


def _make_loader(
    dataset,
    tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Build a loader with repeatable shuffling for a given seed."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collator,
    }
    if shuffle:
        loader_kwargs["generator"] = generator
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if pin_memory:
        loader_kwargs["pin_memory"] = True
    return DataLoader(dataset, **loader_kwargs)


def _predict_proba(
    model,
    loader,
    device: str,
    *,
    amp_dtype=None,
    pin_memory: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (P(hate=1), labels, mean loss) over a dataloader."""
    import torch
    import torch.nn.functional as functional

    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            if amp_dtype is not None:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            probs = functional.softmax(outputs.logits.float(), dim=-1)[:, 1]
            scores.append(probs.detach().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels), float(np.mean(losses))


def _plot_training_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(frame["epoch"], frame["train_macro_f1"], marker="o", label="train (sample)")
    axes[0].plot(frame["epoch"], frame["dev_macro_f1"], marker="o", label="dev")
    axes[0].set_title("Macro-F1 per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("macro-F1")
    axes[0].set_xticks(frame["epoch"])
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(frame["epoch"], frame["train_loss"], marker="o", label="train")
    axes[1].plot(frame["epoch"], frame["dev_loss"], marker="o", label="dev")
    axes[1].set_title("Loss per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_xticks(frame["epoch"])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_step_losses(step_losses: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(step_losses)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["step"], frame["loss"], linewidth=0.8)
    ax.set_title("Train loss per optimizer step")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["non-hate (0)", "hate (1)"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_transformer_experiment(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    benchmark_steps: int = 0,
) -> Path:
    """Fine-tune a transformer classifier; if benchmark_steps > 0, time a short
    run and print a full-training estimate instead of training to completion."""
    _require_torch()

    import torch
    from torch.optim import AdamW
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    seed = int(config["seed"])
    set_global_seed(seed)
    torch.manual_seed(seed)

    device = select_device(str(config["model"].get("device", "auto")))
    model_cfg = _resolve_model_settings(config["model"], device)
    data_cfg = config["data"]
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))

    pretrained_name = str(model_cfg["pretrained_name"])
    max_length = int(model_cfg.get("max_length", 128))
    learning_rate = float(model_cfg.get("learning_rate", 1e-5))
    weight_decay = float(model_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(model_cfg.get("warmup_ratio", 0.06))
    batch_size = int(model_cfg.get("batch_size", 8))
    grad_accum = int(model_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(model_cfg.get("epochs", 3))
    max_grad_norm = float(model_cfg.get("max_grad_norm", 1.0))
    train_eval_sample = int(model_cfg.get("train_eval_sample", 5000))
    use_class_weight = model_cfg.get("class_weight") == "balanced"
    use_gradient_checkpointing = bool(model_cfg.get("gradient_checkpointing", False))
    num_workers = int(model_cfg.get("num_workers", 0))
    pin_memory = bool(model_cfg.get("pin_memory", device == "cuda"))
    use_bf16 = bool(model_cfg.get("bf16", False)) and device == "cuda"
    use_fp16 = bool(model_cfg.get("fp16", False)) and device == "cuda" and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)

    early_cfg = model_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 1))

    train_df, dev_df, test_df = load_canonical_dataset(
        str(config["dataset"]),
        train_path=data_cfg.get("train_path"),
        dev_path=data_cfg.get("dev_path"),
        test_path=data_cfg.get("test_path"),
    )
    x_train, y_train = xy_from_frame(train_df, text_column=text_column, label_column=label_column)
    x_dev, y_dev = xy_from_frame(dev_df, text_column=text_column, label_column=label_column)
    x_test, y_test = xy_from_frame(test_df, text_column=text_column, label_column=label_column)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def encode(texts: list[str]) -> dict[str, list[list[int]]]:
        return dict(tokenizer(list(texts), truncation=True, max_length=max_length))

    train_ds = _TextDataset(encode(x_train), np.asarray(y_train, dtype=int))
    dev_ds = _TextDataset(encode(x_dev), np.asarray(y_dev, dtype=int))
    test_ds = _TextDataset(encode(x_test), np.asarray(y_test, dtype=int))

    # Fixed stratified subsample of train for per-epoch curve evaluation
    # (full-train inference every epoch would add ~30% wall time).
    rng = np.random.default_rng(seed)
    if train_eval_sample and train_eval_sample < len(train_ds):
        labels_arr = np.asarray(y_train, dtype=int)
        sample_indices: list[int] = []
        for label in (0, 1):
            label_idx = np.flatnonzero(labels_arr == label)
            take = max(1, round(train_eval_sample * len(label_idx) / len(labels_arr)))
            sample_indices.extend(rng.choice(label_idx, size=min(take, len(label_idx)), replace=False))
        sample_indices = sorted(sample_indices)
        train_eval_ds = _TextDataset(
            {key: [values[i] for i in sample_indices] for key, values in train_ds.encodings.items()},
            labels_arr[sample_indices],
        )
    else:
        train_eval_ds = train_ds

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = _make_loader(
        train_ds, tokenizer, batch_size=batch_size, shuffle=True, seed=seed, **loader_kwargs
    )
    eval_batch = min(batch_size * 4, 128)
    eval_loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_eval_loader = _make_loader(
        train_eval_ds, tokenizer, batch_size=eval_batch, shuffle=False, seed=seed, **eval_loader_kwargs
    )
    dev_loader = _make_loader(
        dev_ds, tokenizer, batch_size=eval_batch, shuffle=False, seed=seed, **eval_loader_kwargs
    )
    test_loader = _make_loader(
        test_ds, tokenizer, batch_size=eval_batch, shuffle=False, seed=seed, **eval_loader_kwargs
    )

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=2)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    if amp_dtype is None:
        # DeBERTa (and some other encoders) can load with mixed dtypes on CUDA;
        # full fp32 training requires an explicit cast or the first optimizer step NaNs.
        model.float()

    def free_device_cache() -> None:
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    if use_class_weight:
        weights = _class_weights(pd.Series(np.asarray(y_train, dtype=int)))
        loss_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        loss_weights = None

    steps_per_epoch = -(-len(train_loader) // grad_accum)  # ceil division
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    llrd_cfg = model_cfg.get("layerwise_lr_decay", {})
    llrd_enabled = bool(llrd_cfg.get("enabled", False))
    llrd_decay = float(llrd_cfg.get("decay", 0.9))

    # foreach=False on MPS avoids large Adam temporaries; CUDA benefits from foreach=True.
    if llrd_enabled:
        param_groups = _build_layerwise_adamw_groups(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            decay=llrd_decay,
        )
        min_lr, max_lr = _layerwise_lr_summary(param_groups)
        optimizer = AdamW(param_groups, foreach=device == "cuda")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=device == "cuda",
        )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    def compute_loss(outputs, labels):
        if loss_weights is None:
            return outputs.loss
        return torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=loss_weights)

    def forward_batch(batch: dict[str, Any]):
        if amp_dtype is not None:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                outputs = model(**batch)
                loss = compute_loss(outputs, batch["labels"])
        else:
            outputs = model(**batch)
            loss = compute_loss(outputs, batch["labels"])
        return outputs, loss

    results_dir = resolve_results_dir(config)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device == "cuda":
        amp_label = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
        print(
            f"CUDA settings: batch={batch_size}, accum={grad_accum}, workers={num_workers}, "
            f"amp={amp_label}, checkpointing={use_gradient_checkpointing}"
        )
    print(f"Train rows: {len(train_ds):,} | dev: {len(dev_ds):,} | test: {len(test_ds):,}")
    if llrd_enabled:
        print(
            f"Layer-wise LR decay: decay={llrd_decay}, "
            f"base_lr={learning_rate:.2e}, lr range [{min_lr:.2e}, {max_lr:.2e}] "
            f"({len(param_groups)} groups)"
        )
    print(
        f"Steps/epoch: {steps_per_epoch:,} (micro-batches {len(train_loader):,} x accum {grad_accum}) "
        f"| total optimizer steps: {total_steps:,} | warmup: {warmup_steps:,}"
    )

    # ------------------------------------------------------------------ benchmark
    if benchmark_steps > 0:
        model.train()
        micro_needed = benchmark_steps * grad_accum
        timed = 0.0
        done_micro = 0
        warmup_micro = min(3 * grad_accum, micro_needed)
        iterator = iter(train_loader)
        while done_micro < micro_needed + warmup_micro:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            start = time.perf_counter()
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            _, loss = forward_batch(batch)
            scaled_loss = loss / grad_accum
            if use_fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if (done_micro + 1) % grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if device in {"mps", "cuda"}:
                if device == "mps":
                    torch.mps.synchronize()
                else:
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if done_micro >= warmup_micro:
                timed += elapsed
            done_micro += 1

        sec_per_step = timed / benchmark_steps
        epoch_train_sec = sec_per_step * steps_per_epoch
        # Eval passes per epoch: train sample + dev; final: train sample + dev + test.
        eval_rows_per_epoch = len(train_eval_ds) + len(dev_ds)
        # Inference is roughly 3x faster per row than a training step per row.
        rows_per_train_sec = (batch_size * grad_accum) / sec_per_step
        eval_sec_per_epoch = eval_rows_per_epoch / (rows_per_train_sec * 3)
        total_est = epochs * (epoch_train_sec + eval_sec_per_epoch)
        print(f"\nBenchmark over {benchmark_steps} optimizer steps ({device}):")
        print(f"  {sec_per_step:.2f} s/optimizer step ({rows_per_train_sec:.0f} rows/s)")
        print(f"  Estimated train time/epoch: {epoch_train_sec / 60:.1f} min")
        print(f"  Estimated eval time/epoch:  {eval_sec_per_epoch / 60:.1f} min")
        print(f"  Estimated total ({epochs} epochs): {total_est / 60:.0f} min ({total_est / 3600:.1f} h)")
        return results_dir

    # ------------------------------------------------------------------ training
    history: list[dict[str, Any]] = []
    step_losses: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_dir = results_dir / "model"
    global_step = 0
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        running_losses: list[float] = []
        optimizer.zero_grad()

        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            _, loss = forward_batch(batch)
            scaled_loss = loss / grad_accum
            if use_fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_losses.append(float(loss.detach().cpu()))

            if micro_step % grad_accum == 0 or micro_step == len(train_loader):
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 25 == 0:
                    recent = float(np.mean(running_losses[-25 * grad_accum:]))
                    step_losses.append({"step": global_step, "loss": recent})
                if global_step % 200 == 0:
                    pct = 100 * global_step / total_steps
                    print(
                        f"  step {global_step:,}/{total_steps:,} ({pct:.0f}%) "
                        f"loss={float(np.mean(running_losses[-200 * grad_accum:])):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

        train_time = time.perf_counter() - epoch_started
        free_device_cache()

        eval_started = time.perf_counter()
        train_scores, train_labels, train_eval_loss = _predict_proba(
            model, train_eval_loader, device, amp_dtype=amp_dtype, pin_memory=pin_memory
        )
        dev_scores, dev_labels, dev_loss = _predict_proba(
            model, dev_loader, device, amp_dtype=amp_dtype, pin_memory=pin_memory
        )
        eval_time = time.perf_counter() - eval_started

        from sklearn.metrics import f1_score

        train_f1 = float(f1_score(train_labels, (train_scores >= 0.5).astype(int), average="macro", zero_division=0))
        dev_f1 = float(f1_score(dev_labels, (dev_scores >= 0.5).astype(int), average="macro", zero_division=0))

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(running_losses)),
            "train_eval_loss": train_eval_loss,
            "dev_loss": dev_loss,
            "train_macro_f1": train_f1,
            "dev_macro_f1": dev_f1,
            "learning_rate_end": scheduler.get_last_lr()[0],
            "train_seconds": round(train_time, 1),
            "eval_seconds": round(eval_time, 1),
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs}: train_loss={record['train_loss']:.4f} "
            f"dev_loss={dev_loss:.4f} train_f1={train_f1:.4f} dev_f1={dev_f1:.4f} "
            f"({train_time / 60:.1f} min train + {eval_time / 60:.1f} min eval)",
            flush=True,
        )

        if dev_f1 > best_metric:
            best_metric = dev_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
        else:
            epochs_without_improvement += 1
            if early_enabled and epochs_without_improvement > patience:
                print(f"Early stopping after epoch {epoch} (best epoch {best_epoch}).")
                break

    total_training_seconds = time.perf_counter() - training_started

    # Reload best checkpoint for final evaluation.
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    threshold_cfg = config["training"].get("threshold_tuning", {})
    dev_scores, dev_labels, _ = _predict_proba(
        model, dev_loader, device, amp_dtype=amp_dtype, pin_memory=pin_memory
    )
    # Tune on dev once, then use the same decision rule for every reported split.
    threshold = 0.5
    dev_threshold_metric = None
    if threshold_cfg.get("enabled", False):
        threshold, dev_threshold_metric = tune_binary_threshold(
            dev_labels,
            dev_scores,
            metric=str(threshold_cfg.get("metric", "macro_f1")),
            min_threshold=float(threshold_cfg.get("min_threshold", 0.05)),
            max_threshold=float(threshold_cfg.get("max_threshold", 0.95)),
            step=float(threshold_cfg.get("step", 0.01)),
        )

    split_metrics: dict[str, Any] = {}
    confusion_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, loader in (("train", train_eval_loader), ("dev", dev_loader), ("test", test_loader)):
        scores, labels, _ = _predict_proba(
            model, loader, device, amp_dtype=amp_dtype, pin_memory=pin_memory
        )
        preds = apply_threshold(scores, threshold)
        split_metrics[split_name] = classification_metrics(labels, preds)
        confusion_inputs[split_name] = (labels, preds)

    def _split_class_distribution(labels) -> dict[str, int]:
        counts = pd.Series(np.asarray(labels, dtype=int)).value_counts().sort_index()
        return {str(int(key)): int(value) for key, value in counts.items()}

    artifact = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "seed": seed,
        "config_path": str(config_path) if config_path is not None else None,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "device": device,
        "primary_metric": config["training"].get("primary_metric", "macro_f1"),
        "threshold": threshold,
        "dev_threshold_macro_f1": dev_threshold_metric,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "total_training_seconds": round(total_training_seconds, 1),
        "train_eval_note": (
            f"train metrics computed on a fixed stratified sample of {len(train_eval_ds):,} rows"
            if len(train_eval_ds) < len(train_ds)
            else "train metrics computed on the full train split"
        ),
        "data": {
            "train_rows": len(train_df),
            "dev_rows": len(dev_df),
            "test_rows": len(test_df),
            "train_class_distribution": _split_class_distribution(y_train),
            "dev_class_distribution": _split_class_distribution(y_dev),
            "test_class_distribution": _split_class_distribution(y_test),
        },
        "history": history,
        "metrics": split_metrics,
        "model": model_cfg,
        "training": config["training"],
    }

    (results_dir / "metrics.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (results_dir / "config.resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _plot_training_curves(history, results_dir / "training_curves.png")
    if step_losses:
        _plot_step_losses(step_losses, results_dir / "step_losses.png")
    test_labels, test_preds = confusion_inputs["test"]
    _plot_confusion_matrix(test_labels, test_preds, results_dir / "confusion_matrix.png")

    return results_dir
