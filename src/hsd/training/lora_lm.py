"""LoRA / QLoRA fine-tuning for instruction-tuned causal LMs (binary hate classification)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsd.data.canonical import load_canonical_dataset, xy_from_frame
from hsd.evaluation.metrics import apply_threshold, classification_metrics, tune_binary_threshold
from hsd.prompts.loader import load_prompt_config, resolve_prompt_config_path
from hsd.training.transformer import _plot_confusion_matrix, _plot_training_curves, select_device
from hsd.evaluation.label_parse import parse_zero_shot_label
from hsd.utils.config import resolve_results_dir
from hsd.utils.seed import set_global_seed


def _require_lora_deps() -> None:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        msg = "LoRA training requires: pip install -e '.[dev,ml,lora]'"
        raise RuntimeError(msg) from exc


def _use_qlora(model_cfg: dict[str, Any], device: str) -> bool:
    """Enable 4-bit QLoRA only when requested on a CUDA device."""
    return bool(model_cfg.get("qlora", True)) and device == "cuda"


def _chat_template_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extra kwargs for tokenizer.apply_chat_template (e.g. Qwen3.5 thinking mode)."""
    if "enable_thinking" not in model_cfg:
        return {}
    return {"enable_thinking": bool(model_cfg["enable_thinking"])}


def _resolve_model_settings(model_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Apply CUDA-specific model settings after the shared settings."""
    resolved = dict(model_cfg)
    if device == "cuda" and isinstance(model_cfg.get("cuda"), dict):
        resolved.update(model_cfg["cuda"])
    return resolved


def resolve_lora_prompt_config(config: dict[str, Any]) -> dict[str, Any]:
    """Select the prompt assigned to the current training corpus."""
    dataset = str(config["dataset"])
    model_cfg = config["model"]
    mapping = model_cfg.get("prompt_config_by_dataset", {})
    if dataset not in mapping:
        msg = f"No prompt_config for dataset {dataset!r} in model.prompt_config_by_dataset"
        raise KeyError(msg)
    path = resolve_prompt_config_path(str(mapping[dataset]))
    return load_prompt_config(path)


def _label_text(label: int) -> str:
    return "hate" if int(label) == 1 else "non-hate"


def _build_sft_messages(
    prompt_cfg: dict[str, Any],
    text: str,
    label: int | None = None,
) -> list[dict[str, str]]:
    """Build the same chat turns for supervised training and inference."""
    messages_cfg = prompt_cfg["messages"]
    user_content = str(messages_cfg["user"]).format(text=text).strip()
    messages: list[dict[str, str]] = []
    if "system" in messages_cfg:
        messages.append({"role": "system", "content": str(messages_cfg["system"]).strip()})
    messages.append({"role": "user", "content": user_content})
    if label is not None:
        messages.append({"role": "assistant", "content": _label_text(label)})
    return messages


def _truncate_text(text: str, max_chars: int = 900) -> str:
    """Limit very long posts before chat formatting so the instruction remains in context."""
    cleaned = str(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


@dataclass
class _TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def _as_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        return [int(value) for value in tokenized["input_ids"]]
    if hasattr(tokenized, "input_ids"):
        ids = tokenized.input_ids
        if isinstance(ids[0], list):
            return [int(value) for value in ids[0]]
        return [int(value) for value in ids]
    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            return [int(value) for value in tokenized[0]]
        return [int(value) for value in tokenized]
    msg = f"Unexpected tokenized type: {type(tokenized)!r}"
    raise TypeError(msg)


def _tokenize_sft_example(
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    text: str,
    label: int,
    *,
    max_length: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> _TokenizedExample:
    """Tokenize one chat and train only on the assistant label tokens.

    If the chat is too long, keep its tail because it contains the answer slot.
    """
    clipped = _truncate_text(text)
    messages = _build_sft_messages(prompt_cfg, clipped, label)
    prompt_messages = messages[:-1]
    template_kwargs = chat_template_kwargs or {}
    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            **template_kwargs,
        )
    )
    full_ids = _as_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, **template_kwargs)
    )
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        prompt_len = min(len(prompt_ids), max(1, len(full_ids) - 4))
    else:
        prompt_len = len(prompt_ids)

    label_ids = [-100] * len(full_ids)
    for index in range(prompt_len, len(full_ids)):
        label_ids[index] = full_ids[index]

    attention_mask = [1] * len(full_ids)
    return _TokenizedExample(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=label_ids,
    )


class _SFTDataset:
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        *,
        tokenizer: Any,
        prompt_cfg: dict[str, Any],
        max_length: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.texts = texts
        self.labels = np.asarray(labels, dtype=int)
        self.tokenizer = tokenizer
        self.prompt_cfg = prompt_cfg
        self.max_length = max_length
        self.chat_template_kwargs = chat_template_kwargs or {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        example = _tokenize_sft_example(
            self.tokenizer,
            self.prompt_cfg,
            self.texts[idx],
            int(self.labels[idx]),
            max_length=self.max_length,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


class _SFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_inference_peft_model(
    model_cfg: dict[str, Any],
    adapter_dir: Path,
    device: str,
) -> tuple[Any, Any]:
    """Reload a saved adapter on its base model for final evaluation."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_qlora = _use_qlora(model_cfg, device)
    if use_qlora:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if model_cfg.get("bf16", True) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        dtype = (
            torch.bfloat16 if device == "cuda" and model_cfg.get("bf16", True) else torch.float32
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": True,
        }
        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device != "cuda":
            base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    return model, tokenizer


def _load_qlora_model(model_cfg: dict[str, Any], device: str) -> tuple[Any, Any]:
    """Load the base model and attach trainable LoRA layers."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_qlora = _use_qlora(model_cfg, device)
    if use_qlora:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if model_cfg.get("bf16", True) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = (
            torch.bfloat16 if device == "cuda" and model_cfg.get("bf16", True) else torch.float32
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": True,
        }
        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device != "cuda":
            model.to(device)

    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=int(model_cfg.get("lora_r", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(
            model_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def _sample_indices(labels: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Choose a repeatable class-aware sample for faster training evaluation."""
    if sample_size <= 0 or sample_size >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, sample_size // len(classes))
    chosen: list[int] = []
    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        take = min(per_class, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    remaining = sample_size - len(chosen)
    if remaining > 0:
        pool = [idx for idx in range(len(labels)) if idx not in chosen]
        if pool:
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            chosen.extend(extra.tolist())
    return np.array(sorted(set(chosen)), dtype=int)


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Per-example weights for sklearn-style balanced sampling."""
    labels = np.asarray(labels, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(classes)
    class_weight = {
        int(cls): n_samples / (n_classes * int(count)) for cls, count in zip(classes, counts)
    }
    return np.asarray([class_weight[int(label)] for label in labels], dtype=np.float64)


def _predict_batch_generative(
    model: Any,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    texts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    device: str,
    max_length: int = 256,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[str], int]:
    """Greedily generate labels and treat unparseable answers as non-hate."""
    import torch

    output_cfg = prompt_cfg.get("output", {})
    predictions: list[int] = []
    raw_responses: list[str] = []
    parse_failures = 0

    # Match training truncation (keep the tail: instruction + answer slot).
    tokenizer.truncation_side = "left"

    model.eval()
    template_kwargs = chat_template_kwargs or {}
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                _build_sft_messages(prompt_cfg, _truncate_text(text)),
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            for text in batch_texts
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if device != "cuda" or not hasattr(model, "hf_device_map"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # With left padding, generated tokens start at the padded prompt width
        # for every row, NOT at attention_mask.sum() (that slices into the prompt).
        prompt_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for output_ids in outputs:
            generated_ids = output_ids[prompt_width:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            label, _ = parse_zero_shot_label(raw, output_cfg)
            if label is None:
                parse_failures += 1
                label = 0
            predictions.append(label)
            raw_responses.append(raw)
    return predictions, raw_responses, parse_failures


def _candidate_first_token_ids(tokenizer: Any, label_strings: list[str]) -> list[int]:
    """Collect unique first-token ids for accepted label spellings."""
    token_ids: list[int] = []
    seen: set[int] = set()
    for label in label_strings:
        encoded = tokenizer.encode(str(label), add_special_tokens=False)
        if not encoded:
            continue
        token_id = int(encoded[0])
        if token_id not in seen:
            seen.add(token_id)
            token_ids.append(token_id)
    if not token_ids:
        msg = f"No tokenizer ids for label strings: {label_strings!r}"
        raise ValueError(msg)
    return token_ids


def _resolve_score_token_ids(
    tokenizer: Any, prompt_cfg: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """Map the prompt parser's accepted labels to scoreable first tokens."""
    output_cfg = prompt_cfg.get("output", {})
    hate_labels = [str(value) for value in output_cfg.get("parse_as_hate", ["hate", "yes"])]
    non_hate_labels = [
        str(value) for value in output_cfg.get("parse_as_non_hate", ["non-hate", "nonhate", "no"])
    ]
    return (
        _candidate_first_token_ids(tokenizer, hate_labels),
        _candidate_first_token_ids(tokenizer, non_hate_labels),
    )


def hate_prob_from_logits_row(
    row_logits: np.ndarray,
    hate_token_ids: list[int],
    non_hate_token_ids: list[int],
) -> float:
    """Compute P(hate) from the next-token logits.

    Log-sum-exp combines alternative accepted spellings within each class.
    """
    hate_ids = np.asarray(hate_token_ids, dtype=int)
    hate_logit = float(np.logaddexp.reduce(row_logits[hate_ids]))
    non_hate_ids = np.asarray(non_hate_token_ids, dtype=int)
    non_hate_logit = float(np.logaddexp.reduce(row_logits[non_hate_ids]))
    stacked = np.array([non_hate_logit, hate_logit], dtype=np.float64)
    stacked -= stacked.max()
    probs = np.exp(stacked)
    return float(probs[1] / probs.sum())


def _predict_batch_hate_scores(
    model: Any,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    texts: list[str],
    *,
    batch_size: int,
    device: str,
    max_length: int = 256,
    chat_template_kwargs: dict[str, Any] | None = None,
    hate_token_ids: list[int] | None = None,
    non_hate_token_ids: list[int] | None = None,
) -> np.ndarray:
    """Score hate versus non-hate at the first assistant token."""
    import torch

    if hate_token_ids is None or non_hate_token_ids is None:
        hate_token_ids, non_hate_token_ids = _resolve_score_token_ids(tokenizer, prompt_cfg)

    tokenizer.truncation_side = "left"
    template_kwargs = chat_template_kwargs or {}
    scores: list[float] = []

    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                _build_sft_messages(prompt_cfg, _truncate_text(text)),
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            for text in batch_texts
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if device != "cuda" or not hasattr(model, "hf_device_map"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
        last_logits = logits[:, -1, :].float().detach().cpu().numpy()
        for row_logits in last_logits:
            scores.append(
                hate_prob_from_logits_row(row_logits, hate_token_ids, non_hate_token_ids)
            )
    return np.asarray(scores, dtype=np.float64)


def predict_hate_scores(
    model: Any,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    texts: list[str],
    model_cfg: dict[str, Any],
) -> np.ndarray:
    """Return P(hate) for each text using first-token logits at the answer position."""
    batch_size = int(model_cfg.get("inference_batch_size", 16))
    device = str(model_cfg.get("_device", "cuda"))
    return _predict_batch_hate_scores(
        model,
        tokenizer,
        prompt_cfg,
        texts,
        batch_size=batch_size,
        device=device,
        max_length=int(model_cfg.get("max_length", 256)),
        chat_template_kwargs=_chat_template_kwargs(model_cfg),
    )


def evaluate_lora_logit_threshold(
    model: Any,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    *,
    splits: dict[str, tuple[list[str], np.ndarray]],
) -> dict[str, Any]:
    """Score splits with answer-token logits; tune threshold on dev; apply to all splits."""
    threshold_cfg = config.get("training", {}).get("threshold_tuning", {})
    tune_split = str(threshold_cfg.get("split", "dev"))
    if tune_split not in splits:
        msg = f"Threshold tuning split {tune_split!r} not in provided splits"
        raise KeyError(msg)

    tune_texts, tune_labels = splits[tune_split]
    print(f"Scoring {tune_split} for threshold tuning ({len(tune_texts):,} rows)...", flush=True)
    tune_scores = predict_hate_scores(model, tokenizer, prompt_cfg, tune_texts, model_cfg)
    threshold = 0.5
    dev_threshold_metric = None
    if bool(threshold_cfg.get("enabled", True)):
        threshold, dev_threshold_metric = tune_binary_threshold(
            tune_labels,
            tune_scores,
            metric=str(threshold_cfg.get("metric", "macro_f1")),
            min_threshold=float(threshold_cfg.get("min_threshold", 0.05)),
            max_threshold=float(threshold_cfg.get("max_threshold", 0.95)),
            step=float(threshold_cfg.get("step", 0.01)),
        )
        print(
            f"Dev-tuned threshold={threshold:.3f} dev_macro_f1={dev_threshold_metric:.4f}",
            flush=True,
        )

    hate_token_ids, non_hate_token_ids = _resolve_score_token_ids(tokenizer, prompt_cfg)
    split_metrics: dict[str, Any] = {}
    split_scores: dict[str, list[float]] = {}
    for split_name, (texts, labels) in splits.items():
        print(f"Logit eval: {split_name} ({len(texts):,} rows)...", flush=True)
        scores = (
            tune_scores
            if split_name == tune_split
            else predict_hate_scores(model, tokenizer, prompt_cfg, texts, model_cfg)
        )
        preds = apply_threshold(scores, threshold)
        split_metrics[split_name] = classification_metrics(labels, preds)
        split_scores[split_name] = scores.tolist()
        print(
            f"  {split_name}: acc={split_metrics[split_name]['accuracy']:.4f} "
            f"f1_macro={split_metrics[split_name]['f1_macro']:.4f}",
            flush=True,
        )

    return {
        "method": "logit_threshold",
        "threshold": threshold,
        "dev_threshold_macro_f1": dev_threshold_metric,
        "hate_token_ids": hate_token_ids,
        "non_hate_token_ids": non_hate_token_ids,
        "metrics": split_metrics,
        "scores": split_scores,
    }


def _evaluate_split(
    model: Any,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    texts: list[str],
    labels: np.ndarray,
    model_cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, int]:
    batch_size = int(model_cfg.get("inference_batch_size", 16))
    max_new_tokens = int(model_cfg.get("max_new_tokens", 8))
    device = str(model_cfg.get("_device", "cuda"))
    template_kwargs = _chat_template_kwargs(model_cfg)
    preds, _, parse_failures = _predict_batch_generative(
        model,
        tokenizer,
        prompt_cfg,
        texts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        max_length=int(model_cfg.get("max_length", 256)),
        chat_template_kwargs=template_kwargs,
    )
    y_pred = np.asarray(preds, dtype=int)
    metrics = classification_metrics(labels, y_pred)
    return metrics, y_pred, parse_failures


def run_lora_experiment(config: dict[str, Any], *, config_path: Path | None = None) -> Path:
    """Fine-tune a causal LM with QLoRA and evaluate via generative labels."""
    _require_lora_deps()
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    seed = int(config["seed"])
    set_global_seed(seed)

    prompt_cfg = resolve_lora_prompt_config(config)
    model_cfg = _resolve_model_settings(
        config["model"], select_device(str(config["model"].get("device", "auto")))
    )
    device = select_device(str(model_cfg.get("device", "auto")))
    model_cfg["_device"] = device

    data_cfg = config["data"]
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))

    train_df, dev_df, test_df = load_canonical_dataset(str(config["dataset"]))
    x_train, y_train = xy_from_frame(train_df, text_column=text_column, label_column=label_column)
    x_dev, y_dev = xy_from_frame(dev_df, text_column=text_column, label_column=label_column)
    x_test, y_test = xy_from_frame(test_df, text_column=text_column, label_column=label_column)

    train_max_rows = model_cfg.get("train_max_rows")
    if train_max_rows is not None:
        train_idx = _sample_indices(
            np.asarray(y_train, dtype=int),
            int(train_max_rows),
            seed,
        )
        x_train = [x_train[i] for i in train_idx]
        y_train = np.asarray(y_train, dtype=int)[train_idx]
        print(
            f"Train subsample: {len(x_train):,} rows (train_max_rows={train_max_rows})",
            flush=True,
        )

    max_length = int(model_cfg.get("max_length", 256))
    batch_size = int(model_cfg.get("batch_size", 2))
    grad_accum = int(model_cfg.get("gradient_accumulation_steps", 8))
    epochs = int(model_cfg.get("epochs", 3))
    learning_rate = float(model_cfg.get("learning_rate", 2e-4))
    weight_decay = float(model_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(model_cfg.get("warmup_ratio", 0.06))
    max_grad_norm = float(model_cfg.get("max_grad_norm", 1.0))

    model, tokenizer = _load_qlora_model(model_cfg, device)
    model_device = next(model.parameters()).device
    template_kwargs = _chat_template_kwargs(model_cfg)

    train_ds = _SFTDataset(
        x_train,
        np.asarray(y_train, dtype=int),
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        max_length=max_length,
        chat_template_kwargs=template_kwargs,
    )

    collator = _SFTCollator(pad_token_id=tokenizer.pad_token_id)
    num_workers = int(model_cfg.get("num_workers", 0))
    use_balanced_sampling = bool(model_cfg.get("class_balanced_sampling", False))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collator,
    }
    if use_balanced_sampling:
        import torch
        from torch.utils.data import WeightedRandomSampler

        sample_weights = _balanced_sample_weights(np.asarray(y_train, dtype=int))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        loader_kwargs["sampler"] = sampler
        print(
            f"Class-balanced sampling enabled (train rows={len(train_ds):,})",
            flush=True,
        )
    else:
        loader_kwargs["shuffle"] = True
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, **loader_kwargs)

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, (len(train_loader) // grad_accum) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_eval_sample = int(model_cfg.get("epoch_eval_sample", 2000))
    dev_eval_idx = _sample_indices(np.asarray(y_dev, dtype=int), epoch_eval_sample, seed)
    dev_eval_texts = [x_dev[i] for i in dev_eval_idx]
    dev_eval_labels = np.asarray(y_dev, dtype=int)[dev_eval_idx]

    early_cfg = model_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", 1))

    results_dir = resolve_results_dir(config)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = results_dir / "model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_dev_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        train_epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(model_device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            running_losses.append(float(outputs.loss.detach().cpu()))

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_seconds = time.perf_counter() - train_epoch_start
        train_loss = float(np.mean(running_losses)) if running_losses else 0.0

        eval_start = time.perf_counter()
        dev_metrics, _, dev_parse_failures = _evaluate_split(
            model,
            tokenizer,
            prompt_cfg,
            dev_eval_texts,
            dev_eval_labels,
            model_cfg,
        )
        eval_seconds = time.perf_counter() - eval_start
        dev_f1 = float(dev_metrics["f1_macro"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": None,
            "train_macro_f1": None,
            "dev_macro_f1": dev_f1,
            "dev_parse_failures": dev_parse_failures,
            "learning_rate_end": scheduler.get_last_lr()[0],
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} dev_f1={dev_f1:.4f} "
            f"({train_seconds / 60:.1f} min train + {eval_seconds / 60:.1f} min eval)",
            flush=True,
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
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

    del model
    del train_loader
    del optimizer
    del scheduler
    if device == "cuda":
        torch.cuda.empty_cache()

    # Final metrics must use the best saved epoch, not the last epoch in memory.
    model, tokenizer = _load_inference_peft_model(model_cfg, best_model_dir, device)

    train_eval_sample = int(model_cfg.get("train_eval_sample", 2000))
    train_eval_idx = _sample_indices(np.asarray(y_train, dtype=int), train_eval_sample, seed + 1)
    train_eval_texts = [x_train[i] for i in train_eval_idx]
    train_eval_labels = np.asarray(y_train, dtype=int)[train_eval_idx]

    split_metrics: dict[str, Any] = {}
    split_parse_failures: dict[str, int] = {}
    confusion_preds: np.ndarray | None = None
    for split_name, texts, labels in (
        ("train", train_eval_texts, train_eval_labels),
        ("dev", x_dev, np.asarray(y_dev, dtype=int)),
        ("test", x_test, np.asarray(y_test, dtype=int)),
    ):
        print(f"Final eval: {split_name} ({len(texts)} rows)...", flush=True)
        metrics, preds, parse_failures = _evaluate_split(
            model,
            tokenizer,
            prompt_cfg,
            texts,
            labels,
            model_cfg,
        )
        split_metrics[split_name] = metrics
        split_parse_failures[split_name] = parse_failures
        if split_name == "test":
            confusion_preds = preds
        print(
            f"  {split_name} f1_macro={metrics['f1_macro']:.4f} parse_failures={parse_failures}",
            flush=True,
        )

    def _split_class_distribution(labels: np.ndarray) -> dict[str, int]:
        counts = pd.Series(labels).value_counts().sort_index()
        return {str(int(key)): int(value) for key, value in counts.items()}

    artifact = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "seed": seed,
        "config_path": str(config_path) if config_path is not None else None,
        "prompt_id": prompt_cfg.get("prompt_id"),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "device": device,
        "primary_metric": config["training"].get("primary_metric", "macro_f1"),
        "threshold": None,
        "dev_threshold_macro_f1": None,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "total_training_seconds": round(total_training_seconds, 1),
        "class_balanced_sampling": use_balanced_sampling,
        "parse_failures": split_parse_failures,
        "train_eval_note": (
            f"train metrics computed on a stratified sample of {len(train_eval_texts):,} rows"
            if len(train_eval_texts) < len(x_train)
            else "train metrics computed on the full train split"
        ),
        "epoch_eval_note": f"epoch dev F1 on stratified sample of {len(dev_eval_texts):,} dev rows",
        "data": {
            "train_rows": len(train_df),
            "dev_rows": len(dev_df),
            "test_rows": len(test_df),
            "train_class_distribution": _split_class_distribution(np.asarray(y_train, dtype=int)),
            "dev_class_distribution": _split_class_distribution(np.asarray(y_dev, dtype=int)),
            "test_class_distribution": _split_class_distribution(np.asarray(y_test, dtype=int)),
        },
        "history": history,
        "metrics": split_metrics,
        "model": model_cfg,
        "training": config["training"],
    }

    (results_dir / "metrics.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (results_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    _plot_training_curves(history, results_dir / "training_curves.png")
    if confusion_preds is not None:
        _plot_confusion_matrix(
            np.asarray(y_test, dtype=int), confusion_preds, results_dir / "confusion_matrix.png"
        )

    return results_dir
