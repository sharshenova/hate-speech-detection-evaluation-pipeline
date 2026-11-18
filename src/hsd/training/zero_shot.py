"""Zero-shot LLM evaluation from YAML experiment configs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hsd.data.canonical import load_canonical_split, resolve_canonical_paths, xy_from_frame
from hsd.evaluation.metrics import classification_metrics
from hsd.evaluation.label_parse import parse_zero_shot_label
from hsd.prompts.loader import load_prompt_config, resolve_prompt_config_path
from hsd.utils.config import resolve_results_dir
from hsd.utils.seed import set_global_seed


def load_zero_shot_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a zero-shot experiment YAML config."""
    config_path = Path(path)
    if not config_path.is_file():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        msg = f"Config root must be a mapping: {config_path}"
        raise ValueError(msg)

    required = ("experiment_id", "dataset", "seed", "model", "prompt_config", "results_dir")
    missing = [key for key in required if key not in config]
    if missing:
        msg = f"Missing config keys: {missing}"
        raise ValueError(msg)

    return config


def _render_few_shot_block(prompt_cfg: dict[str, Any]) -> str:
    """Format configured examples for insertion into the user prompt."""
    examples = prompt_cfg.get("few_shot_examples") or []
    if not examples:
        return ""

    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        post = str(example["post"]).strip()
        label = str(example["label"]).strip()
        blocks.append(f"Example {index}:\nPost: {post}\nLabel: {label}")
    return "\n\n".join(blocks)


def _build_messages(
    prompt_cfg: dict[str, Any],
    text: str,
    *,
    enable_thinking: bool = False,
) -> list[dict[str, str]]:
    """Build chat messages and place few-shot examples in the user turn."""
    messages_cfg = prompt_cfg["messages"]
    user_template = str(messages_cfg["user"])
    few_shot_block = _render_few_shot_block(prompt_cfg)
    if few_shot_block:
        user_content = user_template.format(text=text, few_shot_block=few_shot_block)
    else:
        user_content = user_template.format(text=text, few_shot_block="")
    messages: list[dict[str, str]] = []
    if "system" in messages_cfg:
        system_content = str(messages_cfg["system"]).strip()
    else:
        system_content = ""
    # Gemma 4: prefix system with <|think|> to enable reasoning mode (Ollama docs).
    if enable_thinking:
        if system_content:
            system_content = f"<|think|>\n{system_content}"
        else:
            system_content = "<|think|>"
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content.strip()})
    return messages


def _resolve_eval_frame(config: dict[str, Any], prompt_cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the configured dev or test split, with the prompt default as fallback."""
    data_cfg = config.get("data", {})
    dataset = str(config["dataset"])
    _, dev_path, test_path = resolve_canonical_paths(
        dataset,
        train_path=data_cfg.get("train_path"),
        dev_path=data_cfg.get("dev_path"),
        test_path=data_cfg.get("test_path"),
    )
    eval_split = str(config.get("eval_split", prompt_cfg.get("eval_split", "dev")))
    if eval_split == "dev":
        return load_canonical_split(dev_path)
    if eval_split == "test":
        return load_canonical_split(test_path)
    msg = f"Unsupported eval_split: {eval_split!r}"
    raise ValueError(msg)


def _pick_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_and_tokenizer(model_cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(model_cfg["name_or_path"])
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(device)

    model.eval()
    return model, tokenizer, device


def _generate_batch(
    model: Any,
    tokenizer: Any,
    messages_batch: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
    device: Any,
) -> list[str]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        for messages in messages_batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    if device.type != "cuda":
        inputs = {key: value.to(device) for key, value in inputs.items()}

    # With left padding, generated tokens start at the padded prompt width.
    prompt_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses: list[str] = []
    for output_ids in outputs:
        generated_ids = output_ids[prompt_width:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(text.strip())
    return responses


def _generate_ollama(
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
) -> str:
    """Send one chat to Ollama and preserve any separate thinking text."""
    import urllib.error
    import urllib.request

    base_url = str(model_cfg.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
    model_name = str(model_cfg["name_or_path"])
    timeout_s = int(model_cfg.get("request_timeout_s", 300))
    temperature = float(model_cfg.get("temperature", 0))
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        },
    }
    stop = model_cfg.get("stop")
    if stop:
        body["options"]["stop"] = list(stop)
    # Ollama Gemma 4: explicit think flag when supported by the server.
    if "think" in model_cfg:
        body["think"] = bool(model_cfg["think"])
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        msg = f"Ollama request failed for model {model_name!r}: {exc}"
        raise RuntimeError(msg) from exc

    message = data.get("message", {})
    content = str(message.get("content", "")).strip()
    thinking = message.get("thinking")
    if thinking and content:
        return f"<|channel>thought\n{thinking}<channel|>{content}"
    if thinking and not content:
        return str(thinking).strip()
    return content


def _predict_zero_shot_responses(
    texts: list[str],
    prompt_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> list[str]:
    """Route configured Ollama models per row and local models in batches."""
    model_type = str(model_cfg.get("type", "transformers_causal_lm"))
    max_new_tokens = int(model_cfg.get("max_new_tokens", 8))
    enable_thinking = bool(model_cfg.get("enable_thinking", False))
    responses: list[str] = []
    total = len(texts)

    if model_type == "ollama_chat":
        log_every = int(model_cfg.get("log_every", 100))
        for index, text in enumerate(texts, start=1):
            messages = _build_messages(
                prompt_cfg,
                text,
                enable_thinking=enable_thinking,
            )
            responses.append(
                _generate_ollama(
                    model_cfg,
                    messages,
                    max_new_tokens=max_new_tokens,
                )
            )
            if index == 1 or index % log_every == 0 or index == total:
                print(f"  ollama progress: {index}/{total}", flush=True)
        return responses

    batch_size = int(model_cfg.get("batch_size", 8))
    model, tokenizer, device = _load_model_and_tokenizer(model_cfg)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        messages_batch = [
            _build_messages(prompt_cfg, text, enable_thinking=enable_thinking)
            for text in batch_texts
        ]
        responses.extend(
            _generate_batch(
                model,
                tokenizer,
                messages_batch,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        )
        print(f"  transformers progress: {min(start + batch_size, total)}/{total}", flush=True)
    return responses

def run_zero_shot_experiment(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    limit: int | None = None,
) -> Path:
    """Run zero-shot LLM classification and write metrics artifacts."""
    seed = int(config["seed"])
    set_global_seed(seed)

    prompt_path = resolve_prompt_config_path(config["prompt_config"])
    prompt_cfg = load_prompt_config(prompt_path)
    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))

    eval_df = _resolve_eval_frame(config, prompt_cfg)
    if limit is not None:
        eval_df = eval_df.head(limit).copy()

    texts, labels = xy_from_frame(
        eval_df,
        text_column=text_column,
        label_column=label_column,
    )

    raw_responses = _predict_zero_shot_responses(texts, prompt_cfg, model_cfg)

    parsed_labels: list[int] = []
    normalized_responses: list[str] = []
    parse_failures = 0

    for raw in raw_responses:
        label, normalized = parse_zero_shot_label(raw, prompt_cfg["output"])
        if label is None:
            # A conservative fallback keeps parse failures visible without adding false positives.
            parse_failures += 1
            label = 0
        normalized_responses.append(normalized)
        parsed_labels.append(label)

    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(parsed_labels, dtype=int)
    split_name = str(config.get("eval_split", prompt_cfg.get("eval_split", "dev")))
    metrics = classification_metrics(y_true, y_pred)

    results_dir = resolve_results_dir(config)
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions = eval_df.copy()
    predictions["y_true"] = y_true
    predictions["y_pred"] = y_pred
    predictions["model_response"] = raw_responses
    predictions["normalized_response"] = normalized_responses
    predictions.to_csv(results_dir / "predictions.tsv", sep="\t", index=False)

    artifact = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "seed": seed,
        "config_path": str(config_path) if config_path is not None else None,
        "prompt_config_path": str(prompt_path),
        "prompt_id": prompt_cfg["prompt_id"],
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "eval_split": split_name,
        "primary_metric": "macro_f1",
        "parse_failures": parse_failures,
        "eval_rows": len(eval_df),
        "class_distribution": {
            str(int(key)): int(value)
            for key, value in pd.Series(labels).value_counts().sort_index().items()
        },
        "metrics": {split_name: metrics},
        "model": model_cfg,
        "prompt": {
            "prompt_id": prompt_cfg["prompt_id"],
            "name": prompt_cfg.get("name"),
            "description": prompt_cfg.get("description"),
        },
    }

    metrics_path = results_dir / "metrics.json"
    config_copy_path = results_dir / "config.resolved.json"
    metrics_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    config_copy_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    prompt_copy_path = results_dir / "prompt.resolved.yaml"
    prompt_copy_path.write_text(
        yaml.safe_dump(prompt_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return results_dir
