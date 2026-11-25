#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

So: QLoRA is the enabler, whisper-large-v3 is the lever. See RECOMMENDED RUNS.

Augmentation: additive Gaussian noise ONLY. 

One deliberate protocol improvement: augmentation is applied **on the fly** per
epoch rather than baked in once by a `.map()` over the dataset. Each epoch now
sees fresh noise, which is what the augmentation was meant to do, and it also
avoids materialising 128-mel features for the whole corpus.

RECOMMENDED RUNS
----------------
Arm A — isolates the cost of quantization:
    python swahili_asr_qlora.py --base-model openai/whisper-small --epochs 25

Arm B — the sub-10% attempt:
    python swahili_asr_qlora.py --base-model openai/whisper-large-v3 --epochs 5

Arm C — optional layer-adaptive rank on top of Arm B:
    python swahili_asr_qlora.py --base-model openai/whisper-large-v3 --epochs 5 \
        --layer-adaptive

Run each arm over >= 3 seeds (--seed) before claiming any ordering. 

SETUP (Colab, Tesla T4)
-----------------------
    !pip install -q "transformers>=4.46" "peft>=0.13" datasets \
                    accelerate "bitsandbytes>=0.46" evaluate jiwer soundfile librosa soxr

HOW TO LAUNCH IT FROM A NOTEBOOK
--------------------------------
Do NOT `%run` this file or paste it into a cell and call `main()` with no arguments:
a Jupyter kernel is started as `... -f /root/.../kernel-<uuid>.json`, and that stray
`-f` lands in `sys.argv`. Either of these is correct:

    # (a) as a subprocess — output streams into the cell
    !python swahili_asr_qlora.py --base-model openai/whisper-large-v3 --epochs 5

    # (b) in-process — keeps the model and trainer around for inspection afterwards
    from swahili_asr_qlora import main
    main(["--base-model", "openai/whisper-large-v3", "--epochs", "5"])

(b) is usually the better fit on Colab: tqdm renders properly and a dropped cell does
not kill the kernel. `--resume <checkpoint>` picks up a run that was interrupted.

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("swahili_qlora")

SAMPLING_RATE = 16_000
THESIS_SPLIT_SEED = 42          # reproduces the exact 7200/900/900 clip-level split
NOISE_MIN_AMPLITUDE = 0.001     
NOISE_MAX_AMPLITUDE = 0.010     
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Lower-case, drop punctuation except the apostrophe, collapse whitespace.


    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text).lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------------------
# Augmentation — additive Gaussian noise only 
# --------------------------------------------------------------------------------------

def add_gaussian_noise(
    samples: np.ndarray,
    rng: np.random.Generator,
    p: float = 0.3,
    min_amplitude: float = NOISE_MIN_AMPLITUDE,
    max_amplitude: float = NOISE_MAX_AMPLITUDE,
) -> np.ndarray:

    if p <= 0.0 or rng.random() >= p:
        return samples
    sigma = rng.uniform(min_amplitude, max_amplitude)
    noise = rng.standard_normal(samples.shape).astype(np.float32)
    return (samples + sigma * noise).astype(np.float32)


# --------------------------------------------------------------------------------------
# WER metric
# --------------------------------------------------------------------------------------

class _JiwerWer:
    """Minimal stand-in for `evaluate.load('wer')`.

    `evaluate`'s WER metric is a thin wrapper over jiwer, so this returns identical
    numbers — it just avoids a Hub download that can fail behind a proxy or offline.
    """

    def compute(self, predictions: List[str], references: List[str]) -> float:
        import jiwer
        return float(jiwer.wer(list(references), list(predictions)))


def load_wer_metric():
    """Prefer the `evaluate` metric ; fall back to jiwer."""
    try:
        import evaluate as hf_evaluate
        metric = hf_evaluate.load("wer")
        logger.info("WER metric: huggingface `evaluate`")
        return metric
    except Exception as exc:
        logger.warning("Could not load `evaluate` WER (%s); using jiwer directly "
                       "(numerically identical).", type(exc).__name__)
        return _JiwerWer()


# --------------------------------------------------------------------------------------
# Log noise
# --------------------------------------------------------------------------------------

_LENGTH_WARNING_FILTERED = False


def silence_generation_length_warning() -> None:
    """Drop transformers' "Both `max_new_tokens` and `max_length` seem to have been set".

    It is emitted once per generation call, so a 900-clip evaluation buries the run in
    hundreds of identical lines. It is purely informational: max_new_tokens wins, which
    is exactly what we want.

    Note the two non-fixes, both tried and rejected:
      * Setting `generation_config.max_length = None` does NOT help — generate falls
        back to the library default of 20 tokens and silently truncates transcripts.
      * Moving max_new_tokens onto the generation_config does not help either; Whisper's
        generate sets max_length internally regardless, so the warning still fires.
    Dropping max_new_tokens entirely does silence it, but then generation runs to
    Whisper's full 448-token window instead of 225, roughly doubling evaluation time.
    """
    global _LENGTH_WARNING_FILTERED
    if _LENGTH_WARNING_FILTERED:
        return

    class _DropLengthWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "seem to have been set" not in record.getMessage()

    flt = _DropLengthWarning()
    # A filter on a logger only sees records made by that logger, not propagated ones,
    # so attach to the emitting module loggers AND to the library handler.
    for name in ("transformers.generation.utils",
                 "transformers.generation.configuration_utils",
                 "transformers"):
        logging.getLogger(name).addFilter(flt)
    for handler in logging.getLogger("transformers").handlers:
        handler.addFilter(flt)

    _LENGTH_WARNING_FILTERED = True


# --------------------------------------------------------------------------------------
# LoRA / QLoRA configuration
# --------------------------------------------------------------------------------------

def build_lora_kwargs(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    layer_adaptive: bool = False,
    encoder_r: int = 64,
    decoder_r: int = 16,
    encoder_alpha: Optional[int] = None,
    decoder_alpha: Optional[int] = None,
) -> Dict[str, Any]:
    """Return kwargs for `peft.LoraConfig`.

    Default (layer_adaptive=False) : r=32, alpha=64,
    dropout=0.05 on q/k/v/out/fc1/fc2.

    With layer_adaptive=True, ranks follow Han et al. (2024): a higher rank on the
    encoder (acoustic adaptation is where the Swahili-specific work happens) and a
    lower rank on the decoder. `alpha` is scaled with `r` so the LoRA scaling factor
    alpha/r stays at 2.0 everywhere — otherwise a rank change silently becomes a
    learning-rate change too, and the ablation is confounded.

    PEFT matches `rank_pattern` keys as the regex `.*\\.{key}$` against each full
    module path (e.g. "model.encoder.layers.0.self_attn.q_proj"), which is why the
    keys below are written as regexes rather than plain substrings.
    """
    kwargs: Dict[str, Any] = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    if layer_adaptive:
        scaling = alpha / r  # preserve alpha/r == 2.0
        # Explicit alphas override the scaling rule. Needed to reproduce an existing
        # run exactly: setting encoder_alpha=encoder_r (and likewise for the decoder)
        # gives alpha/r == 1.0, i.e. adapters contributing at half the strength the
        # scaling rule would give them. Seeds in one multi-seed set must agree on this
        # or they are not replicates of the same configuration.
        enc_alpha = encoder_alpha if encoder_alpha is not None else int(round(encoder_r * scaling))
        dec_alpha = decoder_alpha if decoder_alpha is not None else int(round(decoder_r * scaling))
        kwargs["rank_pattern"] = {
            r"encoder\..*": encoder_r,
            r"decoder\..*": decoder_r,
        }
        kwargs["alpha_pattern"] = {
            r"encoder\..*": enc_alpha,
            r"decoder\..*": dec_alpha,
        }
    return kwargs


# --------------------------------------------------------------------------------------
# Dataset transform + collator
# --------------------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    """Decode an audio file to a mono float32 waveform at `target_sr`.

    Deliberately does NOT use `datasets.Audio`: its decoding backend changed across
    major versions (soundfile in datasets<4, torchcodec in >=4), which makes a notebook
    that worked last month fail today. soundfile handles wav/flac/ogg and, with
    libsndfile >= 1.1, mp3; librosa is the fallback for anything it refuses.
    """
    arr = None
    sr = None
    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        arr = None
    if arr is None:
        import librosa
        arr, sr = librosa.load(path, sr=None, mono=True)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:  # (frames, channels) -> mono
        arr = arr.mean(axis=1)

    if sr != target_sr:
        try:
            import soxr
            arr = soxr.resample(arr, sr, target_sr)
        except ImportError:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        arr = np.asarray(arr, dtype=np.float32)

    return arr


def make_transform(processor, augment: bool, seed: int, noise_prob: float):
    """Build an on-the-fly `set_transform` callable.

    Decoding, noise injection and log-Mel extraction all happen per batch access, so
    every epoch sees a fresh noise draw. `augment` must be False for val/test.
    """
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    rng = np.random.default_rng(seed)

    def _transform(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        paths = batch["audio_path"]
        sentences = batch["sentence"]

        waveforms = []
        for path in paths:
            arr = load_audio(path)
            if augment:
                arr = add_gaussian_noise(arr, rng, p=noise_prob)
            waveforms.append(arr)

        features = feature_extractor(
            waveforms,
            sampling_rate=SAMPLING_RATE,
            return_tensors="np",
        ).input_features

        labels = tokenizer(
            [normalize_text(s) for s in sentences]
        ).input_ids

        return {
            "input_features": [f.astype(np.float32) for f in features],
            "labels": labels,
        }

    return _transform


@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad log-Mel features and right-pad labels with -100 so they are ignored by the loss.

    Whisper pads/truncates audio to a fixed 30s window, so `input_features` are already
    uniform; `feature_extractor.pad` is used anyway to stay correct if that changes.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Whisper's decoder prepends the start token itself; drop it if the tokenizer
        # already added one, or the model sees it twice.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# --------------------------------------------------------------------------------------
# Trainer with generation-based WER evaluation
# --------------------------------------------------------------------------------------

def _build_trainer_class():
    """Defined lazily so the module imports without transformers present (for unit tests)."""
    import torch
    from transformers import Seq2SeqTrainer

    class WhisperWERTrainer(Seq2SeqTrainer):
        """Seq2SeqTrainer that scores WER by running real generation.

        `Seq2SeqTrainer(predict_with_generate=True)` is unreliable when the model is a
        PEFT adapter over a 4-bit backbone, so generation is done here explicitly under
        autocast. Best-checkpoint selection and early stopping are handled in this class
        rather than via `load_best_model_at_end` / `EarlyStoppingCallback`, which both
        have rough edges around adapter-only checkpoints.
        """

        def __init__(
            self,
            *args,
            processor=None,
            wer_metric=None,
            gen_max_new_tokens: int = 225,
            gen_num_beams: int = 1,
            best_adapter_dir: str = "best_adapter",
            early_stopping_patience: int = 5,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.processor = processor
            self.wer_metric = wer_metric
            self.gen_max_new_tokens = gen_max_new_tokens
            self.gen_num_beams = gen_num_beams
            self.best_adapter_dir = best_adapter_dir
            self.early_stopping_patience = early_stopping_patience
            self.best_wer = float("inf")
            self.evals_without_improvement = 0
            self.wer_history: List[Dict[str, float]] = []
            silence_generation_length_warning()

            # Carry the best WER across a resume. Without this, best_wer restarts at
            # infinity and the first evaluation of the resumed run overwrites the saved
            # best adapter even when it is worse than what the earlier run achieved.
            marker = os.path.join(self.best_adapter_dir, "best_metric.json")
            if os.path.exists(marker):
                try:
                    with open(marker) as fh:
                        previous = json.load(fh)
                    self.best_wer = float(previous["val_wer"])
                    logger.info("Carried over best val WER %.2f%% (step %s) from the "
                                "interrupted run", self.best_wer, previous.get("step", "?"))
                except Exception as exc:
                    logger.warning("Could not read %s (%s); best WER starts fresh",
                                   marker, type(exc).__name__)

        @torch.no_grad()
        def compute_wer(self, dataset, description: str = "eval") -> Dict[str, Any]:
            model = self.model
            was_training = model.training
            model.eval()

            dataloader = self.get_eval_dataloader(dataset)
            device = self.args.device
            use_cuda = device.type == "cuda"

            predictions: List[str] = []
            references: List[str] = []
            tokenizer = self.processor.tokenizer

            for batch in dataloader:
                input_features = batch["input_features"].to(device)
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16,
                                    enabled=use_cuda):
                    generated = model.generate(
                        input_features=input_features,
                        max_new_tokens=self.gen_max_new_tokens,
                        num_beams=self.gen_num_beams,
                        use_cache=True,
                    )
                generated = generated.detach().cpu().numpy()

                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
                references.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

                del generated, input_features
                if use_cuda:
                    torch.cuda.empty_cache()

            predictions = [normalize_text(p) for p in predictions]
            references = [normalize_text(r) for r in references]

            # An empty reference makes WER undefined; drop those pairs and say how many.
            kept = [(p, r) for p, r in zip(predictions, references) if r]
            dropped = len(references) - len(kept)
            if dropped:
                logger.warning("%s: dropped %d sample(s) with empty reference", description, dropped)
            if not kept:
                raise ValueError(f"{description}: no scorable samples (all references empty)")

            preds, refs = zip(*kept)
            wer = 100.0 * self.wer_metric.compute(predictions=list(preds), references=list(refs))

            if was_training:
                model.train()

            return {"wer": wer, "n_scored": len(kept), "n_dropped": dropped,
                    "predictions": list(preds), "references": list(refs)}

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            result = self.compute_wer(dataset, description=metric_key_prefix)
            wer = result["wer"]

            metrics = {
                f"{metric_key_prefix}_wer": wer,
                f"{metric_key_prefix}_n_scored": result["n_scored"],
            }
            self.log(metrics)
            self.wer_history.append({"step": int(self.state.global_step), "wer": wer})

            if wer < self.best_wer:
                self.best_wer = wer
                self.evals_without_improvement = 0
                os.makedirs(self.best_adapter_dir, exist_ok=True)
                # Save the adapter only — the 4-bit backbone is reloaded from the hub.
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.save_pretrained(self.best_adapter_dir)
                self.processor.save_pretrained(self.best_adapter_dir)
                with open(os.path.join(self.best_adapter_dir, "best_metric.json"), "w") as fh:
                    json.dump({"step": int(self.state.global_step), "val_wer": wer}, fh, indent=2)
                logger.info("New best val WER %.2f%% at step %d -> saved to %s",
                            wer, self.state.global_step, self.best_adapter_dir)
            else:
                self.evals_without_improvement += 1
                logger.info("No improvement (%.2f%% vs best %.2f%%), patience %d/%d",
                            wer, self.best_wer,
                            self.evals_without_improvement, self.early_stopping_patience)
                if self.evals_without_improvement >= self.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    self.control.should_training_stop = True

            return metrics

    return WhisperWERTrainer


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_prebuilt_splits(split_csv: str, audio_dir: str):
    """Load a split prepared elsewhere, from a CSV carrying a `split` column.

    Used for the speaker-familiarity study, where the partition is built by speaker
    (see make_speaker_split.py) rather than by random clip assignment. Every split name
    present in the file becomes a dataset, so `seen_test` and `unseen_test` come through
    untouched for the evaluation script to score later.
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict

    df = pd.read_csv(split_csv)
    for col in ("path", "transcript", "split"):
        if col not in df.columns:
            raise SystemExit(f"{split_csv} has no {col!r} column")

    df["path"] = df["path"].apply(lambda p: os.path.join(audio_dir, os.path.basename(str(p))))
    df["transcript"] = df["transcript"].apply(normalize_text)
    df = df[df["transcript"].notnull() & (df["transcript"] != "")]

    exists = df["path"].apply(os.path.exists)
    if (~exists).any():
        logger.warning("Dropping %d row(s) whose audio file is missing", int((~exists).sum()))
        df = df[exists]

    out = {}
    for name, part in df.groupby("split"):
        out[str(name)] = Dataset.from_dict({"audio_path": part["path"].tolist(),
                                            "sentence": part["transcript"].tolist()})
    if "train" not in out:
        raise SystemExit(f"{split_csv} has no rows with split == 'train'")
    if "val" not in out:
        raise SystemExit(f"{split_csv} has no rows with split == 'val' (needed for "
                         f"model selection)")

    logger.info("Pre-built splits: %s",
                ", ".join(f"{k}={len(v)}" for k, v in sorted(out.items())))
    return DatasetDict(out)


def load_splits(csv_path: str, audio_dir: str, n_samples: int, split_seed: int):

    import pandas as pd
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path).head(n_samples)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    df["path"] = df["path"].apply(
        lambda p: os.path.join(audio_dir, os.path.basename(str(p)))
    )
    df["transcript"] = df["transcript"].apply(normalize_text)
    df = df[df["transcript"].notnull() & (df["transcript"] != "")]

    exists = df["path"].apply(os.path.exists)
    if (~exists).any():
        logger.warning("Dropping %d row(s) whose audio file is missing", int((~exists).sum()))
        df = df[exists]

    if len(df) == 0:
        raise ValueError("No usable rows after cleaning — check --csv and --audio-dir.")

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=split_seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=split_seed, shuffle=True)

    def to_ds(frame):
        # Paths stay plain strings; decoding happens in the transform (see load_audio).
        return Dataset.from_dict({
            "audio_path": frame["path"].tolist(),
            "sentence": frame["transcript"].tolist(),
        })

    splits = DatasetDict({"train": to_ds(train_df), "val": to_ds(val_df), "test": to_ds(test_df)})
    logger.info("Split sizes — train %d, val %d, test %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))
    return splits


# --------------------------------------------------------------------------------------
# Environment preflight
# --------------------------------------------------------------------------------------

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the highest-numbered `checkpoint-N` directory in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    found = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        path = os.path.join(output_dir, name)
        if match and os.path.isdir(path):
            found.append((int(match.group(1)), path))
    if not found:
        return None
    return max(found)[1]


def check_output_dir_persistence(output_dir: str) -> None:
    """Warn loudly if checkpoints are going to the ephemeral Colab VM disk.

    A Colab VM that disconnects or is reclaimed takes /content with it. A multi-hour
    run whose --output-dir is not under /content/drive has nothing to resume from.
    """
    abs_dir = os.path.abspath(output_dir)
    on_colab = os.path.isdir("/content")
    persistent = abs_dir.startswith(("/content/drive", "/gdrive", "/content/gdrive"))
    if on_colab and not persistent:
        logger.warning(
            "\n" + "!" * 72 +
            f"\n  --output-dir is {abs_dir}"
            "\n  That is the EPHEMERAL VM disk. If this runtime disconnects or is"
            "\n  reclaimed, every checkpoint is lost and there is nothing to resume."
            "\n  For a multi-hour run, mount Drive and point --output-dir at it:"
            "\n      from google.colab import drive; drive.mount('/content/drive')"
            "\n      --output-dir /content/drive/MyDrive/swahili-asr/qlora-large-v3"
            "\n" + "!" * 72
        )
    elif persistent:
        logger.info("Checkpoints        : %s (persistent)", abs_dir)


def preflight(use_4bit: bool, base_model: str) -> None:
    """Fail fast, with a fix, before spending minutes on dataset loading.

    Checks the things that actually break on a fresh Colab runtime: no GPU, a
    bitsandbytes too old for the installed Triton, and a large base model asked for
    without quantization on a small card.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible.\n"
            "  Colab: Runtime > Change runtime type > T4 GPU, then re-run.\n"
            "  (Pass --no-4bit only if you genuinely intend to train on CPU/fp32.)"
        )

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    major, _ = torch.cuda.get_device_capability(0)
    logger.info("GPU                : %s (%.1f GB, sm_%d.x)", name, total_gb, major)
    if major < 8:
        logger.info("                     Turing or older — fp16 compute, no bf16. Handled.")

    if not use_4bit:
        if "large" in base_model and total_gb < 24:
            logger.warning(
                "--no-4bit with %s on a %.0f GB card will almost certainly OOM. "
                "Drop --no-4bit to use NF4.", base_model, total_gb
            )
        return

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        detail = str(exc)
        if "triton.ops" in detail:
            raise SystemExit(
                "bitsandbytes is too old for the installed Triton.\n"
                "  `triton.ops` was removed in Triton 3.x, but bitsandbytes < 0.44 still\n"
                "  imports it, so the import fails before any of your code runs.\n\n"
                "  Fix (both steps — the second is not optional):\n"
                "      !pip install -q -U bitsandbytes\n"
                "      then Runtime > Restart session\n\n"
                "  The restart matters: the failed import is cached in sys.modules and in\n"
                "  transformers' lazy-module table, so a reinstall alone changes nothing."
            ) from exc
        raise SystemExit(
            f"bitsandbytes could not be imported: {type(exc).__name__}: {detail}\n"
            "  Try:  !pip install -q -U bitsandbytes   then restart the session."
        ) from exc

    logger.info("bitsandbytes       : %s", getattr(bitsandbytes, "__version__", "unknown"))


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def load_quantized_model(base_model: str, use_4bit: bool, gradient_checkpointing: bool):
    import torch
    from transformers import WhisperForConditionalGeneration

    if use_4bit:
        from transformers import BitsAndBytesConfig

        # fp16 compute, not bf16: the Tesla T4 is Turing (sm_75) and has no bf16 support.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model, quantization_config=quant_config, device_map="auto"
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required so gradients reach the adapters through a frozen, checkpointed backbone.
        model.enable_input_require_grads()

    return model


def configure_generation(model, processor, language: str = "sw", task: str = "transcribe"):
    """Pin the decoder prompt to Swahili transcription and clear legacy forced ids."""
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    # Deliberately NOT clearing `suppress_tokens`. Older Whisper fine-tuning notebooks
    # set `config.suppress_tokens = []`; that both trips a deprecation path in current
    # transformers and removes Whisper's non-speech token suppression, which costs WER.
    return model


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def _resolve_argv(argv: Optional[List[str]]) -> List[str]:
    """Return the argument list to parse, tolerating a Jupyter/Colab kernel.

    A notebook kernel is launched as `... -f /root/.../kernel-<uuid>.json`, so a bare
    `main()` called from a cell would otherwise see that flag in `sys.argv` and argparse
    would exit with "unrecognized arguments". Inside a kernel there are no real
    command-line arguments anyway, so pass them explicitly:

        from swahili_asr_qlora import main
        main(["--base-model", "openai/whisper-large-v3", "--epochs", "5"])
    """
    if argv is not None:
        return list(argv)

    raw = list(sys.argv[1:])
    in_kernel = "ipykernel" in sys.modules or "google.colab" in sys.modules
    if not in_kernel:
        return raw

    cleaned: List[str] = []
    skip_next = False
    for token in raw:
        if skip_next:
            skip_next = False
            continue
        if token == "-f":
            skip_next = True          # drop the kernel connection-file path that follows
            continue
        if token.startswith("--f=") or token.startswith("-f="):
            continue
        if token.endswith(".json") and "kernel" in os.path.basename(token):
            continue
        cleaned.append(token)

    if cleaned:
        logger.warning("Running inside a notebook kernel; parsing %r from sys.argv. "
                       "Prefer passing arguments explicitly: main([...])", cleaned)
    return cleaned


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Whisper for Swahili ASR (Gaussian-noise augmentation only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--csv", default="/content/drive/MyDrive/Colab/New Swahili Dataset/"
                                    "mozilla-swa-dataset/swahili_dataset.csv",
                   help="CSV with 'path' and 'transcript' columns.")
    p.add_argument("--audio-dir", default="/content/drive/MyDrive/Colab/New Swahili Dataset/"
                                          "mozilla-swa-dataset/clips",
                   help="Directory holding the audio clips.")
    p.add_argument("--n-samples", type=int, default=9000,
                   help="Clips to take from the head of the CSV (9000 = subset).")
    p.add_argument("--split-seed", type=int, default=THESIS_SPLIT_SEED,
                   help="Keep at 42 to reproduce the split exactly.")
    p.add_argument("--split-csv", default=None,
                   help="A CSV with path/transcript/split columns (see "
                        "make_speaker_split.py). Overrides --csv and the internal "
                        "random partition; use it for the speaker-familiarity study.")

    # Model / PEFT
    p.add_argument("--base-model", default="openai/whisper-large-v3",
                   help="HF model id. Use openai/whisper-small for the like-for-like arm.")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable NF4 quantization (only if you have an A100/L4 with headroom).")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--layer-adaptive", action="store_true",
                   help="Layer-adaptive ranks (Han et al. 2024): high encoder rank, low decoder rank.")
    p.add_argument("--encoder-r", type=int, default=64)
    p.add_argument("--decoder-r", type=int, default=16)
    p.add_argument("--encoder-alpha", type=int, default=None,
                   help="Explicit encoder LoRA alpha, overriding the alpha/r=2.0 rule. "
                        "Pass the same value as --encoder-r to get alpha/r=1.0.")
    p.add_argument("--decoder-alpha", type=int, default=None,
                   help="Explicit decoder LoRA alpha. See --encoder-alpha.")

    # Augmentation
    p.add_argument("--noise-prob", type=float, default=0.3,
                   help="Probability of applying Gaussian noise to a training clip.")
    p.add_argument("--no-augment", action="store_true",
                   help="Train without augmentation (control arm).")

    # Optimisation
    p.add_argument("--epochs", type=float, default=5.0,
                   help="25 matches for whisper-small; 4-6 is ample for large-v3.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16,
                   help="batch-size * grad-accum = 32")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", default="linear",
                   choices=["linear", "cosine", "constant_with_warmup"])
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42,
                   help="Training seed. Vary this across >=3 runs before ranking anything.")

    # Evaluation
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-steps", type=int, default=100,
                   help="Checkpoint interval. Kept well below --eval-steps on purpose: a "
                        "checkpoint is cheap, and it caps how much wall-clock a crash costs. "
                        "Evaluation runs generation and is the expensive one.")
    p.add_argument("--eval-subset", type=int, default=300,
                   help="Val clips used for mid-training WER (generation is slow). "
                        "0 = use the full validation split.")
    p.add_argument("--num-beams", type=int, default=1,
                   help="1 = greedy.")
    p.add_argument("--max-new-tokens", type=int, default=225)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--fleurs-csv", default=None,
                   help="Optional CSV for out-of-corpus FLEURS evaluation.")
    p.add_argument("--fleurs-audio-dir", default=None)

    # Bookkeeping
    p.add_argument("--output-dir", default="./whisper-qlora-gaussian",
                   help="Put this on Google Drive for any run over ~1 hour, or a "
                        "disconnect loses every checkpoint.")
    p.add_argument("--resume", default="auto",
                   help="'auto' picks up the newest checkpoint in --output-dir if one "
                        "exists and starts fresh otherwise; 'none' always starts from "
                        "scratch; or pass an explicit checkpoint-N path.")
    p.add_argument("--ignore-data-skip", action="store_true",
                   help="On resume, do not fast-forward the dataloader to the saved step. "
                        "Saves replaying (and re-decoding) every batch already seen, at the "
                        "cost of a different sample order for the remainder of that epoch.")
    p.add_argument("--skip-test", action="store_true",
                   help="Skip the held-out test evaluation (for quick sanity runs).")

    return p.parse_args(_resolve_argv(argv))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Seq2SeqTrainingArguments, WhisperProcessor, set_seed

    use_4bit = not args.no_4bit
    preflight(use_4bit, args.base_model)
    check_output_dir_persistence(args.output_dir)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Base model         : %s", args.base_model)
    logger.info("Quantization       : %s", "4-bit NF4 (double quant, fp16 compute)" if use_4bit else "none")
    logger.info("Augmentation       : %s", "none" if args.no_augment
                else f"additive Gaussian noise, p={args.noise_prob}, sigma~U(0.001,0.010)")
    logger.info("LoRA               : %s", "layer-adaptive" if args.layer_adaptive
                else f"uniform r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info("Seed               : %d", args.seed)

    # ---- data -------------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="Swahili", task="transcribe"
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    if args.split_csv:
        splits = load_prebuilt_splits(args.split_csv, args.audio_dir)
    else:
        splits = load_splits(args.csv, args.audio_dir, args.n_samples, args.split_seed)

    splits["train"].set_transform(
        make_transform(processor, augment=not args.no_augment,
                       seed=args.seed, noise_prob=args.noise_prob)
    )
    eval_transform = make_transform(processor, augment=False, seed=args.seed, noise_prob=0.0)
    # Every split but train is scored, never augmented. Named generically because a
    # pre-built split CSV supplies seen_test/unseen_test instead of a single "test".
    for _name in splits:
        if _name != "train":
            splits[_name].set_transform(eval_transform)

    val_for_training = splits["val"]
    if args.eval_subset and args.eval_subset < len(splits["val"]):
        val_for_training = splits["val"].select(range(args.eval_subset))
        val_for_training.set_transform(eval_transform)
        logger.info("Mid-training WER measured on a %d-clip validation subset; "
                    "the full %d-clip split is scored once at the end.",
                    args.eval_subset, len(splits["val"]))

    # ---- model ------------------------------------------------------------------------
    gradient_checkpointing = not args.no_gradient_checkpointing
    model = load_quantized_model(args.base_model, use_4bit, gradient_checkpointing)
    model = configure_generation(model, processor)

    lora_kwargs = build_lora_kwargs(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        layer_adaptive=args.layer_adaptive,
        encoder_r=args.encoder_r, decoder_r=args.decoder_r,
        encoder_alpha=args.encoder_alpha, decoder_alpha=args.decoder_alpha,
    )
    model = get_peft_model(model, LoraConfig(task_type=None, **lora_kwargs))
    model.print_trainable_parameters()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ---- training ---------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        ignore_data_skip=args.ignore_data_skip,
        logging_steps=25,
        predict_with_generate=False,   # generation is handled by WhisperWERTrainer
        load_best_model_at_end=False,  # best adapter is saved explicitly instead
        remove_unused_columns=False,   # set_transform supplies the columns
        label_names=["labels"],
        dataloader_num_workers=2,
        seed=args.seed,
        report_to=[],
    )

    wer_metric = load_wer_metric()
    best_adapter_dir = os.path.join(args.output_dir, "best_adapter")
    WhisperWERTrainer = _build_trainer_class()

    trainer = WhisperWERTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=val_for_training,
        data_collator=collator,
        processing_class=processor.feature_extractor,
        processor=processor,
        wer_metric=wer_metric,
        gen_max_new_tokens=args.max_new_tokens,
        gen_num_beams=args.num_beams,
        best_adapter_dir=best_adapter_dir,
        early_stopping_patience=args.early_stopping_patience,
    )

    resume_from: Optional[str] = None
    if args.resume in ("none", "None", ""):
        logger.info("Resume             : disabled (--resume none) — starting from scratch")
    elif args.resume == "auto":
        resume_from = find_latest_checkpoint(args.output_dir)
        if resume_from:
            logger.info("Resume             : %s  (pass --resume none to start over)",
                        resume_from)
        else:
            logger.info("Resume             : no checkpoint found in %s — starting fresh",
                        args.output_dir)
    else:
        resume_from = args.resume
        if not os.path.isdir(resume_from):
            raise SystemExit(
                f"--resume {resume_from} is not a directory.\n"
                f"  Checkpoints found in {args.output_dir}: "
                f"{find_latest_checkpoint(args.output_dir) or 'none'}\n"
                "  Use --resume auto to pick the newest automatically."
            )
        logger.info("Resume             : %s (explicit)", resume_from)

    started = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    train_minutes = (time.time() - started) / 60.0
    logger.info("Training finished in %.1f min. Best validation WER: %.2f%%",
                train_minutes, trainer.best_wer)

    # ---- final evaluation -------------------------------------------------------------
    results: Dict[str, Any] = {
        "base_model": args.base_model,
        "quantization": "nf4-4bit" if use_4bit else "none",
        "augmentation": "none" if args.no_augment else "gaussian_noise",
        "noise_prob": None if args.no_augment else args.noise_prob,
        "lora": lora_kwargs,
        "layer_adaptive": args.layer_adaptive,
        "seed": args.seed,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "num_beams": args.num_beams,
        "decoding": "greedy" if args.num_beams == 1 else f"beam-{args.num_beams}",
        "best_val_wer_subset": trainer.best_wer,
        "val_wer_history": trainer.wer_history,
        "train_minutes": round(train_minutes, 1),
    }

    if os.path.isdir(best_adapter_dir):
        logger.info("Reloading best adapter from %s for final scoring", best_adapter_dir)
        from peft import PeftModel

        # Free the training model and its optimizer state first. Holding both copies
        # plus Adam moments is what pushes a T4 over the edge on large-v3.
        trainer.model = None
        trainer.optimizer = None
        trainer.lr_scheduler = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base = load_quantized_model(args.base_model, use_4bit, gradient_checkpointing=False)
        base = configure_generation(base, processor)
        best_model = PeftModel.from_pretrained(base, best_adapter_dir)
        best_model.eval()
        trainer.model = best_model

    full_val = trainer.compute_wer(splits["val"], description="val_full")
    results["val_wer_full"] = full_val["wer"]
    logger.info("Full validation WER: %.2f%% (n=%d)", full_val["wer"], full_val["n_scored"])

    if not args.skip_test and "test" in splits:
        test = trainer.compute_wer(splits["test"], description="test")
        results["test_wer"] = test["wer"]
        results["test_n"] = test["n_scored"]
        logger.info("HELD-OUT TEST WER: %.2f%%  (n=%d)", test["wer"], test["n_scored"])

        preds_path = os.path.join(args.output_dir, f"test_transcriptions_seed{args.seed}.csv")
        import pandas as pd
        pd.DataFrame({"reference": test["references"], "prediction": test["predictions"]}) \
            .to_csv(preds_path, index=False)
        logger.info("Test transcriptions written to %s", preds_path)

    if args.fleurs_csv and args.fleurs_audio_dir:
        logger.info("Running out-of-corpus FLEURS evaluation")
        fleurs = load_splits(args.fleurs_csv, args.fleurs_audio_dir,
                             n_samples=10 ** 9, split_seed=args.split_seed)
        # load_splits partitions; for FLEURS we want every clip, so re-merge.
        from datasets import concatenate_datasets
        fleurs_all = concatenate_datasets([fleurs["train"], fleurs["val"], fleurs["test"]])
        fleurs_all.set_transform(eval_transform)
        fl = trainer.compute_wer(fleurs_all, description="fleurs")
        results["fleurs_wer"] = fl["wer"]
        results["fleurs_n"] = fl["n_scored"]
        logger.info("FLEURS (out-of-corpus) WER: %.2f%%  (n=%d)", fl["wer"], fl["n_scored"])

    results_path = os.path.join(args.output_dir, f"results_seed{args.seed}.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results written to %s", results_path)

    print("\n" + "=" * 68)
    print(f"  base model     : {args.base_model}")
    print(f"  quantization   : {results['quantization']}")
    print(f"  augmentation   : {results['augmentation']}")
    print(f"  LoRA           : {'layer-adaptive' if args.layer_adaptive else f'uniform r={args.lora_r}'}")
    print(f"  seed           : {args.seed}   decoding: {results['decoding']}")
    print(f"  validation WER : {results['val_wer_full']:.2f}%")
    if "test_wer" in results:
        print(f"  TEST WER       : {results['test_wer']:.2f}%   <-- report this one")
    print("=" * 68)
    print("  Repeat with --seed 1 --seed 2 ... and report mean +/- SD. ")
    print("  showed single-run differences of ~1 point are inside seed variance.")
    print("=" * 68 + "\n")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
