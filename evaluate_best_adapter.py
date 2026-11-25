#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score a saved QLoRA adapter on the held-out splits — self-contained.

This script loads the saved adapter explicitly and defaults to greedy.

USAGE
-----
    !python evaluate_best_adapter.py \
        --adapter-dir /content/drive/MyDrive/whisper-qlora-gaussian/best_adapter \
        --splits val test

    
    !python evaluate_best_adapter.py --splits val test --also-beam5

    # out-of-corpus generalisation on FLEURS (unseen speakers):
    !python evaluate_best_adapter.py --splits fleurs \
        --fleurs-csv  /content/drive/MyDrive/.../fleurs_test.tsv \
        --fleurs-audio-dir /content/drive/MyDrive/.../fleurs_audio

The in-corpus split is rebuilt from the same CSV with the same seed, so the test set
is byte-identical to the one held out during training. FLEURS is never partitioned —
it is already external, so every usable row is scored.

FLEURS input formats handled:
  * the raw headerless release TSV (id, file_name, raw_transcription, transcription,
    num_samples, gender) — detected and named automatically
  * any CSV with a header, where the transcript and filename columns are found by
    name; override with --fleurs-text-column / --fleurs-path-column
Clips are looked up by basename, and if they are not at the top level of
--fleurs-audio-dir the directory is indexed recursively (FLEURS usually unpacks into
train/dev/test subfolders).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("eval_adapter")

SAMPLING_RATE = 16_000
THESIS_SPLIT_SEED = 42

_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Identical normalisation to training, applied to references and hypotheses."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text).lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def load_audio(path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    arr, sr = None, None
    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        arr = None
    if arr is None:
        try:
            import librosa
            arr, sr = librosa.load(path, sr=None, mono=True)
        except Exception:
            arr = None
    if arr is None or len(arr) == 0:
        logger.error("UNREADABLE AUDIO, substituting silence: %s", path)
        return np.zeros(target_sr, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
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


def make_eval_transform(processor):
    """No augmentation — evaluation must be deterministic."""
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def _transform(batch):
        waveforms = [load_audio(p) for p in batch["audio_path"]]
        features = feature_extractor(
            waveforms, sampling_rate=SAMPLING_RATE, return_tensors="np"
        ).input_features
        labels = tokenizer([normalize_text(s) for s in batch["sentence"]]).input_ids
        return {"input_features": [f.astype(np.float32) for f in features],
                "labels": labels}

    return _transform


# --------------------------------------------------------------------------------------
# External (out-of-corpus) sets — FLEURS
# --------------------------------------------------------------------------------------

# The raw FLEURS release ships headerless TSVs in this column order.
FLEURS_RAW_COLUMNS = ["id", "file_name", "raw_transcription", "transcription",
                      "num_samples", "gender"]
# `transcription` is FLEURS's already-normalised field; `raw_transcription` keeps
# casing and punctuation. normalize_text() collapses both to the same form, so either
# works — preferring the normalised one just avoids surprises.
TEXT_CANDIDATES = ["transcription", "raw_transcription", "transcript", "sentence", "text"]
PATH_CANDIDATES = ["file_name", "path", "filename", "audio_path", "audio", "file"]


def _read_delimited(path: str, sep: Optional[str], **kwargs):
    """Read a delimited file, surviving stray quote characters in the text.

    Transcripts routinely contain a lone `"` (a quoted phrase, an inch mark, an
    unbalanced apostrophe substitute). The csv module then reports
    `'\\t' expected after '"'` and the read dies. In a tab-separated file the delimiter
    cannot occur inside a field, so quoting carries no information at all and reading
    `"` as ordinary text is both safe and correct. Only applied as a fallback, so a
    genuine comma-separated file with properly quoted fields is still parsed normally.
    """
    import csv as _csv
    import pandas as pd

    try:
        return pd.read_csv(path, sep=sep, engine="python", **kwargs)
    except pd.errors.ParserError as exc:
        logger.warning("Quote characters confused the parser (%s); re-reading %s with "
                       "quoting disabled", exc, os.path.basename(path))
        return pd.read_csv(path, sep=sep, engine="python",
                           quoting=_csv.QUOTE_NONE, **kwargs)


def _read_table(path: str, explicit: Optional[List[Optional[str]]] = None):
    """Read a CSV or TSV, coping with the headerless raw FLEURS format.

    `explicit` carries any column names the user supplied. If one of them is present in
    the header, that settles it — otherwise a file with unusual column names could
    never be loaded, even though naming the columns is exactly what those flags are for.
    """
    import pandas as pd

    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else None
    df = _read_delimited(path, sep)

    header_columns = list(df.columns)
    lowered = [str(c).strip().lower() for c in header_columns]
    named = [c for c in (explicit or []) if c]
    recognised = (any(c in lowered for c in TEXT_CANDIDATES + PATH_CANDIDATES)
                  or any(c in header_columns for c in named))
    if not recognised:
        # Possibly the headerless raw FLEURS TSV — but only assume that if the data
        # actually looks like it. Guessing wrongly would silently consume a real header
        # row as data and mislabel every column, which is worse than failing.
        headerless = _read_delimited(path, sep, header=None)
        audio_like = any(
            headerless[col].astype(str).str.lower()
            .str.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")).any()
            for col in headerless.columns
        )
        if headerless.shape[1] >= 4 and audio_like:
            width = min(headerless.shape[1], len(FLEURS_RAW_COLUMNS))
            df = headerless.iloc[:, :width]
            df.columns = FLEURS_RAW_COLUMNS[:width]
            logger.info("No header row detected, and the rows carry audio filenames — "
                        "assuming raw FLEURS column order")
        else:
            raise SystemExit(
                f"{path}: cannot identify the columns.\n"
                f"  Header row read as: {header_columns}\n"
                f"  None of those name a transcript ({TEXT_CANDIDATES}) or a filename "
                f"({PATH_CANDIDATES}),\n"
                f"  and the file does not match the raw FLEURS layout "
                f"({len(headerless.columns)} columns, "
                f"{'no ' if not audio_like else ''}audio filenames found).\n"
                f"  Name them explicitly with --fleurs-text-column and "
                f"--fleurs-path-column."
            )

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick_column(df, explicit: Optional[str], candidates: List[str], what: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"Column {explicit!r} not in {list(df.columns)}")
        return explicit
    lowered = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise SystemExit(
        f"Could not identify the {what} column. Columns present: {list(df.columns)}\n"
        f"  Pass --fleurs-{what}-column to name it explicitly."
    )


def _resolve_audio_paths(names, audio_dir: str) -> List[str]:
    """Join basenames to audio_dir; fall back to a recursive index.

    FLEURS is commonly unpacked into train/dev/test subfolders, so a flat join misses.
    """
    direct = [os.path.join(audio_dir, os.path.basename(str(n))) for n in names]
    missing = [p for p in direct if not os.path.exists(p)]
    if not missing:
        return direct

    logger.info("%d/%d clips not at the top level — indexing %s recursively",
                len(missing), len(direct), audio_dir)
    index: Dict[str, str] = {}
    for root, _dirs, files in os.walk(audio_dir):
        for fname in files:
            index.setdefault(fname, os.path.join(root, fname))

    resolved, unresolved = [], 0
    for name, path in zip(names, direct):
        if os.path.exists(path):
            resolved.append(path)
            continue
        hit = index.get(os.path.basename(str(name)))
        if hit:
            resolved.append(hit)
        else:
            resolved.append(path)
            unresolved += 1
    if unresolved:
        logger.warning("%d clip(s) could not be located under %s and will be dropped",
                       unresolved, audio_dir)
    return resolved


def load_external_set(csv_path: str, audio_dir: str,
                      text_column: Optional[str] = None,
                      path_column: Optional[str] = None,
                      limit: Optional[int] = None):
    """Load an out-of-corpus evaluation set (e.g. the FLEURS Swahili test partition).

    Unlike `load_splits`, this does NOT partition: every usable row is scored, because
    an external benchmark is already a held-out set.
    """
    from datasets import Dataset

    df = _read_table(csv_path, [text_column, path_column])
    text_col = _pick_column(df, text_column, TEXT_CANDIDATES, "text")
    path_col = _pick_column(df, path_column, PATH_CANDIDATES, "path")
    logger.info("External set: %d row(s); text column %r, path column %r",
                len(df), text_col, path_col)

    df = df[[path_col, text_col]].copy()
    df.columns = ["name", "sentence"]
    df["sentence"] = df["sentence"].apply(normalize_text)
    df = df[df["sentence"].notnull() & (df["sentence"] != "")]

    df["audio_path"] = _resolve_audio_paths(df["name"].tolist(), audio_dir)
    before = len(df)
    df = df[df["audio_path"].apply(os.path.exists)]
    if len(df) < before:
        logger.warning("Dropped %d row(s) with no audio file", before - len(df))
    if len(df) == 0:
        raise SystemExit(
            f"No usable rows from {csv_path}.\n"
            f"  Columns used: text={text_col!r}, path={path_col!r}.\n"
            f"  Either --fleurs-audio-dir ({audio_dir}) does not hold the clips named "
            f"in {path_col!r},\n"
            f"  or those columns were identified wrongly — name them yourself with "
            f"--fleurs-text-column / --fleurs-path-column."
        )
    if limit:
        df = df.head(limit)

    logger.info("External set ready: %d clip(s)", len(df))
    return Dataset.from_dict({"audio_path": df["audio_path"].tolist(),
                              "sentence": df["sentence"].tolist()})


def load_prebuilt_splits(split_csv: str, audio_dir: str):
    """Load a partition prepared by make_speaker_split.py (path/transcript/split).

    Duplicated from the training script on purpose: this file stays runnable on its own,
    so scoring never depends on the training script being present and in sync.
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
    df = df[df["path"].apply(os.path.exists)]

    out = {str(name): Dataset.from_dict({"audio_path": part["path"].tolist(),
                                         "sentence": part["transcript"].tolist()})
           for name, part in df.groupby("split")}
    logger.info("Pre-built splits: %s",
                ", ".join(f"{k}={len(v)}" for k, v in sorted(out.items())))
    return DatasetDict(out)


def load_splits(csv_path: str, audio_dir: str, n_samples: int, split_seed: int):
    """Rebuild the exact training-time partition: same head(), same seed, same order."""
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path).head(n_samples)
    df["path"] = df["path"].apply(lambda p: os.path.join(audio_dir, os.path.basename(str(p))))
    df["transcript"] = df["transcript"].apply(normalize_text)
    df = df[df["transcript"].notnull() & (df["transcript"] != "")]
    df = df[df["path"].apply(os.path.exists)]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=split_seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=split_seed, shuffle=True)

    def to_ds(frame):
        return Dataset.from_dict({"audio_path": frame["path"].tolist(),
                                  "sentence": frame["transcript"].tolist()})

    splits = DatasetDict({"train": to_ds(train_df), "val": to_ds(val_df), "test": to_ds(test_df)})
    logger.info("Splits rebuilt — train %d, val %d, test %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))
    return splits


def collate(features, processor):
    import torch
    batch = processor.feature_extractor.pad(
        [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
    labels_batch = processor.tokenizer.pad(
        [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch


def wer_fn():
    try:
        import evaluate
        return evaluate.load("wer").compute
    except Exception:
        import jiwer
        return lambda predictions, references: float(jiwer.wer(list(references),
                                                               list(predictions)))


def silence_length_warning():
    class _Drop(logging.Filter):
        def filter(self, record):
            return "seem to have been set" not in record.getMessage()
    flt = _Drop()
    for name in ("transformers.generation.utils",
                 "transformers.generation.configuration_utils", "transformers"):
        logging.getLogger(name).addFilter(flt)
    for h in logging.getLogger("transformers").handlers:
        h.addFilter(flt)


def build_model(base_model: str, adapter_dir: str):
    import torch
    from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration
    from peft import PeftModel

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model, quantization_config=quant, device_map="auto")

    # Pin Swahili transcription via generation_config only. Setting
    # `forced_decoder_ids` as well is the legacy path: it is redundant here and is
    # what emits the "Passing `generation_config` together with generation-related
    # arguments is deprecated" warning on every single generate call.
    base.generation_config.language = "sw"
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    if hasattr(base.config, "forced_decoder_ids"):
        base.config.forced_decoder_ids = None

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def score(model, processor, dataset, batch_size: int, num_beams: int,
          max_new_tokens: int, label: str) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    device = next(p for p in model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda f: collate(f, processor))
    tokenizer = processor.tokenizer
    preds: List[str] = []
    refs: List[str] = []
    started = time.time()
    total = math.ceil(len(dataset) / batch_size)

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            feats = batch["input_features"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model.generate(input_features=feats,
                                     max_new_tokens=max_new_tokens,
                                     num_beams=num_beams, use_cache=True)
            out = out.detach().cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

            if i % 10 == 0 or i == total:
                done = time.time() - started
                eta = done / i * (total - i)
                logger.info("  %s  batch %d/%d   elapsed %.1f min   eta %.1f min",
                            label, i, total, done / 60, eta / 60)
            del out, feats
            if device.type == "cuda":
                torch.cuda.empty_cache()

    preds = [normalize_text(p) for p in preds]
    refs = [normalize_text(r) for r in refs]
    kept = [(p, r) for p, r in zip(preds, refs) if r]
    dropped = len(refs) - len(kept)
    if not kept:
        raise ValueError(f"{label}: nothing scorable (all references empty)")
    p_keep, r_keep = zip(*kept)
    wer = 100.0 * wer_fn()(predictions=list(p_keep), references=list(r_keep))
    return {"wer": wer, "n": len(kept), "dropped": dropped,
            "predictions": list(p_keep), "references": list(r_keep),
            "minutes": round((time.time() - started) / 60, 1)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--adapter-dir",
                   default="/content/drive/MyDrive/whisper-qlora-gaussian/best_adapter")
    p.add_argument("--base-model", default="openai/whisper-large-v3")
    p.add_argument("--csv", default="/content/drive/MyDrive/Colab/New Swahili Dataset/"
                                    "mozilla-swa-dataset/swahili_dataset.csv")
    p.add_argument("--audio-dir", default="/content/drive/MyDrive/Colab/New Swahili Dataset/"
                                          "mozilla-swa-dataset/clips")
    p.add_argument("--n-samples", type=int, default=9000)
    p.add_argument("--split-seed", type=int, default=THESIS_SPLIT_SEED)
    p.add_argument("--splits", nargs="+", default=["test"],
                   help="Split names to score. Without --split-csv: val/test/train, "
                        "plus 'fleurs'. With --split-csv: any split name in that file, "
                        "e.g. seen_test unseen_test. 'fleurs' always needs "
                        "--fleurs-csv and --fleurs-audio-dir.")
    p.add_argument("--split-csv", default=None,
                   help="CSV with path/transcript/split columns (make_speaker_split.py). "
                        "Scores the named splits from it instead of re-partitioning.")
    p.add_argument("--fleurs-csv", default=None,
                   help="CSV or TSV for the out-of-corpus set. The raw headerless "
                        "FLEURS TSV is detected automatically.")
    p.add_argument("--fleurs-audio-dir", default=None,
                   help="Directory holding the FLEURS clips; searched recursively if "
                        "they are not at the top level.")
    p.add_argument("--fleurs-text-column", default=None,
                   help="Override transcript column autodetection.")
    p.add_argument("--fleurs-path-column", default=None,
                   help="Override filename column autodetection.")
    p.add_argument("--fleurs-limit", type=int, default=None,
                   help="Score only the first N clips (for a quick smoke run).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Eval needs no gradients, so this can exceed the training batch size.")
    p.add_argument("--num-beams", type=int, default=1,
                   help="1 = greedy")
    p.add_argument("--also-beam5", action="store_true",
                   help="Additionally score every split with beam-5, reported separately.")
    p.add_argument("--max-new-tokens", type=int, default=225)
    p.add_argument("--out-dir", default=None,
                   help="Defaults to a results/ folder beside the adapter.")
    args = p.parse_args(argv if argv is not None else
                        ([] if "ipykernel" in sys.modules else sys.argv[1:]))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    silence_length_warning()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Runtime > Change runtime type > T4 GPU.")

    if not os.path.isdir(args.adapter_dir):
        raise SystemExit(f"No adapter at {args.adapter_dir}\n"
                         f"  Check the path with: !ls {os.path.dirname(args.adapter_dir)}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.adapter_dir.rstrip("/")),
                                           "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    marker = os.path.join(args.adapter_dir, "best_metric.json")
    trained_to = None
    if os.path.exists(marker):
        trained_to = json.load(open(marker))
        logger.info("Adapter is from step %s, val WER at save time %.2f%%",
                    trained_to.get("step"), trained_to.get("val_wer", float("nan")))

    from transformers import WhisperProcessor
    try:
        processor = WhisperProcessor.from_pretrained(args.adapter_dir,
                                                     language="Swahili", task="transcribe")
        logger.info("Processor loaded from the adapter directory")
    except Exception:
        processor = WhisperProcessor.from_pretrained(args.base_model,
                                                     language="Swahili", task="transcribe")
        logger.info("Processor loaded from the base model")
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    wants_fleurs = "fleurs" in args.splits
    in_corpus = [s for s in args.splits if s != "fleurs"]

    if wants_fleurs and not (args.fleurs_csv and args.fleurs_audio_dir):
        raise SystemExit("--splits fleurs needs both --fleurs-csv and --fleurs-audio-dir.")

    transform = make_eval_transform(processor)
    datasets: Dict[str, Any] = {}

    if in_corpus:
        if args.split_csv:
            splits = load_prebuilt_splits(args.split_csv, args.audio_dir)
        else:
            splits = load_splits(args.csv, args.audio_dir, args.n_samples, args.split_seed)
        for name in in_corpus:
            if name not in splits:
                raise SystemExit(
                    f"No split named {name!r}. Available: {sorted(splits)}")
            splits[name].set_transform(transform)
            datasets[name] = splits[name]

    if wants_fleurs:
        fleurs = load_external_set(args.fleurs_csv, args.fleurs_audio_dir,
                                   args.fleurs_text_column, args.fleurs_path_column,
                                   args.fleurs_limit)
        fleurs.set_transform(transform)
        datasets["fleurs"] = fleurs

    logger.info("Loading %s in 4-bit and applying the adapter...", args.base_model)
    model = build_model(args.base_model, args.adapter_dir)

    arms = [("greedy" if args.num_beams == 1 else f"beam{args.num_beams}", args.num_beams)]
    if args.also_beam5 and args.num_beams != 5:
        arms.append(("beam5", 5))

    results: Dict[str, Any] = {
        "adapter_dir": args.adapter_dir,
        "base_model": args.base_model,
        "adapter_trained_to": trained_to,
        "split_seed": args.split_seed,
        "n_samples": args.n_samples,
    }

    import pandas as pd
    for arm_name, beams in arms:
        for split_name in args.splits:
            label = f"{split_name}/{arm_name}"
            logger.info("Scoring %s (%d clips, num_beams=%d)",
                        label, len(datasets[split_name]), beams)
            r = score(model, processor, datasets[split_name], args.batch_size,
                      beams, args.max_new_tokens, label)
            results[f"{split_name}_wer_{arm_name}"] = r["wer"]
            results[f"{split_name}_n_{arm_name}"] = r["n"]
            logger.info("%s WER = %.2f%%  (n=%d, %.1f min)",
                        label, r["wer"], r["n"], r["minutes"])

            csv_path = os.path.join(out_dir, f"preds_{split_name}_{arm_name}.csv")
            pd.DataFrame({"reference": r["references"],
                          "prediction": r["predictions"]}).to_csv(csv_path, index=False)
            logger.info("  predictions -> %s", csv_path)

    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 66)
    for key in sorted(k for k in results if k.endswith(tuple(a[0] for a in arms))
                      and "_wer_" in k):
        print(f"  {key:<28} {results[key]:.2f}%")
    print("=" * 66)
    print(f"  results -> {json_path}")
    print(" ; beam-5 is a separate arm.")
    print("=" * 66 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
