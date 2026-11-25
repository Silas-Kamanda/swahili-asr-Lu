#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the QLoRA adapter into whisper-large-v3 and produce a standalone model.

WHY fp16 AND NOT 4-bit
----------------------
The adapter was trained over an NF4-quantized backbone, but it must NOT be merged
into one. Merging adds BA to the base weight; if that base is a 4-bit approximation,
the merged result inherits the quantization error permanently and `merge_and_unload()`
raises no warning. Load the base in fp16, merge there, save fp16.

This script verifies the merge actually changed the weights — the failure mode it
guards against is a merge that silently no-ops and ships you the stock model.

RUN IT IN COLAB (needs ~8GB RAM on CPU, or any GPU):

    !python merge_adapter.py \
        --adapter-dir /content/drive/MyDrive/whisper-qlora-gaussian/best_adapter \
        --out /content/merged-swahili \
        --push-to your-hf-username/whisper-large-v3-swahili-qlora


from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--adapter-dir", required=True,
                   help="The best_adapter directory from training.")
    p.add_argument("--base-model", default="openai/whisper-large-v3")
    p.add_argument("--out", default="./merged-swahili")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cpu is fine and avoids competing with anything on the GPU.")
    p.add_argument("--push-to", default=None,
                   help="HF repo id to upload the merged model to, e.g. user/name.")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv if argv is not None else
                        ([] if "ipykernel" in sys.modules else sys.argv[1:]))

    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not os.path.isdir(args.adapter_dir):
        raise SystemExit(f"No adapter at {args.adapter_dir}")

    print(f"Loading {args.base_model} in fp16 on {args.device} ...")
    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16).to(args.device)

    # Snapshot one adapted weight so we can prove the merge did something.
    probe_name = next(n for n, _ in base.named_parameters()
                      if n.endswith("encoder.layers.0.self_attn.q_proj.weight"))
    before = dict(base.named_parameters())[probe_name].detach().clone().float()

    print(f"Applying adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, args.adapter_dir, torch_dtype=torch.float16)

    print("Merging ...")
    model = model.merge_and_unload()

    after = dict(model.named_parameters())[probe_name].detach().float()
    delta = (after - before).abs().max().item()
    print(f"  max |Δ| on {probe_name}: {delta:.6g}")
    if delta == 0.0:
        raise SystemExit(
            "The merge changed nothing — the merged model is identical to stock "
            "whisper-large-v3.\n"
            "  Usual cause: the adapter's target_modules do not match this base model, "
            "or an empty adapter was saved."
        )

    # Pin Swahili transcription so callers get it by default and need no prompt args.
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="Swahili", task="transcribe")
    model.generation_config.language = "sw"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving to {args.out} ...")
    # `safe_serialization` was dropped once safetensors became the only format, so pass
    # it only where it is still accepted rather than pinning a transformers version.
    try:
        model.save_pretrained(args.out, safe_serialization=True)
    except TypeError:
        model.save_pretrained(args.out)
    processor.save_pretrained(args.out)

    size = sum(os.path.getsize(os.path.join(args.out, f))
               for f in os.listdir(args.out)) / 1024 ** 3
    print(f"  merged model is {size:.2f} GB")

    if args.push_to:
        print(f"Pushing to https://huggingface.co/{args.push_to} ...")
        # Upload the saved folder rather than calling model.push_to_hub(): it sends the
        # weights, config and processor files in one request, and it does not depend on
        # a transformers signature that has changed across versions.
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to, repo_type="model", private=args.private,
                        exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to,
                          repo_type="model")
        print(f"  done -> https://huggingface.co/{args.push_to}")

    print("\nSanity-check it before deploying:\n"
          "    from transformers import pipeline\n"
          f"    p = pipeline('automatic-speech-recognition', model='{args.out}')\n"
          "    print(p('some_swahili_clip.mp3')['text'])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
