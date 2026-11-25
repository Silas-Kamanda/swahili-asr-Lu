# Swahili ASR — parameter-efficient adaptation and speaker-aware evaluation

Code and evaluation protocol for the Swahili speech recognition work carried out under
a Laikipia University Internal Research Grant.

**Try it:** [Hugging Face Space](https://huggingface.co/spaces/SWataki/SWASR) ·
**Model:** [SWataki/whisper-large-v3-swahili-qlora](https://huggingface.co/SWataki/whisper-large-v3-swahili-qlora)

---



## The speaker-aware evaluation protocol

Common Voice splits are usually made at the clip level, so the same speaker appears in
both training and test. `make_speaker_split.py` builds a four-way partition that
separates speaker familiarity from everything else, holding the model fixed and varying
only the test set:

| split | contents |
|---|---|
| `train` | 75% of clips from training speakers |
| `val` | held-out clips, training speakers — model selection only |
| `seen_test` | held-out clips, **training** speakers — what a clip-level split measures |
| `unseen_test` | every clip from **held-out** speakers |

FLEURS is scored as a fourth, out-of-corpus condition.

Two consequences of the corpus are deliberate. `seen_test` inherits the corpus skew (the
largest speaker supplies ~13% of it) because that is exactly what a clip-level split
does. And `unseen_test` speakers are capped so no single voice can dominate, which means
they are drawn from lower-volume contributors — forced by a corpus where 10 of 111
speakers hold 61% of the clips.

## Results

QLoRA with layer-adaptive rank and Gaussian-noise augmentation, 13.5 hours of Mozilla
Common Voice Swahili, single Tesla T4. Greedy decoding, mean ± SD over three seeds.

| base model | `seen_test` | `unseen_test` | FLEURS |
|---|---|---|---|
| whisper-small (244M) | 12.95 ± 0.55 | 18.59 ± 2.71 | 30.11 ± 0.64 |
| whisper-medium (769M) | 9.08 ± 0.85 | 13.13 ± 2.36 | 25.04 ± 0.68 |
| whisper-large-v3 (1.55B) | **7.61 ± 0.61** | 10.72 ± 1.73 | 21.32 ± 0.09 |

Three findings:

**The proportional speaker penalty does not shrink with scale.** The `unseen_test` /
`seen_test` ratio is 1.43, 1.44, 1.40 across the three model sizes. A six-fold larger
model pays the same relative cost for an unfamiliar voice, even though the absolute gap
narrows (5.64 → 4.05 → 3.10 points).

**The cross-corpus penalty does grow.** FLEURS / `seen_test` moves 2.33 → 2.77 → 2.81.
At n=3 the confidence intervals overlap, so this is suggestive rather than established,
but it points the same way throughout: scaling transfers within a corpus better than
across one.

**Which speakers are held out matters more than model size.** Grouping the ratio by
split seed rather than by model, the rows are tight and the columns are not:

| split seed | small | medium | large |
|---|---|---|---|
| 42 | 1.222 | 1.270 | 1.259 |
| 1 | 1.507 | 1.556 | 1.448 |
| 2 | 1.572 | 1.491 | 1.499 |

With 28 held-out speakers drawn from a skewed corpus, the composition of that set swings
WER by about five points, and the effect persists across every model size.

### Known limitation

The grid uses one seed for both the split and training, so `unseen_test` variance mixes
training noise with split-composition noise. FLEURS, scored on the same 487 clips every
time, shows the training noise alone (SD 0.09–0.68). Separating the two cleanly needs a
run that fixes one split and varies training seeds across it.

### The deployed model

The checkpoint on the Space is an earlier single-seed run trained on a clip-level split
(8.1% held-out clips, 22.5% FLEURS). It is a different model from the grid above and its
numbers are not interchangeable with them — it can never have a legitimate held-out
speaker score, because its training set included the speakers that would be tested.

## Layout

```
swahili_asr_qlora.py      training: QLoRA + Gaussian noise, resumable
make_speaker_split.py     builds the four-way speaker-aware split
evaluate_best_adapter.py  scores a saved adapter on any split, plus FLEURS
grid_cell.py              the model x seed study, resumable across sessions
merge_adapter.py          merges the adapter into a standalone fp16 model
app.py                    the Gradio demo, as deployed on ZeroGPU
splits/                   split CSVs — needed to reproduce the comparison
```

## Reproducing

Data is not redistributed here. Both corpora are public:

- [Mozilla Common Voice 12.0, Swahili](https://commonvoice.mozilla.org/en/datasets) —
  the first 9,000 validated clips (~13.5 h) were used, which is deterministic.
- [FLEURS, Swahili](https://huggingface.co/datasets/google/fleurs) — the 487-clip test
  partition, used only for evaluation.

```bash
python make_speaker_split.py --csv swahili_with_speakers.csv --seed 42
python swahili_asr_qlora.py --split-csv splits/splits_seed42.csv --layer-adaptive \
    --base-model openai/whisper-large-v3 --audio-dir /path/to/clips
python evaluate_best_adapter.py --adapter-dir <out>/best_adapter \
    --split-csv splits/splits_seed42.csv --splits seen_test unseen_test
```

Training was done on a single Tesla T4 (16 GB) via Google Colab. Every script resumes
from its newest checkpoint, because sessions of this length do not survive intact.

## Licence and funding

Apache-2.0. Supported by a Laikipia University Internal Research Grant