import os, json, math, statistics, subprocess, sys

# ----------------------------- edit these -----------------------------
SEEDS  = [42, 1, 2]
MODELS = [   # cheapest first, so a full row of results lands early
    dict(tag="small",  hf="openai/whisper-small",    epochs="12", bs="8", ga="4"),
    dict(tag="medium", hf="openai/whisper-medium",   epochs="12", bs="4", ga="8"),
    dict(tag="large",  hf="openai/whisper-large-v3", epochs="6",  bs="2", ga="16"),
]
ROOT    = "/content/drive/MyDrive/Colab/SwahiliASR_study"
SPEAKER = "/content/drive/MyDrive/Colab/swahili_with_speakers.csv"
AUDIO   = "/content/drive/MyDrive/Colab/New Swahili Dataset/mozilla-swa-dataset/clips"
FL_CSV  = "/content/drive/MyDrive/Colab/Fleurs Swahili Dataset/test.tsv"
FL_DIR  = "/content/drive/MyDrive/Colab/Fleurs Swahili Dataset/test/test"
SPLIT, TRAIN, EVAL = ("/content/make_speaker_split.py",
                      "/content/swahili_asr_qlora.py",
                      "/content/evaluate_best_adapter.py")
CONDITIONS = ["seen_test", "unseen_test", "fleurs"]
# ----------------------------------------------------------------------

for _p in (SPLIT, TRAIN, EVAL):
    if not os.path.exists(_p):
        raise SystemExit(f"{_p} is missing. The Colab VM is wiped between sessions:\n"
                         f'  !cp "/content/drive/MyDrive/Colab/scripts"/*.py /content/')

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
def sh(c):
    # subprocess.call writes to the OS file descriptors, which Jupyter does NOT capture
    # into the cell — a child traceback would vanish and you would be debugging blind.
    import shlex
    cmd = " ".join(shlex.quote(x) for x in c)
    print("$", cmd, flush=True)
    try:
        ip = get_ipython()
    except NameError:
        ip = None
    if ip is not None:
        ip.system(cmd)
        return int(ip.user_ns.get("_exit_code", 0) or 0)
    p_ = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1)
    for line in p_.stdout:
        print(line.rstrip(), flush=True)
    return p_.wait()
os.makedirs(f"{ROOT}/splits", exist_ok=True)

# 1. splits (cheap, idempotent)
for s in SEEDS:
    f = f"{ROOT}/splits/splits_seed{s}.csv"
    if not os.path.exists(f):
        if sh([sys.executable, SPLIT, "--csv", SPEAKER, "--seed", str(s), "--out", f]):
            raise SystemExit(f"split for seed {s} failed — see the traceback above")

# 2. the grid
stop = False
for m in MODELS:
    if stop: break
    for s in SEEDS:
        tag  = f"{m['tag']}_seed{s}"
        out  = f"{ROOT}/{tag}"
        done = f"{out}/done.json"
        split = f"{ROOT}/splits/splits_seed{s}.csv"
        os.makedirs(out, exist_ok=True)
        if os.path.exists(done):
            print(f"{tag}: done, skipping"); continue
        print(f"\n===== {tag} =====", flush=True)

        # Training is complete only when the script has written its results file at the
        # very end of main(). An existing best_adapter/ proves nothing: a run killed
        # after its first checkpoint leaves one behind, and treating that as "trained"
        # silently evaluates a half-trained model. Always re-enter training; --resume
        # auto continues from the newest checkpoint, and returns almost immediately if
        # the run had in fact finished.
        if not os.path.exists(f"{out}/results_seed{s}.json"):
            rc = sh([sys.executable, TRAIN, "--base-model", m["hf"],
                     "--split-csv", split, "--audio-dir", AUDIO, "--output-dir", out,
                     "--seed", str(s), "--epochs", m["epochs"],
                     "--batch-size", m["bs"], "--grad-accum", m["ga"],
                     "--layer-adaptive", "--encoder-r", "64", "--decoder-r", "16",
                     "--eval-steps", "250", "--save-steps", "100",
                     "--eval-subset", "300", "--num-beams", "1",
                     "--resume", "auto", "--skip-test"])
            if rc:
                print(f"\n{tag} stopped. RE-RUN THIS CELL — it resumes."); stop = True; break
        if not os.path.exists(f"{out}/results_seed{s}.json"):
            print(f"\n{tag}: training exited 0 but wrote no results file; stopping "
                  f"rather than scoring a possibly half-trained adapter."); stop = True; break

        rc = sh([sys.executable, EVAL, "--adapter-dir", f"{out}/best_adapter",
                 "--base-model", m["hf"], "--split-csv", split, "--audio-dir", AUDIO,
                 "--splits", *CONDITIONS, "--fleurs-csv", FL_CSV,
                 "--fleurs-audio-dir", FL_DIR, "--batch-size", "4",
                 "--out-dir", f"{out}/eval_results"])
        if rc:
            print(f"\n{tag}: eval failed. RE-RUN THIS CELL."); stop = True; break
        json.dump({"model": m["hf"], "seed": s}, open(done, "w"))
        print(f"{tag}: complete")

# 3. aggregate
def stat(vals):
    n = len(vals); mu = statistics.mean(vals)
    if n < 2: return mu, 0.0, 0.0
    sd = statistics.stdev(vals)
    return mu, sd, T95.get(n - 1, 2.0) * sd / math.sqrt(n)

print("\n" + "=" * 78)
print("  WER by base model and speaker familiarity   (greedy, mean ± SD over seeds)")
print("=" * 78)
print(f"  {'model':<9}{'n':>3}{'seen_test':>16}{'unseen_test':>16}{'fleurs':>16}"
      f"{'unseen/seen':>13}")
print("-" * 78)
table = {}
for m in MODELS:
    got, cells = {}, []
    for cond in CONDITIONS:
        vals = []
        for s in SEEDS:
            f = f"{ROOT}/{m['tag']}_seed{s}/eval_results/eval_results.json"
            if os.path.exists(f):
                d = json.load(open(f))
                if f"{cond}_wer_greedy" in d: vals.append(d[f"{cond}_wer_greedy"])
        got[cond] = vals
        if not vals:                      cells.append("—")
        elif len(vals) < 2:               cells.append(f"{vals[0]:.2f} (n=1)")
        else: mu, sd, _ = stat(vals);     cells.append(f"{mu:.2f}±{sd:.2f}")
    n = len(got["seen_test"])
    ratio = (statistics.mean(got["unseen_test"]) / statistics.mean(got["seen_test"])
             if got["seen_test"] and got["unseen_test"] else None)
    print(f"  {m['tag']:<9}{n:>3}" + "".join(f"{c:>16}" for c in cells) +
          (f"{ratio:>12.2f}x" if ratio else f"{'—':>13}"))
    table[m["tag"]] = {c: got[c] for c in CONDITIONS}
print("=" * 78)
for m in MODELS:
    v = table[m["tag"]]["seen_test"]
    if len(v) >= 2:
        mu, sd, ci = stat(v)
        print(f"  {m['tag']:<9} seen_test  mean {mu:.2f}%  SD {sd:.2f}  "
              f"95% CI [{mu-ci:.2f}, {mu+ci:.2f}]")
json.dump(table, open(f"{ROOT}/study_summary.json", "w"), indent=2)
missing = [f"{m['tag']}_seed{s}" for m in MODELS for s in SEEDS
           if not os.path.exists(f"{ROOT}/{m['tag']}_seed{s}/done.json")]
if missing: print(f"\n  outstanding ({len(missing)}): {', '.join(missing[:6])}"
                  f"{' …' if len(missing) > 6 else ''}\n  re-run this cell")
print(f"  summary -> {ROOT}/study_summary.json")
