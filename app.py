"""
Swahili ASR demo — Hugging Face Space on ZeroGPU.

WHAT'S DIFFERENT FROM A NORMAL GRADIO APP
-----------------------------------------
ZeroGPU attaches a GPU only while a function decorated with `@spaces.GPU` is running,
then takes it away. So:

  * the model is constructed at module level with device="cuda" — ZeroGPU intercepts
    CUDA initialisation at startup and defers it, which is why this does not crash on
    a machine with no GPU attached yet;
  * every line that actually touches the GPU must sit inside the decorated function;
  * `duration` is the ceiling, not the charge. Returning early costs less, so a
    generous ceiling is safe and protects against a slow cold start being killed.

THE QUOTA IS THE DESIGN CONSTRAINT
----------------------------------
A free account gets about 5 GPU-minutes per day, shared across everyone who visits.
At roughly 8 seconds of GPU per clip that is ~35 transcriptions a day for the whole
world, so MAX_SECONDS is deliberately tight: halving clip length roughly doubles how
many people get to try it. Raise it only if you move to paid hardware.
"""

import gradio as gr
import torch


def _patch_gradio_client_bool_schema() -> None:
    """Stop /api_info dying on boolean JSON-schema nodes.

    gradio_client's get_type() does `if "const" in schema`, assuming every schema node
    is a dict. Pydantic emits a bare `true` for `additionalProperties`, so the route
    raises "argument of type 'bool' is not iterable" on every request — the UI then
    fails to initialise even though the model loaded fine. Known Gradio issue #11722.

    Patched here rather than by pinning a Gradio version, because the Space's gradio
    comes from sdk_version and we would be guessing at which release carries the fix.
    Harmless on versions that already handle it.
    """
    try:
        import gradio_client.utils as gcu
    except Exception:
        return

    if getattr(gcu, "_bool_schema_patched", False):
        return

    original_get_type = getattr(gcu, "get_type", None)
    if original_get_type is not None:
        def get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            return original_get_type(schema)
        gcu.get_type = get_type

    original_to_python = getattr(gcu, "_json_schema_to_python_type", None)
    if original_to_python is not None:
        def _json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            return original_to_python(schema, defs)
        gcu._json_schema_to_python_type = _json_schema_to_python_type

    gcu._bool_schema_patched = True


_patch_gradio_client_bool_schema()

MODEL_REPO = "SWataki/whisper-large-v3-swahili-qlora"
BASE_MODEL = "openai/whisper-large-v3"   # tokenizer/feature extractor come from here
MAX_SECONDS = 30          # also Whisper's native window, so no stitching is needed
GPU_DURATION = 30         # see below — this number affects QUEUE PRIORITY, not just cost

# Declaring a shorter duration raises your place in the ZeroGPU queue, so this is not
# simply a safety ceiling: asking for 90s when a 30s clip needs ~3s on an H200 pushed
# every visitor further back for no benefit. The default is 60s; 30s is comfortable
# headroom for MAX_SECONDS of audio plus first-call GPU setup, and it means anonymous
# visitors — who sit at the bottom of the queue anyway — wait less.

# `spaces` only exists on Space hardware. Rather than aliasing the decorator for local
# use, install a stand-in MODULE when it is absent — because ZeroGPU looks for the
# literal `@spaces.GPU` decorator when it scans the app at startup, and binding it to
# another name produces "No @spaces.GPU function detected during startup" even though
# the decorator is in fact applied. Shimming the module keeps that literal text intact.
try:
    import spaces
except ImportError:                                     # pragma: no cover — local only
    import sys as _sys
    import types as _types

    spaces = _types.ModuleType("spaces")

    def _GPU(*args, **kwargs):
        if args and callable(args[0]):                  # bare @spaces.GPU
            return args[0]
        return lambda fn: fn                            # @spaces.GPU(duration=...)

    spaces.GPU = _GPU
    _sys.modules["spaces"] = spaces

from transformers import WhisperProcessor, pipeline

# Take the tokenizer and feature extractor from the BASE model, not the merged repo.
# LoRA changes neither, so they are identical in content — but the copies in the merged
# repo were serialised by whatever transformers ran in Colab, and a Space running an
# older version chokes on them ("'list' object has no attribute 'keys'" from
# extra_special_tokens). Reading the canonical base files removes that version skew.
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL, language="Swahili", task="transcribe")

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_REPO,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device="cuda",
    chunk_length_s=30,
)


def probe_duration(path: str) -> float:
    """Length in seconds, from the header where possible rather than by decoding."""
    import soundfile as sf
    try:
        info = sf.info(path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    import librosa                                       # mp3/m4a soundfile won't open
    return float(librosa.get_duration(path=path))


@spaces.GPU(duration=GPU_DURATION)
def _run(path: str) -> str:
    out = asr(path, generate_kwargs={"language": "sw", "task": "transcribe"})
    return (out.get("text") or "").strip()


def transcribe(path):
    """Everything cheap happens here; the GPU is only entered once input is valid."""
    if not path:
        return "Choose an audio file, or record one, first."
    try:
        seconds = probe_duration(path)
    except Exception as exc:
        return f"Could not read that file ({type(exc).__name__}). Try WAV or MP3."
    if seconds > MAX_SECONDS:
        return (f"That clip is {seconds:.0f} seconds. This demo accepts up to "
                f"{MAX_SECONDS} — the free GPU quota is shared by everyone using the "
                f"page, so please trim it and try again.")
    try:
        text = _run(path)
    except Exception as exc:
        return f"Transcription failed: {type(exc).__name__}: {exc}"
    return text or "(no speech detected)"


with gr.Blocks(title="Swahili ASR") as demo:
    gr.Markdown(
        "# Swahili speech recognition\n"
        "Whisper-large-v3 adapted to Swahili with quantized LoRA (QLoRA) and additive "
        "Gaussian-noise augmentation, trained on roughly 13.5 hours of Mozilla Common "
        "Voice.\n\n"
        f"Upload or record up to **{MAX_SECONDS} seconds** of Swahili speech."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                            label="Swahili audio")
        # No show_copy_button: Gradio 6 dropped the argument (the copy control is
        # built into the component now). Passing it is a hard TypeError, not a warning.
        text_out = gr.Textbox(label="Transcription", lines=6)
    gr.Button("Transcribe", variant="primary").click(transcribe, audio_in, text_out)
    audio_in.stop_recording(transcribe, audio_in, text_out)

    # Say what it can't do here, rather than letting a visitor infer it from a bad result.
    gr.Markdown(
        "---\n"
        "### How well does it work?\n"
        "This model scores **8.1%** word error rate on held-out Common Voice clips — "
        "but those clips come from speakers it trained on. On **FLEURS**, a separate "
        "corpus of voices it has never heard, it scores **22.5%**. You are an "
        "unfamiliar speaker, so the second figure is the honest expectation.\n\n"
        "A companion three-seed study of the same method separates the two effects "
        "directly, holding the model fixed and varying only who is in the test set:\n\n"
        "| test set | WER |\n"
        "|---|---|\n"
        "| held-out *clips*, speakers seen in training | 7.6% ± 0.6 |\n"
        "| held-out *speakers*, same corpus | 10.7% ± 1.7 |\n"
        "| FLEURS, different corpus and speakers | 21.3% ± 0.1 |\n\n"
        "Same audio conditions, same model — the gap is speaker familiarity alone. "
        "It is why a single headline number from a standard split overstates what you "
        "will see. (Those three figures come from models trained on a speaker-disjoint "
        "split, so they characterise the method rather than this exact checkpoint.)\n\n"
        "Accuracy falls further with background noise, unfamiliar accents, proper "
        "nouns, and code-switching into English.\n\n"
        "Running on free shared ZeroGPU hardware with a few GPU-minutes per day, so it "
        "may queue or pause when busy.\n\n"
        "*Research prototype, Laikipia University. Audio is processed to produce the "
        "transcription and is not retained.*"
    )

if __name__ == "__main__":
    # ssr_mode=False: the server-side rendering path is experimental and is what the
    # failing /api_info requests were serving. show_api=False drops the route the bug
    # lives on entirely — nothing here needs a programmatic API.
    try:
        demo.launch(show_api=False, ssr_mode=False)
    except TypeError:           # older gradio without one of these kwargs
        demo.launch(show_api=False)
