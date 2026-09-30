"""Model catalog: every model the demo loads, defined once, by name.

Each entry is a Hugging Face repo and the file (or files) the demo needs from
it; `fetch` downloads them into the Hugging Face cache on first use and
returns the local path. The one exception is the face embedder, which has no
LiteRT build on the Hub: `scripts/convert_hsface.py` converts it into
`assets/`.

    uv run python -m emulator.models    # download everything up front

Swap a model for a stage by changing its entry or the stage's name below.
The embedding models and Whisper are loaded by name by their own libraries
(emulator/frame_memory.py, memory.py, whisper_asr.py); `main` fetches them too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"


@dataclass(frozen=True)
class Model:
    """One model: a single file in a Hugging Face repo (`repo` + `file`),
    several files from one (`repo` + `patterns`, fetched as a directory), or
    a file on this machine (`path`)."""

    repo: str | None = None
    file: str | None = None
    patterns: tuple[str, ...] = ()
    path: Path | None = None
    how_to_get: str = ""


MODELS: dict[str, Model] = {
    # YOLOX-Tiny, COCO, Apache-2.0. See emulator/detector.py for its contract.
    "yolox-tiny": Model(repo="litert-community/yolox-tiny-litert",
                        file="yolox_tiny.tflite"),
    # Speech recognition when it runs on the robot (the laptop uses Whisper,
    # emulator/whisper_asr.py). The tokenizer is the original model's.
    "moonshine-tiny": Model(repo="litert-community/moonshine-tiny",
                            file="moonshine_tiny_5s_f32.tflite"),
    "moonshine-tokenizer": Model(repo="UsefulSensors/moonshine-tiny",
                                 file="tokenizer.json"),
    # Speech synthesis: the LiteRT graphs, their runtime (say.py) and the
    # espeak-ng text frontend, all from one repo (emulator/inflect_tts.py).
    "inflect-nano-v2": Model(repo="litert-community/Inflect-Nano-v2",
                             patterns=("say.py", "frontend/*", "frontend/**/*",
                                       "inflect_text_encoder.tflite",
                                       "inflect_decoder.tflite")),
    # Faces: OpenCV's YuNet finds them, HSFace turns one into an identity.
    "yunet": Model(repo="opencv/face_detection_yunet",
                   file="face_detection_yunet_2023mar.onnx"),
    "hsface": Model(path=ASSETS / "hsface10k.tflite",
                    how_to_get="convert it with `uv run --with torch --with "
                               "litert-torch python scripts/convert_hsface.py`"),
    # The language model, in-process through litert-lm (emulator/engines.py).
    "gemma-4-e2b": Model(repo="litert-community/gemma-4-E2B-it-litert-lm",
                         file="gemma-4-E2B-it.litertlm"),
}

# Models the demo runs without: no face embedder, and the robot calls everyone
# "Person" — it still talks, remembers and recalls.
OPTIONAL = ("hsface",)

# Which model each stage uses. Swap a stage = change one name here.
DETECTOR = "yolox-tiny"
ASR = "moonshine-tiny"
TTS = "inflect-nano-v2"
LLM = "gemma-4-e2b"


def get(name: str) -> Model:
    """Look a model up by name; raise loudly on an unknown name."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(MODELS)}") from None


def fetch(model: str | Model) -> Path:
    """The local path to a model — a file, or a directory for a `patterns`
    model — downloading it on first use.

    The cache is tried first, offline: on the robot, which may have no route
    to the Hub, a model carried over by `scripts/robot_service.sh prepare`
    loads without a network round trip."""
    spec = get(model) if isinstance(model, str) else model
    if spec.path is not None:
        if not spec.path.exists():
            raise FileNotFoundError(
                f"{spec.path} is missing — {spec.how_to_get or 'see README.md'}")
        return spec.path
    from huggingface_hub import hf_hub_download, snapshot_download

    def download(**offline):
        if spec.patterns:
            return snapshot_download(repo_id=spec.repo,
                                     allow_patterns=list(spec.patterns),
                                     **offline)
        return hf_hub_download(repo_id=spec.repo, filename=spec.file, **offline)

    try:
        local = Path(download(local_files_only=True))
        # An offline snapshot answers with the folder even when only some of
        # its files were ever downloaded; the named ones must all be there.
        if all((local / name).exists() for name in spec.patterns
               if "*" not in name):
            return local
    except Exception:  # noqa: BLE001 — not cached yet: fetch it
        pass
    return Path(download())


def resolve_llm(name_or_path: str) -> Model:
    """The LLM to load: a catalog name, or a path to any .litertlm file — so
    comparing models downloaded outside the catalog needs a flag, not an edit
    (demo/serve.py's --llm)."""
    path = Path(name_or_path).expanduser()
    if path.suffix == ".litertlm" or path.is_file():
        if not path.is_file():
            raise SystemExit(f"model file not found: {path}")
        return Model(path=path)
    return get(name_or_path)


def main() -> int:
    """Download every model in the catalog, and the embedding models the
    laptop's services load by name (SigLIP 2, bge-small, Whisper)."""
    failed = []
    for name, spec in MODELS.items():
        try:
            print(f"  {name:<20} {fetch(spec)}", flush=True)
        except Exception as exc:  # noqa: BLE001 — report every missing model
            if name in OPTIONAL:
                print(f"  {name:<20} not there (optional): {exc}", flush=True)
                continue
            print(f"  {name:<20} MISSING: {exc}", flush=True)
            failed.append(name)
    from emulator.frame_memory import SiglipEmbedder
    from emulator.memory import DEFAULT_MODEL, _embedder
    from emulator.whisper_asr import DEFAULT_MODEL as WHISPER, WhisperRecognizer

    for label, load in (("siglip2", SiglipEmbedder),
                        ("bge-small", lambda: _embedder(DEFAULT_MODEL)),
                        (f"whisper {WHISPER}", lambda: WhisperRecognizer(WHISPER))):
        try:
            load()
            print(f"  {label:<20} ready", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<20} MISSING: {exc}", flush=True)
            failed.append(label)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
