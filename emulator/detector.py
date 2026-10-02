"""Object detector: frame in, list of findings out.

Its role is a cheap, always-on watcher: it tells the robot what is in view,
and a change in that set is when a frame is worth remembering. Hence two
requirements: stay cheap, and don't count frame jitter as an event.

The model is Ultralytics' **YOLO26n** (COCO, AGPL-3.0) in the LiteRT export
Arm publishes as `Arm/yolo26n-fp16-litert`. It is downloaded on first use and
is not part of this repository (NOTICE). Its contract, from the model's
manifest and checked against the tensors on its sample photo:

- **Input** `[1, 3, 640, 640]`, NCHW, **RGB in [0, 1]**, letterboxed: scaled
  uniformly to fit, padded bottom/right with gray 114/255.
- **Output** `[1, 300, 6]`: one row per detection, best first — the box
  corners (x1, y1, x2, y2) in input pixels, the score, the COCO class. YOLO26
  is NMS-free: its end-to-end head gives one box per object, so nothing is
  suppressed here.

That head lowers to INT64 select ops the LiteRT GPU delegate rejects (on the
Mac's Metal as on the Pi 5's VideoCore), so the detector runs on CPU cores
wherever a frame becomes boxes: this `Detector`, on the robot
(`--on-robot detector`) and in the laptop's service (demo/gpu_detect.py),
where a frame takes about 20 ms.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

PAD_VALUE = 114.0 / 255.0
ROW = 6   # x1, y1, x2, y2, score, class


@dataclasses.dataclass(frozen=True)
class Detection:
    label: int
    score: float
    box: tuple[float, float, float, float]
    """Coordinates as fractions of the frame: (x1, y1, x2, y2)."""


@dataclasses.dataclass(frozen=True)
class Letterbox:
    """How a frame was fitted into the model's square input — what the
    decoder needs to map boxes back onto the frame."""

    ratio: float
    frame_width: int
    frame_height: int


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """An RGB HWC frame of any size as the model's input: `[1, 3, size,
    size]` float32 in [0, 1], scaled to fit and padded bottom/right with
    114/255.

    A frame of [0, 1] floats is taken as it is; one of raw 0–255 pixels,
    as a decoded JPEG comes, is scaled down."""
    from PIL import Image

    data = np.asarray(frame)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB frame, got shape {data.shape}")
    if np.issubdtype(data.dtype, np.floating):
        if data.size and data.max() <= 1.0:
            data = data * 255.0
        data = np.clip(data, 0, 255).astype(np.uint8)
    height, width = data.shape[:2]
    ratio = min(size / width, size / height)
    new_w, new_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    resized = np.asarray(Image.fromarray(data).resize((new_w, new_h),
                                                      Image.BILINEAR),
                         dtype=np.float32) / 255.0
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.float32)
    canvas[:new_h, :new_w] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1))[None]   # NCHW
    return tensor, Letterbox(ratio=ratio, frame_width=width, frame_height=height)


def decode_output(raw: np.ndarray, score_threshold: float,
                  fit: Letterbox) -> list[Detection]:
    """The `[1, N, 6]` head as detections in frame fractions, those under
    the threshold dropped."""
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != ROW:
        raise ValueError(
            f"detector output shape {raw.shape} is not the YOLO26 end-to-end "
            f"head [1, N, {ROW}]")
    rows = raw[0]
    rows = rows[rows[:, 4] >= score_threshold]
    if not len(rows):
        return []
    # Input pixels -> frame pixels -> fractions of the frame. Clipped: a box
    # can reach past the edge, and the robot turns its head by these.
    corners = rows[:, :4] / fit.ratio
    corners /= np.array([fit.frame_width, fit.frame_height,
                         fit.frame_width, fit.frame_height], dtype=np.float32)
    corners = np.clip(corners, 0.0, 1.0)
    return [
        Detection(label=int(row[5]), score=float(row[4]),
                  box=(float(box[0]), float(box[1]),
                       float(box[2]), float(box[3])))
        for row, box in zip(rows, corners)
    ]


def scene_changed(previous: list[Detection],
                  current: list[Detection]) -> bool:
    """Whether the SET of classes in the frame changed.

    We compare sets of classes, not boxes: an object moving doesn't count
    as an event, otherwise every camera jitter would be one.
    """
    return {d.label for d in previous} != {d.label for d in current}


class Detector:
    """YOLO26n on this machine's CPU cores."""

    def __init__(self, model_path: Path, threads: int = 4,
                 score_threshold: float = 0.3) -> None:
        # The runner is chosen for the board (emulator/litert_runtime.py):
        # CompiledModel where it can be built, the Interpreter on the robot's
        # CM4, where it cannot.
        from emulator.litert_runtime import build_runner

        self._runner = build_runner(model_path, threads=threads)
        self._sig = self._runner.only()
        (self._in_name, in_meta), = self._sig.get_input_details().items()
        _, channels, height, width = (int(x) for x in in_meta["shape"])
        if channels != 3 or height != width:
            raise ValueError(f"expected a [1, 3, N, N] input, got "
                             f"{list(in_meta['shape'])}")
        self._size = height
        self._in_dtype = in_meta["dtype"]
        self._threshold = score_threshold

    @property
    def input_size(self) -> int:
        return self._size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, fit = letterbox(frame, self._size)
        out = self._sig(**{self._in_name: tensor.astype(self._in_dtype.type)})
        raw = next(iter(out.values()))
        return decode_output(raw, self._threshold, fit)
