"""Object detector: frame in, list of findings out.

Its role is a cheap, always-on watcher: it tells the robot what is in view,
and a change in that set is when a frame is worth remembering. Hence two
requirements: stay cheap, and don't count frame jitter as an event.

The model is Megvii's **YOLOX-Tiny** (COCO, Apache-2.0) in the LiteRT export
from `litert-community/yolox-tiny-litert`. Its contract, from the model card
and checked against the tensors:

- **Input** `[1, 416, 416, 3]`, NHWC, **BGR, raw 0–255, no normalization**,
  letterboxed: scaled uniformly to fit, padded bottom/right with gray 114.
  Dividing by 255 does not rescale this model, it blinds it.
- **Output** `[1, 3549, 85]`, anchor-major: 4 box values, objectness, 80 class
  scores (both already sigmoid'd). 3549 = 52² + 26² + 13², one row per grid
  cell at strides 8, 16 and 32. The boxes are **not decoded** in the graph —
  that keeps it clean for the GPU delegate — so the grid is applied here:
  centre = (offset + cell) × stride, size = exp(value) × stride.

The same preparation and decoding serve both places a frame becomes boxes:
this `Detector` on CPU cores (on the robot, `--on-robot detector`) and the
laptop's GPU service (demo/gpu_detect.py).
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np

STRIDES = (8, 16, 32)
PAD_VALUE = 114.0
CLASSES = 80


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
    """An RGB HWC frame of any size as the model's input: `[1, size, size, 3]`
    BGR float32 in 0–255, scaled to fit and padded bottom/right with 114.

    A frame in [0, 1] floats is scaled up first: the model reads raw pixel
    values, and a [0, 1] frame would look almost black to it."""
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
                         dtype=np.float32)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.float32)
    canvas[:new_h, :new_w] = resized
    tensor = np.ascontiguousarray(canvas[..., ::-1])[None]   # RGB -> BGR, NHWC
    return tensor, Letterbox(ratio=ratio, frame_width=width, frame_height=height)


@functools.lru_cache(maxsize=4)
def build_grid(input_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Each output row's grid cell (x, y) and stride. Cached: it depends only
    on the input size, and is needed on every frame."""
    grids = []
    strides = []
    for stride in STRIDES:
        side = input_size // stride
        ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        grids.append(np.stack((xs, ys), axis=2).reshape(-1, 2))
        strides.append(np.full((side * side, 1), stride, dtype=np.float32))
    return (np.concatenate(grids).astype(np.float32),
            np.concatenate(strides))


def decode_yolox_output(raw: np.ndarray, score_threshold: float,
                        input_size: int, fit: Letterbox) -> list[Detection]:
    """The raw `[1, N, 85]` head as detections in frame fractions.

    Score is objectness × the best class score, as the model card specifies:
    objectness alone says "something is here", not what."""
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 5 + CLASSES:
        raise ValueError(
            f"detector output shape {raw.shape} is not the YOLOX head "
            f"[1, N, {5 + CLASSES}]")
    predictions = raw[0]
    grid, strides = build_grid(input_size)
    if grid.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"the model returned {predictions.shape[0]} rows, the grid for a "
            f"{input_size} input has {grid.shape[0]}: strides {STRIDES} do "
            "not fit this export")

    class_scores = predictions[:, 5:]
    labels = class_scores.argmax(axis=1)
    scores = predictions[:, 4] * class_scores.max(axis=1)
    keep = scores >= score_threshold
    if not keep.any():
        return []

    rows = predictions[keep]
    centers = (rows[:, :2] + grid[keep]) * strides[keep]
    # Width and height are on a log scale; clipped so a wild value cannot
    # overflow exp().
    sizes = np.exp(np.clip(rows[:, 2:4], -10.0, 10.0)) * strides[keep]
    half = sizes / 2.0
    corners = np.concatenate([centers - half, centers + half], axis=1)
    # Canvas pixels -> frame pixels -> fractions of the frame. Clipped: YOLOX
    # can predict past the edge, and the robot turns its head by these.
    corners = corners / fit.ratio
    corners /= np.array([fit.frame_width, fit.frame_height,
                         fit.frame_width, fit.frame_height], dtype=np.float32)
    corners = np.clip(corners, 0.0, 1.0)

    return [
        Detection(label=int(label), score=float(score),
                  box=(float(box[0]), float(box[1]),
                       float(box[2]), float(box[3])))
        for label, score, box in zip(labels[keep], scores[keep], corners)
    ]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if overlap <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection],
                        iou_threshold: float = 0.45) -> list[Detection]:
    """Keep a single box per object.

    The raw head leaves neighbouring cells reporting the same object — on a
    photo of a dog, fourteen rows above threshold, one dog in fourteen boxes.
    Classes are suppressed independently: a person holding a dog occupy the
    same region of the frame, and both are needed.
    """
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(_iou(candidate.box, other.box) <= iou_threshold
               for other in kept if other.label == candidate.label):
            kept.append(candidate)
    return kept


def scene_changed(previous: list[Detection],
                  current: list[Detection]) -> bool:
    """Whether the SET of classes in the frame changed.

    We compare sets of classes, not boxes: an object moving doesn't count
    as an event, otherwise every camera jitter would be one.
    """
    return {d.label for d in previous} != {d.label for d in current}


class Detector:
    """YOLOX on this machine's CPU cores."""

    def __init__(self, model_path: Path, threads: int = 4,
                 score_threshold: float = 0.3,
                 iou_threshold: float = 0.45) -> None:
        # The runner is chosen for the board (emulator/litert_runtime.py):
        # CompiledModel where it can be built, the Interpreter on the robot's
        # CM4, where it cannot.
        from emulator.litert_runtime import build_runner

        self._runner = build_runner(model_path, threads=threads)
        self._sig = self._runner.only()
        (self._in_name, in_meta), = self._sig.get_input_details().items()
        _, height, width, _ = (int(x) for x in in_meta["shape"])
        if height != width:
            raise ValueError(f"expected a square input, got {height}x{width}")
        self._size = height
        self._in_dtype = in_meta["dtype"]
        self._threshold = score_threshold
        self._iou = iou_threshold

    @property
    def input_size(self) -> int:
        return self._size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, fit = letterbox(frame, self._size)
        out = self._sig(**{self._in_name: tensor.astype(self._in_dtype.type)})
        raw = next(iter(out.values()))
        found = decode_yolox_output(raw, self._threshold, self._size, fit)
        return non_max_suppression(found, self._iou)
