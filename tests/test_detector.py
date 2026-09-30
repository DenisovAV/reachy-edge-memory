"""The detector: YOLOX-Tiny's input and output contract, duplicate
suppression, and what counts as a change of scene.

The preparation and decoding are tested on synthetic tensors; the model
itself runs in test_detector_finds_people_in_the_models_own_sample_photo when it is in the
local Hugging Face cache (it is skipped otherwise, never downloaded here).
"""

from pathlib import Path

import numpy as np
import pytest

from emulator.detector import (
    Detection,
    Detector,
    Letterbox,
    build_grid,
    decode_yolox_output,
    letterbox,
    non_max_suppression,
    scene_changed,
)

SIZE = 416
ROWS = (52 * 52) + (26 * 26) + (13 * 13)   # 3549


def det(label: int, score: float = 0.9) -> Detection:
    return Detection(label=label, score=score, box=(0.1, 0.1, 0.5, 0.5))


# --- scene changes: the set of classes, not their boxes ---

def test_scene_change_on_new_label():
    assert scene_changed([det(0)], [det(0), det(41)])


def test_scene_change_on_disappearance():
    assert scene_changed([det(0), det(41)], [det(0)])


def test_no_change_when_labels_same():
    moved = Detection(label=0, score=0.9, box=(0.3, 0.3, 0.7, 0.7))
    assert not scene_changed([det(0)], [moved])


def test_no_change_when_score_wobbles():
    assert not scene_changed([det(0, 0.91)], [det(0, 0.62)])


def test_empty_to_empty_is_not_a_change():
    assert not scene_changed([], [])


# --- letterbox: raw BGR pixels, scaled to fit, padded with 114 ---

def test_letterbox_keeps_raw_pixel_values_and_flips_to_bgr():
    frame = np.zeros((SIZE, SIZE, 3), np.uint8)
    frame[..., 0] = 200            # red
    tensor, fit = letterbox(frame, SIZE)
    assert tensor.shape == (1, SIZE, SIZE, 3) and tensor.dtype == np.float32
    # Not divided by 255: this model reads raw pixels, and BGR, not RGB.
    assert tensor[0, 10, 10, 2] == pytest.approx(200)
    assert tensor[0, 10, 10, 0] == pytest.approx(0)
    assert fit == Letterbox(ratio=1.0, frame_width=SIZE, frame_height=SIZE)


def test_letterbox_fits_a_wide_frame_and_pads_the_bottom():
    frame = np.full((540, 960, 3), 10, np.uint8)
    tensor, fit = letterbox(frame, SIZE)
    assert fit.ratio == pytest.approx(SIZE / 960)
    height = round(540 * fit.ratio)            # 234 rows of picture
    assert tensor[0, height - 1, 0, 0] == pytest.approx(10)
    assert tensor[0, height + 1, 0, 0] == pytest.approx(114)


def test_letterbox_scales_a_unit_float_frame_up():
    frame = np.full((SIZE, SIZE, 3), 0.5, np.float32)
    tensor, _ = letterbox(frame, SIZE)
    assert tensor[0, 0, 0, 0] == pytest.approx(127, abs=1)


def test_letterbox_refuses_a_frame_without_three_channels():
    with pytest.raises(ValueError):
        letterbox(np.zeros((SIZE, SIZE), np.uint8), SIZE)


# --- decoding the raw head: [1, 3549, 85], boxes in grid units ---

def _raw():
    return np.zeros((1, ROWS, 85), np.float32)


def _set(raw, row, *, offset=(0.5, 0.5), log_size=(0.0, 0.0), obj=0.9,
         label=0, class_score=0.9):
    raw[0, row, 0:2] = offset
    raw[0, row, 2:4] = log_size
    raw[0, row, 4] = obj
    raw[0, row, 5 + label] = class_score


def _row_for(stride, gx, gy):
    """The output row of grid cell (gx, gy) at a stride."""
    start = 0
    for s in (8, 16, 32):
        side = SIZE // s
        if s == stride:
            return start + gy * side + gx
        start += side * side
    raise ValueError(stride)


def test_grid_has_one_row_per_cell_at_three_strides():
    grid, strides = build_grid(SIZE)
    assert grid.shape == (ROWS, 2) and strides.shape == (ROWS, 1)
    assert tuple(grid[_row_for(32, 3, 5)]) == (3, 5)
    assert strides[_row_for(32, 3, 5), 0] == 32


def test_decode_applies_the_grid_and_maps_back_to_the_frame():
    raw = _raw()
    # A 32-px box centred on cell (3, 5) at stride 32: centre (112, 176).
    _set(raw, _row_for(32, 3, 5), log_size=(0.0, 0.0), label=41)
    fit = Letterbox(ratio=0.5, frame_width=832, frame_height=832)
    [found] = decode_yolox_output(raw, 0.3, SIZE, fit)
    assert found.label == 41
    # Canvas (96..128, 160..192) / ratio 0.5 = frame pixels, / 832.
    assert found.box == pytest.approx((192 / 832, 320 / 832, 256 / 832, 384 / 832))


def test_decode_scores_objectness_times_class():
    raw = _raw()
    _set(raw, 0, obj=0.5, class_score=0.5)       # 0.25: under the threshold
    _set(raw, 1, obj=0.9, class_score=0.8)       # 0.72
    fit = Letterbox(1.0, SIZE, SIZE)
    found = decode_yolox_output(raw, 0.3, SIZE, fit)
    assert [round(d.score, 2) for d in found] == [0.72]


def test_decode_clips_boxes_to_the_frame():
    raw = _raw()
    _set(raw, _row_for(32, 0, 0), log_size=(3.0, 3.0))    # far wider than the frame
    [found] = decode_yolox_output(raw, 0.3, SIZE, Letterbox(1.0, SIZE, SIZE))
    assert all(0.0 <= v <= 1.0 for v in found.box)


def test_decode_returns_nothing_under_the_threshold():
    assert decode_yolox_output(_raw(), 0.3, SIZE, Letterbox(1.0, SIZE, SIZE)) == []


def test_decode_refuses_an_output_it_does_not_understand():
    with pytest.raises(ValueError):
        decode_yolox_output(np.zeros((1, 84, 8400), np.float32), 0.3, SIZE,
                            Letterbox(1.0, SIZE, SIZE))
    with pytest.raises(ValueError):
        decode_yolox_output(np.zeros((1, 100, 85), np.float32), 0.3, SIZE,
                            Letterbox(1.0, SIZE, SIZE))


# --- duplicate suppression ---

def test_nms_collapses_overlapping_boxes_of_same_class():
    a = Detection(label=15, score=0.9, box=(0.1, 0.1, 0.5, 0.5))
    b = Detection(label=15, score=0.7, box=(0.12, 0.12, 0.52, 0.52))
    kept = non_max_suppression([a, b], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].score == 0.9, "the most confident detection must be kept"


def test_nms_keeps_distant_boxes():
    a = Detection(label=15, score=0.9, box=(0.0, 0.0, 0.2, 0.2))
    b = Detection(label=15, score=0.8, box=(0.7, 0.7, 0.9, 0.9))
    assert len(non_max_suppression([a, b], iou_threshold=0.5)) == 2


def test_nms_keeps_different_classes_at_same_place():
    # A person holding a dog occupy the same region — both are needed.
    a = Detection(label=0, score=0.9, box=(0.1, 0.1, 0.5, 0.5))
    b = Detection(label=15, score=0.8, box=(0.1, 0.1, 0.5, 0.5))
    assert len(non_max_suppression([a, b], iou_threshold=0.5)) == 2


def test_nms_handles_empty_input():
    assert non_max_suppression([], iou_threshold=0.5) == []


# --- Detector: the runner, and a real frame ---

class _FakeSig:
    def __init__(self, shape):
        self._shape = shape

    def get_input_details(self):
        return {"images": {"shape": self._shape, "dtype": np.dtype(np.float32)}}


def _fake_runner(captured, shape=(1, SIZE, SIZE, 3)):
    class FakeRunner:
        def __init__(self, model_path, threads=4, **kwargs):
            captured["threads"] = threads

        def only(self):
            return _FakeSig(shape)
    return FakeRunner


def test_detector_reads_its_input_size_and_forwards_threads(monkeypatch):
    captured = {}
    monkeypatch.setattr("emulator.litert_runtime.build_runner",
                        _fake_runner(captured))
    detector = Detector(Path("fake.tflite"), threads=2)
    assert detector.input_size == SIZE
    assert captured["threads"] == 2


def _cached(filename):
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download("litert-community/yolox-tiny-litert", filename,
                               local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached: skip rather than download
        return None


@pytest.mark.skipif(_cached("yolox_tiny.tflite") is None
                    or _cached("samples/sample.png") is None,
                    reason="yolox-tiny or its sample photo is not cached")
def test_detector_finds_people_in_the_models_own_sample_photo():
    """The one check that the whole path — letterbox, model, decode — sees
    what a camera sees: a scrambled input or a wrong channel order finds
    nothing here."""
    from PIL import Image

    photo = np.asarray(Image.open(_cached("samples/sample.png")).convert("RGB"))
    found = Detector(Path(_cached("yolox_tiny.tflite")), threads=2).detect(photo)
    people = [d for d in found if d.label == 0]
    assert people and max(d.score for d in people) > 0.5
    assert all(0.0 <= v <= 1.0 for d in found for v in d.box)
