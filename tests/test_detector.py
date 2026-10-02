"""The detector: YOLO26n's input and output contract, duplicate
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
    decode_output,
    letterbox,
    non_max_suppression,
    scene_changed,
)

SIZE = 640


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


# --- letterbox: RGB in [0, 1], NCHW, scaled to fit, padded with 114/255 ---

def test_letterbox_scales_to_unit_range_keeps_rgb_and_puts_channels_first():
    frame = np.zeros((SIZE, SIZE, 3), np.uint8)
    frame[..., 0] = 255            # red
    tensor, fit = letterbox(frame, SIZE)
    assert tensor.shape == (1, 3, SIZE, SIZE) and tensor.dtype == np.float32
    assert tensor[0, 0, 10, 10] == pytest.approx(1.0)     # R, first channel
    assert tensor[0, 2, 10, 10] == pytest.approx(0.0)     # B
    assert fit == Letterbox(ratio=1.0, frame_width=SIZE, frame_height=SIZE)


def test_letterbox_fits_a_wide_frame_and_pads_the_bottom():
    frame = np.full((540, 960, 3), 51, np.uint8)
    tensor, fit = letterbox(frame, SIZE)
    assert fit.ratio == pytest.approx(SIZE / 960)
    height = round(540 * fit.ratio)            # 360 rows of picture
    assert tensor[0, 0, height - 1, 0] == pytest.approx(51 / 255)
    assert tensor[0, 0, height + 1, 0] == pytest.approx(114 / 255)


def test_letterbox_takes_a_unit_float_frame_as_it_is():
    frame = np.full((SIZE, SIZE, 3), 0.5, np.float32)
    tensor, _ = letterbox(frame, SIZE)
    assert tensor[0, 0, 0, 0] == pytest.approx(0.5, abs=0.01)


def test_letterbox_refuses_a_frame_without_three_channels():
    with pytest.raises(ValueError):
        letterbox(np.zeros((SIZE, SIZE), np.uint8), SIZE)


# --- decoding the end-to-end head: [1, 300, 6], corners in input pixels ---

def _raw(*rows):
    raw = np.zeros((1, 300, 6), np.float32)
    for i, row in enumerate(rows):
        raw[0, i] = row
    return raw


def test_decode_maps_input_pixels_back_to_frame_fractions():
    # A cup at (96..128, 160..192) on a canvas the frame was halved into.
    raw = _raw((96, 160, 128, 192, 0.8, 41))
    fit = Letterbox(ratio=0.5, frame_width=832, frame_height=832)
    [found] = decode_output(raw, 0.3, fit)
    assert found.label == 41 and found.score == pytest.approx(0.8)
    assert found.box == pytest.approx((192 / 832, 320 / 832, 256 / 832, 384 / 832))


def test_decode_drops_what_is_under_the_threshold():
    raw = _raw((0, 0, 10, 10, 0.9, 0), (0, 0, 10, 10, 0.25, 0))
    found = decode_output(raw, 0.3, Letterbox(1.0, SIZE, SIZE))
    assert [round(d.score, 2) for d in found] == [0.9]


def test_decode_returns_every_row_over_the_threshold():
    # Overlapping or not: suppressing repeats is the next step, not this one.
    raw = _raw((0, 0, 100, 100, 0.9, 0), (5, 5, 105, 105, 0.8, 0))
    assert len(decode_output(raw, 0.3, Letterbox(1.0, SIZE, SIZE))) == 2


def test_decode_clips_boxes_to_the_frame():
    raw = _raw((-50, -50, 900, 900, 0.9, 0))
    [found] = decode_output(raw, 0.3, Letterbox(1.0, SIZE, SIZE))
    assert found.box == (0.0, 0.0, 1.0, 1.0)


def test_decode_returns_nothing_for_an_empty_scene():
    assert decode_output(_raw(), 0.3, Letterbox(1.0, SIZE, SIZE)) == []


def test_decode_refuses_an_output_it_does_not_understand():
    with pytest.raises(ValueError):
        decode_output(np.zeros((1, 84, 8400), np.float32), 0.3,
                      Letterbox(1.0, SIZE, SIZE))
    with pytest.raises(ValueError):
        decode_output(np.zeros((1, 3549, 85), np.float32), 0.3,
                      Letterbox(1.0, SIZE, SIZE))


# --- duplicate suppression ---

def test_nms_collapses_overlapping_boxes_of_same_class():
    a = Detection(label=0, score=0.74, box=(0.1, 0.5, 0.3, 1.0))
    b = Detection(label=0, score=0.31, box=(0.1, 0.51, 0.3, 1.0))
    kept = non_max_suppression([b, a])
    assert kept == [a], "the most confident detection must be kept"


def test_nms_keeps_distant_boxes():
    a = Detection(label=0, score=0.9, box=(0.0, 0.0, 0.2, 0.2))
    b = Detection(label=0, score=0.8, box=(0.7, 0.7, 0.9, 0.9))
    assert len(non_max_suppression([a, b])) == 2


def test_nms_keeps_different_classes_at_same_place():
    # A person holding a cup occupy the same region — both are needed.
    a = Detection(label=0, score=0.9, box=(0.1, 0.1, 0.5, 0.5))
    b = Detection(label=41, score=0.8, box=(0.1, 0.1, 0.5, 0.5))
    assert len(non_max_suppression([a, b])) == 2


def test_nms_handles_empty_input():
    assert non_max_suppression([]) == []


# --- Detector: the runner, and a real frame ---

class _FakeSig:
    def __init__(self, shape):
        self._shape = shape

    def get_input_details(self):
        return {"images": {"shape": self._shape, "dtype": np.dtype(np.float32)}}


def _fake_runner(captured, shape=(1, 3, SIZE, SIZE)):
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


def test_detector_refuses_an_input_that_is_not_three_square_channels(monkeypatch):
    monkeypatch.setattr("emulator.litert_runtime.build_runner",
                        _fake_runner({}, shape=(1, SIZE, SIZE, 3)))
    with pytest.raises(ValueError):
        Detector(Path("fake.tflite"))


def _cached(filename):
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download("Arm/yolo26n-fp16-litert", filename,
                               local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached: skip rather than download
        return None


@pytest.mark.skipif(_cached("yolo26n_conv2d_f16_weights.tflite") is None
                    or _cached("samples/sample.jpg") is None,
                    reason="yolo26n or its sample photo is not cached")
def test_detector_finds_people_in_the_models_own_sample_photo():
    """The whole path — letterbox, model, decode — on the model's own photo:
    the people it shows, where they stand. A frame fed as raw 0–255 finds
    "people" everywhere and a scrambled layout finds none; a channel-order
    mix-up cannot be told apart on this photo (the synthetic letterbox tests
    pin that)."""
    from PIL import Image

    photo = np.asarray(Image.open(_cached("samples/sample.jpg")).convert("RGB"))
    found = Detector(Path(_cached("yolo26n_conv2d_f16_weights.tflite")),
                     threads=2).detect(photo)
    people = [d for d in found if d.label == 0]
    assert 6 <= len(people) <= 8
    best = max(people, key=lambda d: d.score)
    assert _iou(best.box, (0.015, 0.76, 0.11, 0.93)) > 0.8
    assert all(0.0 <= v <= 1.0 for d in found for v in d.box)


@pytest.mark.skipif(_cached("yolo26n_conv2d_f16_weights.tflite") is None
                    or _cached("samples/sample.jpg") is None,
                    reason="yolo26n or its sample photo is not cached")
def test_a_person_cut_off_by_the_frame_is_one_box(monkeypatch):
    """Why there is NMS: cropped so the people are cut off at the bottom —
    how the robot sees whoever it talks to — the head gives one of them
    twice, and the detector keeps one."""
    from PIL import Image

    import emulator.detector as detector_module

    detector = Detector(Path(_cached("yolo26n_conv2d_f16_weights.tflite")),
                        threads=2)
    crop = np.asarray(Image.open(_cached("samples/sample.jpg")).convert("RGB")
                      .crop((84, 557, 693, 1174)))

    def repeats(found):
        return [(a, b) for i, a in enumerate(found) for b in found[i + 1:]
                if a.label == b.label and _iou(a.box, b.box) > 0.9]

    kept = detector.detect(crop)
    monkeypatch.setattr(detector_module, "non_max_suppression", lambda found: found)
    assert repeats(detector.detect(crop)), "the head repeats a person here"
    assert not repeats(kept)


def _iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0
