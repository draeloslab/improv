import os
import uuid

import numpy as np
import pytest

from experiments.FES.actors import shared_frame_ring as rings


def _name():
    return "improv_test_{}_{}".format(os.getpid(), uuid.uuid4().hex)


@pytest.fixture
def frame_ring():
    ring = rings.FrameRing.create(_name(), (3, 4, 3), n_slots=3)
    try:
        yield ring
    finally:
        ring.unlink()
        ring.close()


def test_name_is_deterministic_and_sanitized():
    assert rings.ring_name("run 1/alpha", 2) == "improv_run_1_alpha_cam2"


def test_publish_and_attach(frame_ring):
    reader = rings.FrameRing.attach(
        frame_ring.name, timeout=0.1, expect_shape=(3, 4, 3)
    )
    try:
        frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        sequence = frame_ring.write(
            frame, frame_number=91, capture_time=1.25, camera_start=2.5
        )

        result = reader.read(sequence)
        assert result is not None
        copied, metadata = result
        assert np.array_equal(copied, frame)
        assert copied is not frame
        assert metadata[:3] == (91, 1.25, 2.5)
        assert metadata[3] > 0
    finally:
        reader.close()


def test_overwritten_frame_is_rejected(frame_ring):
    frame = np.zeros(frame_ring.shape, dtype=np.uint8)
    for frame_number in range(4):
        frame.fill(frame_number)
        frame_ring.write(frame, frame_number)

    assert frame_ring.read(0) is None
    assert frame_ring.oldest_available() == 1
    assert frame_ring.read(1)[1][0] == 1


def test_follower_counts_overwritten_frames(frame_ring):
    follower = rings.RingFollower(frame_ring, start="oldest")
    frame = np.zeros(frame_ring.shape, dtype=np.uint8)
    for frame_number in range(5):
        frame.fill(frame_number)
        frame_ring.write(frame, frame_number)

    available = follower.poll()
    assert [item[0] for item in available] == [2, 3, 4]
    assert [item[2][0] for item in available] == [2, 3, 4]
    assert follower.lost == 2


def test_writer_validates_before_invalidating_a_slot(frame_ring):
    good = np.zeros(frame_ring.shape, dtype=np.uint8)
    sequence = frame_ring.write(good, 1)

    with pytest.raises(TypeError):
        frame_ring.write(good.astype(np.float32), 2)
    with pytest.raises(ValueError):
        frame_ring.write(good[:, :-1], 2)

    assert frame_ring.read(sequence) is not None
    assert frame_ring.latest_sequence() == sequence


def test_live_writer_collision_is_never_replaced(frame_ring):
    with pytest.raises(rings.FrameRingExistsError):
        rings.FrameRing.create(frame_ring.name, frame_ring.shape, 3)
    with pytest.raises(rings.FrameRingInUseError):
        rings.FrameRing.create(
            frame_ring.name, frame_ring.shape, 3, replace_stale=True
        )


def test_mark_closed_stops_writes(frame_ring):
    frame_ring.mark_closed()
    assert frame_ring.closed
    with pytest.raises(rings.FrameRingError):
        frame_ring.write(np.zeros(frame_ring.shape, dtype=np.uint8), 1)


def test_bad_follower_start_is_rejected(frame_ring):
    with pytest.raises(ValueError):
        rings.RingFollower(frame_ring, start="typo")
