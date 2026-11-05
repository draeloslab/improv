"""Shared-memory transport for high-rate camera frames.

The ring removes full image arrays from improv's Redis and multiprocessing
queue paths. A camera is the only writer for its ring; processors, displays,
and recorders may attach as readers. Queue messages should contain only the
camera number and published sequence number.

Readers copy a slot and verify its sequence both before and after the copy.
Slow readers do not block capture: once the writer laps a reader, the reader
observes an explicit miss.

The publication protocol uses aligned int64 loads and stores as a small
seqlock. It is intended for the FES rig's x86-64 CPU. Python and NumPy do not
provide portable inter-process acquire/release atomics, so this implementation
refuses to run on other CPU architectures.
"""

import hashlib
import math
import os
import platform
import re
import tempfile
import time
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np
import psutil


MAGIC = 0x46524D52494E4732  # "FRMRING2"
VERSION = 2

_HEADER_LEN = 16
(
    _H_MAGIC,
    _H_VERSION,
    _H_NSLOTS,
    _H_HEIGHT,
    _H_WIDTH,
    _H_CHANNELS,
    _H_LATEST,
    _H_CLOSED,
    _H_WRITER_PID,
) = range(9)

_TIME_COLS = 3
_T_CAPTURE, _T_CAMERA_START, _T_WRITE_DONE = range(_TIME_COLS)

_VALID_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_X86_MACHINES = {"amd64", "x86_64", "x64"}


class FrameRingError(RuntimeError):
    """Base class for frame-ring errors."""


class FrameRingExistsError(FrameRingError):
    """A segment exists but cannot safely be replaced."""


class FrameRingInUseError(FrameRingExistsError):
    """A live writer already owns the requested ring name."""


def ring_name(run_id, camera_num):
    """Return the deterministic ring name shared by actors in one run."""

    safe_run_id = _VALID_NAME.sub("_", str(run_id)).strip("_.-")
    if not safe_run_id:
        raise ValueError("run_id must contain at least one letter or digit")

    camera_num = int(camera_num)
    if camera_num < 0:
        raise ValueError("camera_num must be non-negative")

    return "improv_{}_cam{}".format(safe_run_id, camera_num)


def _require_x86():
    """Reject architectures without the memory-ordering assumptions we use."""

    machine = platform.machine().lower()
    if machine not in _X86_MACHINES:
        raise RuntimeError(
            "FrameRing's lock-free publication protocol is supported only on "
            "x86-64; this machine reports {!r}".format(platform.machine())
        )


def _validate_geometry(shape, n_slots):
    """Normalize and validate a ring's frame geometry."""

    if len(shape) != 3:
        raise ValueError("shape must be (height, width, channels)")

    shape = tuple(int(value) for value in shape)
    if any(value <= 0 for value in shape):
        raise ValueError("all frame dimensions must be positive")

    n_slots = int(n_slots)
    if n_slots < 2:
        raise ValueError("n_slots must be at least 2")

    return shape, n_slots


def _layout(shape, n_slots):
    """Return byte offsets and total requested segment size."""

    height, width, channels = shape

    int64_size = np.dtype(np.int64).itemsize
    float64_size = np.dtype(np.float64).itemsize

    offset_sequence = _HEADER_LEN * int64_size
    offset_frame_number = offset_sequence + n_slots * int64_size
    offset_times = offset_frame_number + n_slots * int64_size

    offset_frames = (
        offset_times
        + n_slots * _TIME_COLS * float64_size
    )

    # Keep pixel data on a cache-line boundary.
    offset_frames = (offset_frames + 63) // 64 * 64

    total_size = (
        offset_frames
        + n_slots * height * width * channels
    )

    return (
        offset_sequence,
        offset_frame_number,
        offset_times,
        offset_frames,
        total_size,
    )


def _unregister_from_resource_tracker(segment):
    """Opt a Python 3.8-3.12 attachment out of tracker ownership."""

    from multiprocessing import resource_tracker

    resource_tracker.unregister(segment._name, "shared_memory")


def _open_reader(name):
    """Open an existing ring without giving the reader ownership.

    Python 3.13 added the public ``track=False`` argument. Earlier versions
    require the isolated resource-tracker workaround below.
    """

    try:
        return shared_memory.SharedMemory(
            name=name,
            create=False,
            track=False,
        )
    except TypeError:
        segment = shared_memory.SharedMemory(
            name=name,
            create=False,
        )

        try:
            _unregister_from_resource_tracker(segment)
        except Exception:
            segment.close()
            raise

        return segment


def _open_for_stale_check(name):
    """Open a ring for validation and possible unlinking.

    Returns:
        (segment, tracker_registered)

    The segment stays registered on Python 3.8-3.12 until we know whether it
    will be unlinked. This avoids registering and unregistering the same name
    twice.
    """

    try:
        segment = shared_memory.SharedMemory(
            name=name,
            create=False,
            track=False,
        )
        return segment, False
    except TypeError:
        segment = shared_memory.SharedMemory(
            name=name,
            create=False,
        )
        return segment, True


def _pid_is_alive(pid):
    """Check PID existence without sending a signal.

    ``os.kill(pid, 0)`` is not used because its Windows behavior is not a safe,
    portable existence check.
    """

    pid = int(pid)
    if pid <= 0:
        return False

    if pid == os.getpid():
        return True

    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return False

        try:
            return process.status() != psutil.STATUS_ZOMBIE
        except (psutil.AccessDenied, AttributeError):
            return True

    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        # We cannot inspect it, but it exists. Refusing replacement is safer
        # than destroying a possibly live writer's segment.
        return True


def _creation_lock_path(name):
    """Return a short, deterministic lock-file path for a ring name."""

    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
    return os.path.join(
        tempfile.gettempdir(),
        "improv_frame_ring_{}.lock".format(digest),
    )


@contextmanager
def _creation_lock(name, timeout=10.0):
    """Serialize creation and stale replacement for one ring name.

    All FrameRing creators use this lock, so one creator cannot inspect an old
    segment while another creator replaces it under the same name.
    """

    lock_path = _creation_lock_path(name)
    lock_file = open(lock_path, "a+b")
    acquired = False

    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()

            deadline = time.monotonic() + float(timeout)

            while True:
                lock_file.seek(0)

                try:
                    msvcrt.locking(
                        lock_file.fileno(),
                        msvcrt.LK_NBLCK,
                        1,
                    )
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for creation lock for {!r}".format(
                                name
                            )
                        )
                    time.sleep(0.05)

        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            acquired = True

        yield

    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(
                    lock_file.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        lock_file.close()


def _unlink_if_stale(name):
    """Remove a valid ring whose recorded writer is no longer alive.

    The caller must hold ``_creation_lock(name)``.
    """

    segment, tracker_registered = _open_for_stale_check(name)
    tracker_needs_unregister = tracker_registered

    try:
        minimum_size = _HEADER_LEN * np.dtype(np.int64).itemsize

        if segment.size < minimum_size:
            raise FrameRingExistsError(
                "shared memory {!r} exists but is not a complete frame ring; "
                "refusing to delete it automatically".format(name)
            )

        header = np.ndarray(
            (_HEADER_LEN,),
            dtype=np.int64,
            buffer=segment.buf,
        )

        magic = int(header[_H_MAGIC])
        version = int(header[_H_VERSION])
        writer_pid = int(header[_H_WRITER_PID])

        # Release the exported NumPy view before closing the mmap.
        header = None

        if magic != MAGIC or version != VERSION:
            raise FrameRingExistsError(
                "shared memory {!r} exists but is not frame-ring version {}; "
                "refusing to delete it automatically".format(name, VERSION)
            )

        if _pid_is_alive(writer_pid):
            raise FrameRingInUseError(
                "frame ring {!r} already has live writer pid {}".format(
                    name,
                    writer_pid,
                )
            )

        # On Python 3.8-3.12, unlink() also unregisters the tracked segment.
        # On Python 3.13+, this handle was opened with track=False.
        segment.unlink()
        tracker_needs_unregister = False

    finally:
        if tracker_needs_unregister:
            try:
                _unregister_from_resource_tracker(segment)
            except Exception:
                pass

        segment.close()


class FrameRing:
    """A single-writer, multi-reader shared-memory frame ring."""

    def __init__(self, segment, owner=False):
        self._segment = segment
        self.owner = bool(owner)
        self.name = segment.name
        self._closed_locally = False

        minimum_size = _HEADER_LEN * np.dtype(np.int64).itemsize
        if segment.size < minimum_size:
            raise FrameRingError(
                "{}: segment is smaller than its header".format(self.name)
            )

        header = np.ndarray(
            (_HEADER_LEN,),
            dtype=np.int64,
            buffer=segment.buf,
        )

        if (
            int(header[_H_MAGIC]) != MAGIC
            or int(header[_H_VERSION]) != VERSION
        ):
            raise FrameRingError(
                "{}: incomplete ring or incompatible version".format(
                    self.name
                )
            )

        shape = tuple(
            int(header[index])
            for index in (
                _H_HEIGHT,
                _H_WIDTH,
                _H_CHANNELS,
            )
        )

        shape, n_slots = _validate_geometry(
            shape,
            int(header[_H_NSLOTS]),
        )

        offsets = _layout(shape, n_slots)

        # Windows may round named mappings up to an allocation page.
        if segment.size < offsets[-1]:
            raise FrameRingError(
                "{}: segment size {} is smaller than declared size {}".format(
                    self.name,
                    segment.size,
                    offsets[-1],
                )
            )

        self.shape = shape
        self.n_slots = n_slots
        self._header = header

        self._sequences = np.ndarray(
            (n_slots,),
            dtype=np.int64,
            buffer=segment.buf,
            offset=offsets[0],
        )

        self._frame_numbers = np.ndarray(
            (n_slots,),
            dtype=np.int64,
            buffer=segment.buf,
            offset=offsets[1],
        )

        self._times = np.ndarray(
            (n_slots, _TIME_COLS),
            dtype=np.float64,
            buffer=segment.buf,
            offset=offsets[2],
        )

        self._frames = np.ndarray(
            (n_slots,) + shape,
            dtype=np.uint8,
            buffer=segment.buf,
            offset=offsets[3],
        )

        self._next_sequence = int(header[_H_LATEST]) + 1

    @classmethod
    def create(
        cls,
        name,
        shape,
        n_slots,
        replace_stale=False,
    ):
        """Create a ring.

        Existing segments are never deleted merely because their name
        collides. If ``replace_stale`` is true, the existing segment must have
        a valid header and its writer PID must no longer exist.
        """

        _require_x86()
        shape, n_slots = _validate_geometry(shape, n_slots)
        total_size = _layout(shape, n_slots)[-1]

        with _creation_lock(name):
            try:
                segment = shared_memory.SharedMemory(
                    name=name,
                    create=True,
                    size=total_size,
                )

            except FileExistsError:
                if not replace_stale:
                    raise FrameRingExistsError(
                        "frame ring {!r} already exists; pass "
                        "replace_stale=True only when recovering from a "
                        "confirmed crashed run".format(name)
                    )

                _unlink_if_stale(name)

                try:
                    segment = shared_memory.SharedMemory(
                        name=name,
                        create=True,
                        size=total_size,
                    )
                except FileExistsError as error:
                    raise FrameRingExistsError(
                        "frame ring {!r} was recreated outside the "
                        "FrameRing creation lock".format(name)
                    ) from error

            try:
                offsets = _layout(shape, n_slots)

                header = np.ndarray(
                    (_HEADER_LEN,),
                    dtype=np.int64,
                    buffer=segment.buf,
                )
                header[:] = 0

                np.ndarray(
                    (n_slots,),
                    dtype=np.int64,
                    buffer=segment.buf,
                    offset=offsets[0],
                )[:] = -1

                np.ndarray(
                    (n_slots,),
                    dtype=np.int64,
                    buffer=segment.buf,
                    offset=offsets[1],
                )[:] = -1

                header[_H_NSLOTS] = n_slots
                (
                    header[_H_HEIGHT],
                    header[_H_WIDTH],
                    header[_H_CHANNELS],
                ) = shape

                header[_H_LATEST] = -1
                header[_H_WRITER_PID] = os.getpid()
                header[_H_VERSION] = VERSION

                # Publish the magic last so readers reject partial setup.
                header[_H_MAGIC] = MAGIC

                return cls(segment, owner=True)

            except Exception:
                header = None

                try:
                    segment.unlink()
                finally:
                    segment.close()

                raise

    @classmethod
    def attach(
        cls,
        name,
        timeout=10.0,
        expect_shape=None,
    ):
        """Wait for and attach to a ring created by its camera writer."""

        _require_x86()

        timeout = float(timeout)
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError(
                "timeout must be a finite, non-negative number"
            )

        if expect_shape is not None:
            expected_shape = tuple(int(value) for value in expect_shape)
        else:
            expected_shape = None

        deadline = time.monotonic() + timeout
        last_error = None

        while True:
            segment = None

            try:
                segment = _open_reader(name)
                ring = cls(segment, owner=False)

            except FileNotFoundError as error:
                last_error = str(error)

            except ValueError as error:
                # A reader may attach after shm_open but before the writer's
                # ftruncate/initialization has completed.
                last_error = str(error)
                error.__traceback__ = None

                if segment is not None:
                    segment.close()

            except FrameRingError as error:
                last_error = str(error)
                error.__traceback__ = None

                if segment is not None:
                    segment.close()

            else:
                if (
                    expected_shape is not None
                    and expected_shape != ring.shape
                ):
                    ring.close()
                    raise ValueError(
                        "{}: ring holds {}, expected {}".format(
                            name,
                            ring.shape,
                            expected_shape,
                        )
                    )

                return ring

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "frame ring {!r} did not become ready within {} s: "
                    "{}".format(name, timeout, last_error)
                )

            time.sleep(0.05)

    @property
    def writer_pid(self):
        return int(self._header[_H_WRITER_PID])

    @property
    def closed(self):
        return bool(self._header[_H_CLOSED])

    def latest_sequence(self):
        """Return the newest published sequence, or -1 before the first frame."""

        return int(self._header[_H_LATEST])

    def oldest_available(self):
        """Return the oldest sequence that may still be available."""

        return max(
            0,
            self.latest_sequence() - self.n_slots + 1,
        )

    def write(
        self,
        frame,
        frame_number,
        capture_time=None,
        camera_start=None,
    ):
        """Publish one uint8 frame and return its sequence number."""

        if not self.owner:
            raise FrameRingError(
                "only the ring owner may publish frames"
            )

        if self.closed:
            raise FrameRingError(
                "cannot publish to a ring marked closed"
            )

        if isinstance(frame, np.ndarray):
            if frame.shape != self.shape:
                raise ValueError(
                    "{}: frame shape {} does not match {}".format(
                        self.name,
                        frame.shape,
                        self.shape,
                    )
                )

            if frame.dtype != np.uint8:
                raise TypeError(
                    "{}: frame dtype must be uint8".format(self.name)
                )

            source = frame

        else:
            source = np.frombuffer(frame, dtype=np.uint8)
            expected_size = int(np.prod(self.shape))

            if source.size != expected_size:
                raise ValueError(
                    "{}: frame has {} bytes, expected {}".format(
                        self.name,
                        source.size,
                        expected_size,
                    )
                )

        sequence = self._next_sequence
        slot = sequence % self.n_slots

        # Claim the slot before changing pixels or metadata.
        self._sequences[slot] = -1

        destination = self._frames[slot]

        if (
            isinstance(source, np.ndarray)
            and source.shape == destination.shape
        ):
            np.copyto(
                destination,
                source,
                casting="no",
            )
        else:
            np.copyto(
                destination.reshape(-1),
                source,
                casting="no",
            )

        self._frame_numbers[slot] = int(frame_number)

        times = self._times[slot]
        times[_T_CAPTURE] = (
            float("nan")
            if capture_time is None
            else float(capture_time)
        )
        times[_T_CAMERA_START] = (
            float("nan")
            if camera_start is None
            else float(camera_start)
        )
        times[_T_WRITE_DONE] = time.time()

        # Publish the completed slot before announcing it as latest.
        self._sequences[slot] = sequence
        self._header[_H_LATEST] = sequence
        self._next_sequence = sequence + 1

        return sequence

    def metadata(self, sequence):
        """Return stable metadata, or None if the sequence is unavailable."""

        sequence = int(sequence)
        if sequence < 0:
            return None

        slot = sequence % self.n_slots

        if int(self._sequences[slot]) != sequence:
            return None

        frame_number = int(self._frame_numbers[slot])
        times = self._times[slot].copy()

        if int(self._sequences[slot]) != sequence:
            return None

        return (
            frame_number,
            float(times[_T_CAPTURE]),
            float(times[_T_CAMERA_START]),
            float(times[_T_WRITE_DONE]),
        )

    def read(self, sequence, out=None):
        """Copy a stable frame and metadata.

        Returns:
            ``(frame, metadata)`` or ``None`` when the requested sequence has
            not arrived or has already been overwritten.
        """

        sequence = int(sequence)
        if sequence < 0:
            return None

        slot = sequence % self.n_slots

        if int(self._sequences[slot]) != sequence:
            return None

        if out is None:
            out = np.empty(
                self.shape,
                dtype=np.uint8,
            )
        elif out.shape != self.shape or out.dtype != np.uint8:
            raise ValueError(
                "out must be a uint8 array with shape {}".format(
                    self.shape
                )
            )

        np.copyto(
            out,
            self._frames[slot],
            casting="no",
        )

        frame_number = int(self._frame_numbers[slot])
        times = self._times[slot].copy()

        # Reject a frame overwritten during the copy.
        if int(self._sequences[slot]) != sequence:
            return None

        metadata = (
            frame_number,
            float(times[_T_CAPTURE]),
            float(times[_T_CAMERA_START]),
            float(times[_T_WRITE_DONE]),
        )

        return out, metadata

    def read_latest(self, out=None):
        """Copy the newest complete frame.

        Returns:
            ``(sequence, frame, metadata)`` or ``None``.
        """

        sequence = self.latest_sequence()
        result = self.read(sequence, out=out)

        if result is None:
            return None

        frame, metadata = result
        return sequence, frame, metadata

    def mark_closed(self):
        """Tell readers that the writer will publish no more frames."""

        if not self.owner:
            raise FrameRingError(
                "only the ring owner may mark it closed"
            )

        self._header[_H_CLOSED] = 1

    def close(self):
        """Release this process's mapping without unlinking the name."""

        if self._closed_locally:
            return

        self._header = None
        self._sequences = None
        self._frame_numbers = None
        self._times = None
        self._frames = None

        self._segment.close()
        self._closed_locally = True

    def unlink(self):
        """Remove the name after every consumer has stopped."""

        if not self.owner:
            raise FrameRingError(
                "only the ring owner may unlink it"
            )

        try:
            self._segment.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


class RingFollower:
    """Consume available frames in order and count overwritten frames."""

    def __init__(self, ring, start="next"):
        if start not in ("next", "oldest"):
            raise ValueError(
                "start must be 'next' or 'oldest'"
            )

        self.ring = ring
        latest = ring.latest_sequence()

        if start == "next":
            self.next_sequence = latest + 1
        else:
            self.next_sequence = ring.oldest_available()

        self.lost = 0
        self.read_count = 0

    def poll(self, max_frames=None):
        """Return newly available frames from oldest to newest."""

        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames < 0:
                raise ValueError(
                    "max_frames must be non-negative or None"
                )

        output = []
        latest = self.ring.latest_sequence()
        oldest = max(
            0,
            latest - self.ring.n_slots + 1,
        )

        if self.next_sequence < oldest:
            self.lost += oldest - self.next_sequence
            self.next_sequence = oldest

        while (
            self.next_sequence <= latest
            and (
                max_frames is None
                or len(output) < max_frames
            )
        ):
            sequence = self.next_sequence
            self.next_sequence += 1

            result = self.ring.read(sequence)

            if result is None:
                self.lost += 1
                continue

            frame, metadata = result
            output.append(
                (sequence, frame, metadata)
            )
            self.read_count += 1

        return output