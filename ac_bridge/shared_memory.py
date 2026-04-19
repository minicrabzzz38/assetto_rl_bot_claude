"""
Reads Assetto Corsa telemetry from Windows shared memory.

AC exposes three named shared memory pages:
  Local\acpmf_physics  — real-time physics (speed, inputs, tyres, damage...)
  Local\acpmf_graphics — session data (lap time, position, valid lap flag...)
  Local\acpmf_static   — static car/track info (loaded once per session)

We open these with OpenFileMappingW + MapViewOfFile via ctypes kernel32,
which is the correct way to attach to an existing (not create new) mapping.
"""

import ctypes
import ctypes.wintypes
import logging
import platform
import time
from typing import Optional, Tuple

from .data_types import (
    SPageFilePhysics,
    SPageFileGraphics,
    SPageFileStatic,
    ACStatus,
)

logger = logging.getLogger(__name__)

# Windows constants
FILE_MAP_READ = 0x0004


class _SharedPage:
    """Wraps a single named shared memory page."""

    def __init__(self, name: str, structure_type):
        self._name = name
        self._type = structure_type
        self._size = ctypes.sizeof(structure_type)
        self._handle = None
        self._ptr = None
        self._view = None

    def connect(self) -> bool:
        if platform.system() != "Windows":
            logger.error("Shared memory is only available on Windows (AC is Windows-only)")
            return False
        try:
            k32 = ctypes.windll.kernel32
            handle = k32.OpenFileMappingW(FILE_MAP_READ, False, self._name)
            if not handle:
                err = k32.GetLastError()
                logger.debug("OpenFileMappingW(%s) failed: error %d", self._name, err)
                return False
            ptr = k32.MapViewOfFile(handle, FILE_MAP_READ, 0, 0, self._size)
            if not ptr:
                k32.CloseHandle(handle)
                logger.debug("MapViewOfFile(%s) failed", self._name)
                return False
            self._handle = handle
            self._ptr = ptr
            self._view = self._type.from_address(ptr)
            logger.debug("Connected to shared memory: %s", self._name)
            return True
        except Exception as exc:
            logger.warning("Exception connecting to %s: %s", self._name, exc)
            return False

    def read(self) -> Optional[object]:
        """Return a fresh copy of the structure (thread-safe snapshot)."""
        if self._view is None:
            return None
        try:
            raw = ctypes.string_at(self._ptr, self._size)
            return self._type.from_buffer_copy(raw)
        except Exception as exc:
            logger.debug("Read error on %s: %s", self._name, exc)
            return None

    def disconnect(self):
        if self._ptr:
            ctypes.windll.kernel32.UnmapViewOfFile(self._ptr)
            self._ptr = None
            self._view = None
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None

    @property
    def is_connected(self) -> bool:
        return self._view is not None


class ACSharedMemoryReader:
    """
    High-level reader for all three AC shared memory pages.

    Usage:
        reader = ACSharedMemoryReader()
        if reader.connect():
            phys, gfx, static = reader.read_all()
    """

    PHYSICS_NAME = "Local\\acpmf_physics"
    GRAPHICS_NAME = "Local\\acpmf_graphics"
    STATIC_NAME = "Local\\acpmf_static"

    CONNECT_TIMEOUT = 30.0   # seconds to wait for AC to start
    CONNECT_POLL    = 1.0    # seconds between attempts

    def __init__(self):
        self._physics  = _SharedPage(self.PHYSICS_NAME,  SPageFilePhysics)
        self._graphics = _SharedPage(self.GRAPHICS_NAME, SPageFileGraphics)
        self._static   = _SharedPage(self.STATIC_NAME,   SPageFileStatic)
        self._connected = False

    def connect(self, timeout: Optional[float] = None) -> bool:
        """
        Attempt to connect to AC shared memory.
        If timeout is given, poll until AC starts or timeout expires.
        """
        deadline = time.time() + (timeout or self.CONNECT_TIMEOUT)
        attempt = 0
        while True:
            attempt += 1
            ok_phys = self._physics.connect()
            ok_gfx  = self._graphics.connect()
            ok_stat = self._static.connect()

            if ok_phys and ok_gfx and ok_stat:
                self._connected = True
                logger.info("Connected to Assetto Corsa shared memory (attempt %d)", attempt)
                return True

            if time.time() >= deadline:
                logger.error(
                    "Could not connect to AC shared memory after %.0f s. "
                    "Is Assetto Corsa running with the shared memory plugin enabled?",
                    timeout or self.CONNECT_TIMEOUT,
                )
                return False

            logger.info(
                "AC not found yet (attempt %d) — waiting %.0f s ...",
                attempt, self.CONNECT_POLL,
            )
            time.sleep(self.CONNECT_POLL)
            # Disconnect any partial opens before retry
            self._physics.disconnect()
            self._graphics.disconnect()
            self._static.disconnect()

    def read_all(self) -> Tuple[
        Optional[SPageFilePhysics],
        Optional[SPageFileGraphics],
        Optional[SPageFileStatic],
    ]:
        """Return snapshots of all three pages."""
        return (
            self._physics.read(),
            self._graphics.read(),
            self._static.read(),
        )

    def read_physics(self) -> Optional[SPageFilePhysics]:
        return self._physics.read()

    def read_graphics(self) -> Optional[SPageFileGraphics]:
        return self._graphics.read()

    def read_static(self) -> Optional[SPageFileStatic]:
        return self._static.read()

    def is_game_live(self) -> bool:
        """Returns True only when AC is in an active driving session."""
        gfx = self._graphics.read()
        if gfx is None:
            return False
        return gfx.status == ACStatus.LIVE

    def wait_for_live(self, timeout: float = 60.0) -> bool:
        """Block until AC enters LIVE status or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_game_live():
                return True
            time.sleep(0.5)
        return False

    def disconnect(self):
        self._physics.disconnect()
        self._graphics.disconnect()
        self._static.disconnect()
        self._connected = False
        logger.info("Disconnected from AC shared memory")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()
