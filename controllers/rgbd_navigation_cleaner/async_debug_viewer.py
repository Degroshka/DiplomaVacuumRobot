"""Async OpenCV debug-window process for the Webots RGB-D controller.

The Webots controller must remain the only process that talks to Robot(),
Camera, RangeFinder, motors and bumpers.  This helper only displays already
rendered numpy images and returns keyboard codes to the controller.  It removes
cv2.imshow()/waitKey() and GUI event handling from the real-time control loop.
"""

from __future__ import annotations

import atexit
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

FrameWindow = Tuple[str, object, Optional[Tuple[int, int]]]


class AsyncDebugViewerClient:
    """Non-blocking client used by the Webots controller.

    The controller thread never writes large images to the OS pipe directly.
    It only replaces the latest pending frame packet in a queue of size 1.  A
    small writer thread serializes that latest packet to the viewer subprocess.
    If the viewer is slower than the controller, old debug frames are dropped.
    """

    def __init__(self, script_path: str | os.PathLike[str], enabled: bool = True):
        self.enabled = bool(enabled)
        self.script_path = str(script_path)
        self.proc: Optional[subprocess.Popen] = None
        self._frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self._key_queue: "queue.Queue[int]" = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._writer: Optional[threading.Thread] = None
        self._reader: Optional[threading.Thread] = None
        if self.enabled:
            self.start()
            atexit.register(self.close)

    def start(self) -> bool:
        if not self.enabled:
            return False
        if self.proc is not None and self.proc.poll() is None:
            return True
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", self.script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except Exception as exc:
            print(f"Async debug viewer disabled: failed to start ({exc})")
            self.enabled = False
            self.proc = None
            return False
        self._stop.clear()
        self._writer = threading.Thread(target=self._writer_loop, name="debug-view-writer", daemon=True)
        self._reader = threading.Thread(target=self._reader_loop, name="debug-view-reader", daemon=True)
        self._writer.start()
        self._reader.start()
        return True

    def alive(self) -> bool:
        return bool(self.enabled and self.proc is not None and self.proc.poll() is None)

    def submit(self, windows: Iterable[FrameWindow]) -> bool:
        """Submit the latest set of rendered windows without blocking control."""
        if not self.alive():
            return False
        packet = {"cmd": "frames", "windows": list(windows)}
        try:
            self._frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(packet)
            except queue.Full:
                return False
        return True

    def poll_keys(self) -> List[int]:
        keys: List[int] = []
        while True:
            try:
                keys.append(int(self._key_queue.get_nowait()))
            except queue.Empty:
                return keys

    def close(self) -> None:
        if not self.enabled and self.proc is None:
            return
        self._stop.set()
        try:
            if self.alive():
                self._send_packet({"cmd": "close"})
        except Exception:
            pass
        try:
            if self.proc is not None and self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        self.enabled = False

    def _send_packet(self, packet: object) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        data = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        self.proc.stdin.write(struct.pack("<I", len(data)))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self._frame_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            try:
                self._send_packet(packet)
            except Exception:
                self.enabled = False
                return

    def _reader_loop(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        while not self._stop.is_set():
            try:
                line = self.proc.stdout.readline()
            except Exception:
                return
            if not line:
                return
            try:
                text = line.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if text.startswith("KEY "):
                try:
                    key = int(text.split(maxsplit=1)[1])
                    try:
                        self._key_queue.put_nowait(key)
                    except queue.Full:
                        pass
                except Exception:
                    pass
            elif text:
                print("debug-viewer:", text)


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            raise EOFError
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _viewer_main() -> int:
    try:
        import cv2
    except Exception as exc:
        print(f"viewer import error: {exc}", flush=True)
        return 2

    created = set()
    sizes = {}
    try:
        while True:
            header = sys.stdin.buffer.read(4)
            if not header:
                break
            if len(header) < 4:
                break
            (size,) = struct.unpack("<I", header)
            packet = pickle.loads(_read_exact(sys.stdin.buffer, size))
            cmd = packet.get("cmd") if isinstance(packet, dict) else None
            if cmd == "close":
                cv2.destroyAllWindows()
                break
            if cmd != "frames":
                continue
            active_names = set()
            for item in packet.get("windows", []):
                if len(item) == 2:
                    name, image = item
                    desired_size = None
                else:
                    name, image, desired_size = item
                name = str(name)
                active_names.add(name)
                if name not in created:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    created.add(name)
                if desired_size is not None and sizes.get(name) != tuple(desired_size):
                    try:
                        cv2.resizeWindow(name, int(desired_size[0]), int(desired_size[1]))
                        sizes[name] = tuple(desired_size)
                    except Exception:
                        pass
                cv2.imshow(name, image)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                print(f"KEY {int(key)}", flush=True)
    except EOFError:
        pass
    except Exception as exc:
        print(f"viewer error: {exc}", flush=True)
        return 1
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_viewer_main())
