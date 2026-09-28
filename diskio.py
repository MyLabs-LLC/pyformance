"""Unbuffered Windows disk reads and writes.

FILE_FLAG_NO_BUFFERING keeps the filesystem cache out of the measurement.
If the device rejects that mode, callers can fall back to buffered I/O.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
CREATE_ALWAYS = 2
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_WRITE_THROUGH = 0x80000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
FILE_BEGIN = 0

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.WriteFile.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p,
]
_kernel32.WriteFile.restype = wintypes.BOOL
_kernel32.ReadFile.argtypes = [
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p,
]
_kernel32.ReadFile.restype = wintypes.BOOL
_kernel32.SetFilePointerEx.argtypes = [
    wintypes.HANDLE,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_longlong),
    wintypes.DWORD,
]
_kernel32.SetFilePointerEx.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
_kernel32.FlushFileBuffers.restype = wintypes.BOOL

_INVALID = wintypes.HANDLE(-1).value


class DiskError(OSError):
    pass


def _fail(path: str) -> None:
    err = ctypes.get_last_error()
    raise DiskError(err, ctypes.FormatError(err), path)


def _open(path: str, access: int, creation: int, flags: int) -> int:
    handle = _kernel32.CreateFileW(path, access, 0, None, creation, flags, None)
    if handle == _INVALID:
        _fail(path)
    return handle


def _close(handle: int) -> None:
    _kernel32.CloseHandle(handle)


def aligned_region(size: int, alignment: int = 4096) -> tuple[ctypes.Array, int]:
    raw = ctypes.create_string_buffer(size + alignment)
    base = ctypes.addressof(raw)
    aligned = (base + alignment - 1) & ~(alignment - 1)
    return raw, aligned


def fill_pattern(ptr: int, size: int) -> None:
    blob = os.urandom(size)
    ctypes.memmove(ptr, blob, size)


def write_bytes(path: str, ptr: int, block: int, blocks: int, unbuffered: bool) -> None:
    flags = FILE_ATTRIBUTE_NORMAL
    if unbuffered:
        flags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
    handle = _open(path, GENERIC_WRITE, CREATE_ALWAYS, flags)
    try:
        written = wintypes.DWORD(0)
        for _ in range(blocks):
            ok = _kernel32.WriteFile(handle, ctypes.c_void_p(ptr), block, ctypes.byref(written), None)
            if not ok or written.value != block:
                _fail(path)
        if not _kernel32.FlushFileBuffers(handle):
            _fail(path)
    finally:
        _close(handle)


def read_bytes(path: str, ptr: int, block: int, blocks: int, unbuffered: bool) -> None:
    flags = FILE_ATTRIBUTE_NORMAL
    if unbuffered:
        flags |= FILE_FLAG_NO_BUFFERING
    handle = _open(path, GENERIC_READ, OPEN_EXISTING, flags)
    try:
        got = wintypes.DWORD(0)
        for _ in range(blocks):
            ok = _kernel32.ReadFile(handle, ctypes.c_void_p(ptr), block, ctypes.byref(got), None)
            if not ok or got.value != block:
                _fail(path)
    finally:
        _close(handle)


def random_reads(path: str, ptr: int, file_size: int, block: int, ops: int, unbuffered: bool, seed: int = 1) -> int:
    flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS
    if unbuffered:
        flags |= FILE_FLAG_NO_BUFFERING
    handle = _open(path, GENERIC_READ, OPEN_EXISTING, flags)
    try:
        span = file_size // block
        state = seed & 0x7FFFFFFF
        got = wintypes.DWORD(0)
        new_pos = ctypes.c_longlong(0)
        completed = 0
        for _ in range(ops):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            offset = (state % span) * block
            if not _kernel32.SetFilePointerEx(handle, offset, ctypes.byref(new_pos), FILE_BEGIN):
                _fail(path)
            ok = _kernel32.ReadFile(handle, ctypes.c_void_p(ptr), block, ctypes.byref(got), None)
            if not ok or got.value != block:
                _fail(path)
            completed += 1
        return completed
    finally:
        _close(handle)
