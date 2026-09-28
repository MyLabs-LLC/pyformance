"""Workloads used by the benchmark.

Each function returns a checksum so the work stays live for the timed run.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import time
import zlib
from array import array


_SHA_BLOCK = bytes((i * 17 + 31) & 0xFF for i in range(1 << 20))
_COMPRESS_BLOCK = bytes((i * 13) & 0xFF if i % 5 else (i & 0x0F) for i in range(1 << 20))


def integer_kernel(iterations: int) -> int:
    x = 0x123456789ABCDEF
    acc = 0
    for _ in range(iterations):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFFFFFFFFFF
        acc = (acc + (x ^ (x >> 17))) & 0xFFFFFFFFFFFFFFFF
        acc ^= (acc << 7) & 0xFFFFFFFFFFFFFFFF
    return acc


def float_kernel(iterations: int) -> float:
    x = 0.5
    acc = 0.0
    for _ in range(iterations):
        x = math.fmod(x * 1.0000007 + 0.173, 3.5)
        acc += math.sqrt(x * x + 1.25)
    return acc


def count_primes_below(limit: int) -> int:
    if limit < 2:
        return 0
    flags = bytearray(b"\x01") * limit
    flags[0:2] = b"\x00\x00"
    root = int(limit**0.5) + 1
    for i in range(2, root):
        if flags[i]:
            start = i * i
            count = (limit - start + i - 1) // i
            flags[start:limit:i] = bytes(count)
    return flags.count(1)


def prime_kernel(spec: tuple[int, int]) -> int:
    limit, repeats = spec
    total = 0
    for _ in range(repeats):
        total += count_primes_below(limit)
    return total


def sort_kernel(spec: tuple[int, int]) -> int:
    count, repeats = spec
    x = 2_463_534_242
    checksum = 0
    for _ in range(repeats):
        values = [0] * count
        for i in range(count):
            x = (x * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
            values[i] = x
        values.sort()
        checksum ^= values[0] ^ values[-1] ^ values[count // 2]
    return checksum


def sha256_kernel(megabytes: int) -> int:
    digest = hashlib.sha256()
    block = _SHA_BLOCK
    for _ in range(megabytes):
        digest.update(block)
    return digest.digest()[0]


def compress_kernel(megabytes: int) -> int:
    block = _COMPRESS_BLOCK
    total = 0
    for _ in range(megabytes):
        total += len(zlib.compress(block, 1))
    return total


def physics_kernel(spec: tuple[int, int]) -> float:
    bodies, steps = spec
    xs = [0.0] * bodies
    ys = [0.0] * bodies
    vxs = [0.0] * bodies
    vys = [0.0] * bodies
    for i in range(bodies):
        xs[i] = (i % 100) * 0.1
        ys[i] = ((i * 3) % 100) * 0.1
        vxs[i] = ((i % 7) - 3) * 0.02
        vys[i] = ((i % 5) - 2) * 0.02
    acc = 0.0
    for _ in range(steps):
        for i in range(bodies):
            x = xs[i]
            y = ys[i]
            dx = 5.0 - x
            dy = 5.0 - y
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + 0.25)
            vx = (vxs[i] + dx * inv * 0.01) * 0.999
            vy = (vys[i] + dy * inv * 0.01) * 0.999
            vxs[i] = vx
            vys[i] = vy
            xs[i] = x + vx
            ys[i] = y + vy
        acc += xs[0]
    return acc


def build_chase_table(size: int = 1 << 22) -> array:
    """Full-period index map. Each step jumps about 32 KB."""
    table = array("Q", [0]) * size
    mask = size - 1
    for i in range(size):
        table[i] = (i * 4097 + 17) & mask
    return table


def chase_kernel(spec: tuple[array, int]) -> int:
    table, steps = spec
    index = 0
    for _ in range(steps):
        index = table[index]
    return index


def _memset(buf: bytearray, value: int) -> None:
    raw = (ctypes.c_char * len(buf)).from_buffer(buf)
    ctypes.memset(raw, value, len(buf))


def memory_copy_worker(spec: tuple[int, int]) -> tuple[float, int]:
    size, repeats = spec
    source = bytearray(size)
    dest = bytearray(size)
    _memset(source, 0xA5)
    started = time.perf_counter()
    for _ in range(repeats):
        dest[:] = source
    return time.perf_counter() - started, dest[0]


def memory_read_worker(spec: tuple[int, int]) -> tuple[float, int]:
    size, repeats = spec
    buf = bytearray(size)
    _memset(buf, 0xA5)
    found = 0
    started = time.perf_counter()
    for _ in range(repeats):
        found += buf.count(0)
    return time.perf_counter() - started, found


def memory_write_worker(spec: tuple[int, int]) -> tuple[float, int]:
    size, repeats = spec
    buf = bytearray(size)
    started = time.perf_counter()
    for i in range(repeats):
        _memset(buf, 0xA5 if i % 2 == 0 else 0x5A)
    return time.perf_counter() - started, buf[0]
