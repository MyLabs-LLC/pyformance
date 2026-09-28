"""Run the benchmark and turn measurements into comparable scores.

A score of 100 means the result matched the built-in reference rate.
The overall score is the geometric mean of the CPU, memory, disk, and GPU
suite scores, so each suite counts equally.
"""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import platform
import shutil
import tempfile
import threading
import time
import winreg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

from pyformance import __version__
from pyformance.diskio import (
    DiskError,
    aligned_region,
    fill_pattern,
    random_reads,
    read_bytes,
    write_bytes,
)
from pyformance.gpu import describe_gpu, measure_gpu, verify_gpu
from pyformance.kernels import (
    build_chase_table,
    chase_kernel,
    compress_kernel,
    count_primes_below,
    float_kernel,
    integer_kernel,
    memory_copy_worker,
    memory_read_worker,
    memory_write_worker,
    physics_kernel,
    prime_kernel,
    sha256_kernel,
    sort_kernel,
)

# Reference rates. Score = 100 * measured / reference.
_BASELINES = {
    "integer": 40.0,  # Mops/s across all workers
    "float": 60.0,
    "primes": 800.0,  # million candidates / s
    "sort": 12.0,  # million elements / s
    "sha256": 8000.0,  # MB/s, all threads
    "compress": 2000.0,
    "physics": 15.0,  # million body-steps / s
    "single": 5.0,  # Mops/s, one thread
    "mem_copy": 8000.0,
    "mem_read": 3000.0,
    "mem_write": 8000.0,
    "chase": 20.0,  # million dependent lookups / s
    "disk_write": 500.0,
    "disk_read": 800.0,
    "disk_rand": 3000.0,  # 4 KB IOPS
    "gpu_fp32": 10000.0,  # GFLOPS
    "gpu_fp16": 5000.0,
    "gpu_fp64": 100.0,
    "gpu_sfu": 4000.0,  # Gops/s
    "gpu_int": 10000.0,
    "gpu_int64": 800.0,
    "gpu_bit": 20000.0,
    "gpu_bw": 150.0,  # GB/s
    "gpu_read": 200.0,
    "gpu_write": 150.0,
    "gpu_l2": 800.0,
    "gpu_shared": 8000.0,
    "gpu_stride": 2000.0,  # Mlookups/s
}

ProgressFn = Callable[[int, int, str], None]


# id, suite, display name. Order is the run order.
TESTS: tuple[tuple[str, str, str], ...] = (
    ("integer", "cpu", "Integer math"),
    ("float", "cpu", "Floating-point math"),
    ("primes", "cpu", "Prime search"),
    ("sort", "cpu", "Sorting"),
    ("sha256", "cpu", "SHA-256"),
    ("compress", "cpu", "Compression"),
    ("physics", "cpu", "Particle physics"),
    ("single", "cpu", "Single thread"),
    ("mem_copy", "memory", "Copy bandwidth"),
    ("mem_read", "memory", "Read bandwidth"),
    ("mem_write", "memory", "Write bandwidth"),
    ("chase", "memory", "Pointer chase"),
    ("disk_write", "disk", "Sequential write"),
    ("disk_read", "disk", "Sequential read"),
    ("disk_rand", "disk", "Random 4 KB read"),
    ("gpu_fp32", "gpu", "FP32 compute"),
    ("gpu_fp16", "gpu", "FP16 compute"),
    ("gpu_fp64", "gpu", "FP64 compute"),
    ("gpu_sfu", "gpu", "Special functions"),
    ("gpu_int", "gpu", "Integer compute"),
    ("gpu_int64", "gpu", "64-bit integer"),
    ("gpu_bit", "gpu", "Bitwise"),
    ("gpu_bw", "gpu", "Global copy"),
    ("gpu_read", "gpu", "Global read"),
    ("gpu_write", "gpu", "Global write"),
    ("gpu_l2", "gpu", "L2 cache"),
    ("gpu_shared", "gpu", "Shared memory"),
    ("gpu_stride", "gpu", "Random access"),
)


@dataclass
class Settings:
    quick: bool = False
    suites: tuple[str, ...] = ("cpu", "memory", "disk", "gpu")
    tests: tuple[str, ...] | None = None
    workers: int | None = None

    @property
    def cpu_seconds(self) -> float:
        return 0.4 if self.quick else 1.6

    @property
    def memory_seconds(self) -> float:
        return 0.35 if self.quick else 1.35

    @property
    def disk_seconds(self) -> float:
        return 0.3 if self.quick else 1.15

    @property
    def disk_cap_bytes(self) -> int:
        mb = 128 if self.quick else 2048
        return mb << 20

    @property
    def disk_floor_bytes(self) -> int:
        mb = 32 if self.quick else 128
        return mb << 20

    @property
    def gpu_seconds(self) -> float:
        return 0.35 if self.quick else 1.25

    @property
    def worker_count(self) -> int:
        if self.workers:
            return max(1, self.workers)
        return max(1, os.cpu_count() or 1)


@dataclass
class TestResult:
    id: str
    suite: str
    name: str
    value: float | None
    unit: str
    score: float | None
    seconds: float
    detail: str = ""
    error: str | None = None

    def result_text(self) -> str:
        if self.error:
            return self.error
        if self.value is None:
            return ""
        return format_metric(self.value, self.unit)


@dataclass
class Report:
    system: dict
    results: list[TestResult] = field(default_factory=list)
    suite_scores: dict[str, float] = field(default_factory=dict)
    overall: float | None = None
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool": "PyFormance",
            "version": __version__,
            "system": self.system,
            "results": [asdict(item) for item in self.results],
            "suite_scores": self.suite_scores,
            "overall": self.overall,
            "elapsed_seconds": self.elapsed,
        }


def format_metric(value: float, unit: str) -> str:
    if unit == "IOPS":
        return f"{value:,.0f} IOPS"
    if unit == "MB/s":
        if value >= 100:
            return f"{value:,.0f} MB/s"
        return f"{value:,.1f} MB/s"
    if unit == "GB/s":
        if value >= 1000:
            return f"{value / 1000:.2f} TB/s"
        if value >= 100:
            return f"{value:,.0f} GB/s"
        return f"{value:,.1f} GB/s"
    if unit == "GFLOPS":
        if value >= 1000:
            return f"{value / 1000:.2f} TFLOPS"
        return f"{value:,.0f} GFLOPS"
    if unit == "Gops/s":
        if value >= 1000:
            return f"{value / 1000:.2f} Tops"
        return f"{value:,.0f} Gops/s"
    if value >= 100:
        return f"{value:,.0f} {unit}"
    return f"{value:,.1f} {unit}"


def geometric_mean(values: list[float]) -> float:
    usable = [value for value in values if value > 0]
    if not usable:
        return 0.0
    return math.exp(sum(math.log(value) for value in usable) / len(usable))


def score_value(test_id: str, value: float) -> float:
    baseline = _BASELINES[test_id]
    return 100.0 * value / baseline


SCORE_NOTE = (
    "A score of 100 matches the built-in reference for that test. "
    "200 is twice as fast. "
    "Each suite score is the geometric mean of its tests. "
    "Overall is the geometric mean of the suite scores, so each suite counts equally."
)


def format_result_line(item: TestResult) -> str:
    score = "    fail" if item.error or item.score is None else f"{item.score:8.0f}"
    return f"  {item.name:<22} {item.result_text():>18}  {score}"


def format_banner(info: dict) -> str:
    return f"PyFormance {__version__}\n{_hardware_line(info)}"


def system_info() -> dict:
    ram = _physical_ram()
    info = {
        "processor": _processor_name(),
        "logical_processors": os.cpu_count() or 1,
        "ram_bytes": ram,
        "ram": _format_bytes(ram) if ram else "unknown",
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    gpu = describe_gpu()
    if gpu:
        info["gpu"] = gpu
    return info


def _processor_name() -> str:
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(name).strip()
    except OSError:
        return platform.processor() or "Unknown CPU"


def _physical_ram() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return int(status.ullTotalPhys)


def _format_bytes(size: int) -> str:
    gib = size / (1024**3)
    if gib >= 1:
        return f"{gib:.1f} GB"
    return f"{size / (1024**2):.0f} MB"


class Benchmark:
    def __init__(self, settings: Settings, stop: threading.Event | None = None):
        self.settings = settings
        self.stop = stop or threading.Event()
        self.pool: ProcessPoolExecutor | None = None
        self._chase = None

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def run(
        self,
        progress: ProgressFn | None = None,
        on_result: Callable[[TestResult], None] | None = None,
        on_suite: Callable[[str, float], None] | None = None,
        on_start: Callable[[dict], None] | None = None,
    ) -> Report:
        started = time.perf_counter()
        info = system_info()
        if on_start:
            on_start(info)
        plan = self._plan()
        results: list[TestResult] = []
        active_suite: str | None = None
        try:
            index = 0
            while index < len(plan):
                if self.stop.is_set():
                    break
                test_id, suite, name = plan[index]
                if active_suite is not None and suite != active_suite:
                    _emit_suite(results, active_suite, on_suite)
                active_suite = suite
                if suite in {"disk", "gpu"}:
                    grouped = [item for item in plan if item[1] == suite]
                    results.extend(self._run_group(suite, grouped, progress, index, len(plan), on_result))
                    index += len(grouped)
                    continue
                if progress:
                    progress(index, len(plan), name)
                item = self._run_named(test_id, suite, name)
                results.append(item)
                if on_result:
                    on_result(item)
                index += 1
            if active_suite is not None:
                _emit_suite(results, active_suite, on_suite)
        finally:
            self.close()
        report = Report(system=info, results=results, elapsed=time.perf_counter() - started)
        report.suite_scores = _suite_scores(results)
        if report.suite_scores:
            report.overall = geometric_mean(list(report.suite_scores.values()))
        if progress:
            progress(len(plan), len(plan), "Done")
        return report

    def _plan(self) -> list[tuple[str, str, str]]:
        chosen = set(self.settings.tests) if self.settings.tests is not None else None
        return [
            item
            for item in TESTS
            if item[1] in self.settings.suites and (chosen is None or item[0] in chosen)
        ]

    def _run_named(self, test_id: str, suite: str, name: str) -> TestResult:
        started = time.perf_counter()
        try:
            value, unit, detail = self._dispatch(test_id)
            return TestResult(
                id=test_id,
                suite=suite,
                name=name,
                value=value,
                unit=unit,
                score=score_value(test_id, value),
                seconds=time.perf_counter() - started,
                detail=detail,
            )
        except Exception as exc:  # keep the rest of the suite running
            return TestResult(
                id=test_id,
                suite=suite,
                name=name,
                value=None,
                unit="",
                score=None,
                seconds=time.perf_counter() - started,
                error=str(exc),
            )

    def measure_one(self, test_id: str) -> tuple[float, str, str]:
        return self._dispatch(test_id)

    def _dispatch(self, test_id: str) -> tuple[float, str, str]:
        if test_id == "integer":
            return self._cpu_integer()
        if test_id == "float":
            return self._cpu_float()
        if test_id == "primes":
            return self._cpu_primes()
        if test_id == "sort":
            return self._cpu_sort()
        if test_id == "sha256":
            return self._cpu_sha()
        if test_id == "compress":
            return self._cpu_compress()
        if test_id == "physics":
            return self._cpu_physics()
        if test_id == "single":
            return self._cpu_single()
        if test_id == "mem_copy":
            return self._memory_copy()
        if test_id == "mem_read":
            return self._memory_read()
        if test_id == "mem_write":
            return self._memory_write()
        if test_id == "chase":
            return self._memory_chase()
        raise KeyError(test_id)

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.settings.worker_count)
        return self.pool

    def _warmup(self, fn, arg) -> None:
        pool = self._ensure_pool()
        workers = self.settings.worker_count
        list(pool.map(fn, [arg] * workers, chunksize=1))

    def _scale_process(self, fn, seed, target: float):
        """Grow `seed` until one worker takes ~target seconds. Returns the argument."""
        pool = self._ensure_pool()
        arg = seed
        sample = target
        for _ in range(8):
            if self.stop.is_set():
                break
            started = time.perf_counter()
            pool.submit(fn, arg).result()
            sample = max(time.perf_counter() - started, 1e-4)
            if sample >= 0.08:
                break
            arg = _scale_arg(arg, 2)
        factor = max(1, int(round(target / sample)))
        return _scale_arg(arg, factor)

    def _time_workers(self, fn, arg) -> tuple[float, int]:
        pool = self._ensure_pool()
        workers = self.settings.worker_count
        gc.disable()
        try:
            started = time.perf_counter()
            futures = [pool.submit(fn, arg) for _ in range(workers)]
            checksum = 0
            for future in futures:
                checksum += future.result()
            elapsed = max(time.perf_counter() - started, 1e-6)
        finally:
            gc.enable()
        return elapsed, checksum

    def _scale_local(self, fn, seed: int, target: float) -> int:
        amount = seed
        sample = target
        for _ in range(8):
            started = time.perf_counter()
            fn(amount)
            sample = max(time.perf_counter() - started, 1e-4)
            if sample >= 0.08:
                break
            amount *= 2
        return amount * max(1, int(round(target / sample)))

    def _time_threads(self, fn, arg: int, workers: int) -> float:
        gc.disable()
        try:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(fn, arg) for _ in range(workers)]
                for future in futures:
                    future.result()
            return max(time.perf_counter() - started, 1e-6)
        finally:
            gc.enable()

    def _cpu_integer(self) -> tuple[float, str, str]:
        self._warmup(integer_kernel, 20_000)
        iterations = self._scale_process(integer_kernel, 200_000, self.settings.cpu_seconds)
        elapsed, _checksum = self._time_workers(integer_kernel, iterations)
        rate = iterations * self.settings.worker_count / elapsed / 1e6
        return rate, "Mops/s", f"{self.settings.worker_count} workers"

    def _cpu_float(self) -> tuple[float, str, str]:
        self._warmup(float_kernel, 20_000)
        iterations = self._scale_process(float_kernel, 200_000, self.settings.cpu_seconds)
        elapsed, _checksum = self._time_workers(float_kernel, iterations)
        rate = iterations * self.settings.worker_count / elapsed / 1e6
        return rate, "Mops/s", f"{self.settings.worker_count} workers"

    def _cpu_primes(self) -> tuple[float, str, str]:
        limit = 1_000_000 if self.settings.quick else 2_000_000
        self._warmup(prime_kernel, (limit, 1))
        spec = self._scale_process(prime_kernel, (limit, 1), self.settings.cpu_seconds)
        elapsed, _checksum = self._time_workers(prime_kernel, spec)
        candidates = spec[0] * spec[1] * self.settings.worker_count
        rate = candidates / elapsed / 1e6
        return rate, "Mcand/s", f"primes below {spec[0]:,}"

    def _cpu_sort(self) -> tuple[float, str, str]:
        count = 40_000 if self.settings.quick else 80_000
        self._warmup(sort_kernel, (count, 1))
        spec = self._scale_process(sort_kernel, (count, 1), self.settings.cpu_seconds)
        elapsed, _checksum = self._time_workers(sort_kernel, spec)
        elements = spec[0] * spec[1] * self.settings.worker_count
        rate = elements / elapsed / 1e6
        return rate, "Melem/s", f"{spec[0]:,} keys"

    def _cpu_sha(self) -> tuple[float, str, str]:
        workers = self.settings.worker_count
        megabytes = self._scale_local(sha256_kernel, 4, self.settings.cpu_seconds)
        elapsed = self._time_threads(sha256_kernel, megabytes, workers)
        rate = megabytes * workers / elapsed
        return rate, "MB/s", f"{workers} threads"

    def _cpu_compress(self) -> tuple[float, str, str]:
        workers = self.settings.worker_count
        megabytes = self._scale_local(compress_kernel, 2, self.settings.cpu_seconds)
        elapsed = self._time_threads(compress_kernel, megabytes, workers)
        rate = megabytes * workers / elapsed
        return rate, "MB/s", "zlib level 1"

    def _cpu_physics(self) -> tuple[float, str, str]:
        bodies = 2_000 if self.settings.quick else 3_000
        self._warmup(physics_kernel, (bodies, 1))
        spec = self._scale_process(physics_kernel, (bodies, 1), self.settings.cpu_seconds)
        elapsed, _checksum = self._time_workers(physics_kernel, spec)
        steps = spec[0] * spec[1] * self.settings.worker_count
        rate = steps / elapsed / 1e6
        return rate, "Msteps/s", f"{spec[0]:,} bodies"

    def _cpu_single(self) -> tuple[float, str, str]:
        iterations = self._scale_local(integer_kernel, 200_000, self.settings.cpu_seconds)
        gc.disable()
        try:
            started = time.perf_counter()
            integer_kernel(iterations)
            elapsed = max(time.perf_counter() - started, 1e-6)
        finally:
            gc.enable()
        return iterations / elapsed / 1e6, "Mops/s", "1 worker"

    def _memory_streams(self) -> int:
        return 2 if self.settings.quick else min(4, self.settings.worker_count)

    def _memory_buf_size(self) -> int:
        return 32 << 20 if self.settings.quick else 96 << 20

    def _scale_memory(self, fn, size: int) -> int:
        self._ensure_pool()
        repeats = 1
        inner = self.settings.memory_seconds
        for _ in range(8):
            inner, _checksum = self.pool.submit(fn, (size, repeats)).result()
            if inner >= 0.08:
                break
            repeats *= 2
        factor = max(1, int(round(self.settings.memory_seconds / max(inner, 1e-4))))
        return repeats * factor

    def _run_memory(self, fn, size: int, repeats: int, streams: int) -> float:
        pool = self._ensure_pool()
        futures = [pool.submit(fn, (size, repeats)) for _ in range(streams)]
        samples = []
        for future in futures:
            inner, _checksum = future.result()
            samples.append(inner)
        elapsed = max(samples)
        return (size * repeats * streams) / max(elapsed, 1e-6) / 1e6

    def _memory_copy(self) -> tuple[float, str, str]:
        streams = self._memory_streams()
        size = self._memory_buf_size()
        repeats = self._scale_memory(memory_copy_worker, size)
        rate = self._run_memory(memory_copy_worker, size, repeats, streams)
        return rate, "MB/s", f"{streams} streams, {_format_bytes(size)} each"

    def _memory_read(self) -> tuple[float, str, str]:
        streams = self._memory_streams()
        size = self._memory_buf_size()
        repeats = self._scale_memory(memory_read_worker, size)
        rate = self._run_memory(memory_read_worker, size, repeats, streams)
        return rate, "MB/s", f"{streams} streams"

    def _memory_write(self) -> tuple[float, str, str]:
        streams = self._memory_streams()
        size = self._memory_buf_size()
        repeats = self._scale_memory(memory_write_worker, size)
        rate = self._run_memory(memory_write_worker, size, repeats, streams)
        return rate, "MB/s", f"{streams} streams"

    def _memory_chase(self) -> tuple[float, str, str]:
        if self._chase is None:
            self._chase = build_chase_table(1 << 20 if self.settings.quick else 1 << 22)
        table = self._chase
        steps = self._scale_local(lambda n: chase_kernel((table, n)), 50_000, self.settings.memory_seconds)
        gc.disable()
        try:
            started = time.perf_counter()
            chase_kernel((table, steps))
            elapsed = max(time.perf_counter() - started, 1e-6)
        finally:
            gc.enable()
        return steps / elapsed / 1e6, "Mlookups/s", f"{_format_bytes(len(table) * 8)} table"

    def _run_group(self, suite, items, progress, index: int, total: int, on_result) -> list[TestResult]:
        if suite == "disk":
            return self._run_disk(items, progress, index, total, on_result)
        return self._run_gpu(items, progress, index, total, on_result)

    def _run_gpu(self, items, progress, index: int, total: int, on_result) -> list[TestResult]:
        labels = {test_id: name for test_id, suite, name in TESTS if suite == "gpu"}
        order = [test_id for test_id, _suite, _name in items]
        names = {test_id: (suite, name) for test_id, suite, name in items}
        collected: list[TestResult] = []

        def on_phase(key: str) -> None:
            if progress and key in order:
                progress(index + order.index(key), total, labels.get(key, "GPU"))

        def on_measured(key: str, payload: tuple) -> None:
            if key not in names:
                return
            suite, name = names[key]
            value, unit, detail, seconds = payload
            if value is None:
                item = TestResult(key, suite, name, None, unit, None, seconds, error=detail)
            else:
                item = TestResult(
                    id=key,
                    suite=suite,
                    name=name,
                    value=value,
                    unit=unit,
                    score=score_value(key, value),
                    seconds=seconds,
                    detail=detail,
                )
            collected.append(item)
            if on_result:
                on_result(item)

        started = time.perf_counter()
        try:
            measure_gpu(
                self.settings.gpu_seconds,
                self.settings.quick,
                self.stop,
                on_phase,
                set(names),
                on_measured,
            )
        except Exception as exc:
            failed = time.perf_counter() - started
            message = str(exc)
            emitted = {item.id for item in collected}
            for test_id, suite, name in items:
                if test_id in emitted:
                    continue
                item = TestResult(test_id, suite, name, None, "", None, failed, error=message)
                collected.append(item)
                if on_result:
                    on_result(item)
        return collected

    def _run_disk(self, items, progress, index: int, total: int, on_result) -> list[TestResult]:
        names = {test_id: (suite, name) for test_id, suite, name in items}
        order = [test_id for test_id, _suite, _name in items]
        collected: list[TestResult] = []

        def on_phase(key: str) -> None:
            if progress and key in order:
                progress(index + order.index(key), total, names[key][1])

        def on_done(key: str, value: float, unit: str, detail: str, seconds: float) -> None:
            if key not in names:
                return
            suite, name = names[key]
            item = TestResult(
                id=key,
                suite=suite,
                name=name,
                value=value,
                unit=unit,
                score=score_value(key, value),
                seconds=seconds,
                detail=detail,
            )
            collected.append(item)
            if on_result:
                on_result(item)

        started = time.perf_counter()
        try:
            _measure_disk(self.settings, on_phase, on_done)
        except Exception as exc:
            failed = time.perf_counter() - started
            message = str(exc)
            emitted = {item.id for item in collected}
            for test_id, suite, name in items:
                if test_id in emitted:
                    continue
                item = TestResult(test_id, suite, name, None, "", None, failed, error=message)
                collected.append(item)
                if on_result:
                    on_result(item)
        return collected


def _scale_arg(arg, factor: int):
    if isinstance(arg, tuple):
        head, *rest = arg
        return (head, rest[0] * factor) if len(rest) == 1 else arg
    return arg * factor


def _measure_disk(settings: Settings, on_phase=None, on_done=None) -> dict[str, tuple[float, str, str, float]]:
    def phase(key: str) -> None:
        if on_phase:
            on_phase(key)

    def done(key: str, value: float, unit: str, detail: str, seconds: float) -> None:
        if on_done:
            on_done(key, value, unit, detail, seconds)

    root = tempfile.mkdtemp(prefix="pyformance_")
    path = os.path.join(root, "sample.bin")
    block = 1 << 20
    try:
        free = shutil.disk_usage(root).free
        reserve = 512 << 20
        if free < settings.disk_floor_bytes + reserve:
            raise DiskError("Not enough free disk space for the disk tests")

        raw, ptr = aligned_region(block)
        fill_pattern(ptr, block)
        probe_blocks = 32 if settings.quick else 64
        unbuffered = True
        try:
            started = time.perf_counter()
            write_bytes(path, ptr, block, probe_blocks, True)
            probe_elapsed = max(time.perf_counter() - started, 1e-4)
        except DiskError:
            unbuffered = False
            started = time.perf_counter()
            write_bytes(path, ptr, block, probe_blocks, False)
            probe_elapsed = max(time.perf_counter() - started, 1e-4)

        probe_rate = (probe_blocks * block) / probe_elapsed
        target = int(probe_rate * settings.disk_seconds)
        total = max(settings.disk_floor_bytes, min(settings.disk_cap_bytes, target))
        total = max(block, total - (total % block))
        if total > free - reserve:
            total = max(block, ((free - reserve) // block) * block)
        blocks = total // block
        mode = "unbuffered" if unbuffered else "buffered"
        detail = f"{_format_bytes(total)}, {mode}"

        os.remove(path)
        phase("disk_write")
        started = time.perf_counter()
        write_bytes(path, ptr, block, blocks, unbuffered)
        write_elapsed = max(time.perf_counter() - started, 1e-6)
        write_rate = total / write_elapsed / 1e6
        done("disk_write", write_rate, "MB/s", detail, write_elapsed)

        read_raw, read_ptr = aligned_region(block)
        phase("disk_read")
        started = time.perf_counter()
        read_bytes(path, read_ptr, block, blocks, unbuffered)
        read_elapsed = max(time.perf_counter() - started, 1e-6)
        if ctypes.string_at(read_ptr, 64) != ctypes.string_at(ptr, 64):
            raise DiskError("Disk read did not return the written bytes")
        read_rate = total / read_elapsed / 1e6
        done("disk_read", read_rate, "MB/s", detail, read_elapsed)

        rand_block = 4096
        rand_raw, rand_ptr = aligned_region(rand_block)
        phase("disk_rand")
        # Aim for disk_seconds of random IO, with a floor so fast drives still get a sample.
        sample_ops = 40
        started = time.perf_counter()
        random_reads(path, rand_ptr, total, rand_block, sample_ops, unbuffered)
        sample_elapsed = max(time.perf_counter() - started, 1e-4)
        ops = max(sample_ops, int(sample_ops * settings.disk_seconds / sample_elapsed))
        ops = min(ops, 200_000)
        started = time.perf_counter()
        completed = random_reads(path, rand_ptr, total, rand_block, ops, unbuffered)
        rand_elapsed = max(time.perf_counter() - started, 1e-6)
        iops = completed / rand_elapsed
        done("disk_rand", iops, "IOPS", f"4 KB, {mode}", rand_elapsed)
        return {
            "disk_write": (write_rate, "MB/s", detail, write_elapsed),
            "disk_read": (read_rate, "MB/s", detail, read_elapsed),
            "disk_rand": (iops, "IOPS", f"4 KB, {mode}", rand_elapsed),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _emit_suite(results: list[TestResult], suite: str, on_suite: Callable[[str, float], None] | None) -> None:
    if on_suite is None:
        return
    scores = [item.score for item in results if item.suite == suite and item.score is not None]
    if scores:
        on_suite(suite, geometric_mean(scores))


def _suite_scores(results: list[TestResult]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in results:
        if item.score is None:
            continue
        grouped.setdefault(item.suite, []).append(item.score)
    return {suite: geometric_mean(scores) for suite, scores in grouped.items()}


def render_text(report: Report) -> str:
    info = report.system
    lines = [
        "PyFormance {}".format(__version__),
        _hardware_line(info),
        "",
    ]
    suite_title = {"cpu": "CPU", "memory": "Memory", "disk": "Disk", "gpu": "GPU"}
    current = None
    for item in report.results:
        if item.suite != current:
            current = item.suite
            lines.append(suite_title.get(current, current))
        lines.append(format_result_line(item))
    lines.append("")
    for suite, value in report.suite_scores.items():
        lines.append(f"{suite_title.get(suite, suite)} score  {value:.0f}")
    if report.overall is not None:
        lines.append(f"Overall     {report.overall:.0f}")
    lines.append("")
    lines.append(SCORE_NOTE)
    lines.append(f"Finished in {report.elapsed:.1f}s")
    return "\n".join(lines)


def _hardware_line(info: dict) -> str:
    parts = [
        str(info["processor"]),
        f"{info['logical_processors']} threads",
        f"{info['ram']} RAM",
    ]
    if info.get("gpu"):
        parts.append(str(info["gpu"]))
    parts.append(f"Python {info['python']}")
    return "  |  ".join(parts)


def write_json(report: Report, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
        handle.write("\n")


def self_test() -> None:
    if count_primes_below(100) != 25 or count_primes_below(1000) != 168:
        raise SystemExit("prime sieve failed")
    if integer_kernel(1000) == 0 and float_kernel(100) == 0.0:
        raise SystemExit("kernels returned empty checksums")
    settings = Settings(quick=True, suites=("disk",), workers=1)
    bundle = _measure_disk(settings)
    for key in ("disk_write", "disk_read", "disk_rand"):
        value = bundle[key][0]
        if value <= 0:
            raise SystemExit(f"{key} returned {value}")
    with ProcessPoolExecutor(max_workers=2) as pool:
        checksums = list(pool.map(integer_kernel, [2_000, 2_000]))
    if checksums[0] != checksums[1] or checksums[0] == 0:
        raise SystemExit("worker pool returned an unexpected checksum")
    print(f"  gpu: {verify_gpu()}")
    print("self-test ok")
    for key, (value, unit, detail, _seconds) in bundle.items():
        print(f"  {key}: {format_metric(value, unit)} ({detail})")
