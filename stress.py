"""Pin the CPU, the GPU, or both until Ctrl+C."""

from __future__ import annotations

import ctypes
import multiprocessing
import threading
import time

from pyformance.engine import (
    TESTS,
    Benchmark,
    Settings,
    TestResult,
    format_result_line,
    score_value,
)
_PRINT = threading.Lock()


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _ProcessorPower(ctypes.Structure):
    _fields_ = [
        ("number", ctypes.c_ulong),
        ("max_mhz", ctypes.c_ulong),
        ("current_mhz", ctypes.c_ulong),
        ("mhz_limit", ctypes.c_ulong),
        ("max_idle_state", ctypes.c_ulong),
        ("current_idle_state", ctypes.c_ulong),
    ]


# NVML clock type for the streaming multiprocessors.
_NVML_CLOCK_SM = 1


def _filetime(value: _FileTime) -> int:
    return (value.high << 32) | value.low


def _cpu_times() -> tuple[int, int]:
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return 0, 0
    idle_ticks = _filetime(idle)
    total = _filetime(kernel) + _filetime(user)
    return idle_ticks, total


def _cpu_clock_ghz() -> float | None:
    """Sum of the current clocks across every logical processor."""
    count = multiprocessing.cpu_count() or 1
    buf = (_ProcessorPower * count)()
    powrprof = ctypes.WinDLL("powrprof", use_last_error=True)
    call = powrprof.CallNtPowerInformation
    call.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
    call.restype = ctypes.c_long
    status = call(11, None, 0, ctypes.byref(buf), ctypes.sizeof(buf))
    if status != 0:
        return None
    return sum(int(item.current_mhz) for item in buf) / 1000.0


class _CpuLoad:
    """System-wide CPU use since the previous sample. Kernel time includes idle."""

    def __init__(self) -> None:
        self._previous = _cpu_times()

    def percent(self) -> float | None:
        current = _cpu_times()
        idle_delta = current[0] - self._previous[0]
        total_delta = current[1] - self._previous[1]
        self._previous = current
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))

    def clock_ghz(self) -> float | None:
        return _cpu_clock_ghz()


class _GpuLoad:
    """NVIDIA utilization from NVML. This is the Compute engine, not the 3D graph."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._device = ctypes.c_void_p()
        self._cores = 0

    def open(self) -> None:
        lib = ctypes.CDLL("nvml.dll")
        lib.nvmlInit_v2.restype = ctypes.c_int
        if lib.nvmlInit_v2() != 0:
            raise OSError("NVML did not start")
        lib.nvmlShutdown.restype = ctypes.c_int
        lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
        lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization)]
        lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        lib.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        lib.nvmlDeviceGetClockInfo.restype = ctypes.c_int
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
            lib.nvmlShutdown()
            raise OSError("NVML found no NVIDIA GPU")
        chosen = ctypes.c_void_p()
        best = -1
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                continue
            memory = _NvmlMemory()
            if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                continue
            if memory.total > best:
                best = int(memory.total)
                chosen = handle
        if best < 0:
            lib.nvmlShutdown()
            raise OSError("NVML could not read a GPU")
        self._lib = lib
        self._device = chosen
        self._cores = self._core_count()

    def _core_count(self) -> int:
        if self._lib is None:
            return 0
        get_cores = getattr(self._lib, "nvmlDeviceGetNumGpuCores", None)
        if get_cores is None:
            return 0
        get_cores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        get_cores.restype = ctypes.c_int
        value = ctypes.c_uint()
        if get_cores(self._device, ctypes.byref(value)) != 0:
            return 0
        return int(value.value)

    def percent(self) -> float | None:
        sample = self.sample()
        if sample is None:
            return None
        return sample[0]

    def sample(self) -> tuple[float, float | None] | None:
        """Utilization and FP32 throughput. An FMA counts as two FLOPs."""
        if self._lib is None:
            return None
        rates = _NvmlUtilization()
        if self._lib.nvmlDeviceGetUtilizationRates(self._device, ctypes.byref(rates)) != 0:
            return None
        util = float(rates.gpu)
        clock = ctypes.c_uint()
        if self._cores <= 0 or self._lib.nvmlDeviceGetClockInfo(self._device, _NVML_CLOCK_SM, ctypes.byref(clock)) != 0:
            return util, None
        tflops = self._cores * 2 * int(clock.value) * 1e6 * (util / 100.0) / 1e12
        return util, tflops

    def close(self) -> None:
        if self._lib is not None:
            self._lib.nvmlShutdown()
            self._lib = None


def _print_calc(kind: str, test_id: str, name: str, value: float | None, unit: str, error: str | None) -> None:
    score = None if error or value is None else score_value(test_id, value)
    item = TestResult(
        id=test_id,
        suite=kind.lower(),
        name=name,
        value=value,
        unit=unit,
        score=score,
        seconds=0.0,
        error=error,
    )
    with _PRINT:
        print(f"{kind:<5}{format_result_line(item).lstrip()}", flush=True)


def run_stress(mode: str, workers: int | None = None, percent: int = 100) -> None:
    use_cpu = mode in {"cpu", "both"}
    use_gpu = mode in {"gpu", "both"}
    stop = threading.Event()
    processes: list[multiprocessing.Process] = []
    gpu_thread: threading.Thread | None = None
    gpu_error: list[str] = []

    logical = multiprocessing.cpu_count() or 1
    cpu_workers = workers or logical
    if percent < 100:
        cpu_workers = max(1, int(round(cpu_workers * percent / 100)))
    if use_cpu:
        print(f"CPU stress: cycling tests on {cpu_workers} of {logical} threads", flush=True)

        def cpu_target() -> None:
            settings = Settings(workers=cpu_workers, suites=("cpu",))
            bench = Benchmark(settings, stop)
            cpu_tests = [(test_id, name) for test_id, suite, name in TESTS if suite == "cpu"]
            try:
                while not stop.is_set():
                    for test_id, name in cpu_tests:
                        if stop.is_set():
                            return
                        try:
                            value, unit, _detail = bench.measure_one(test_id)
                        except Exception as exc:
                            _print_calc("CPU", test_id, name, None, "", str(exc))
                            continue
                        _print_calc("CPU", test_id, name, value, unit, None)
            finally:
                bench.close()

        cpu_thread = threading.Thread(target=cpu_target, daemon=True)
        cpu_thread.start()
    else:
        cpu_thread = None

    if use_gpu:
        def gpu_target() -> None:
            try:
                from pyformance.gpu import CudaSession

                with CudaSession() as gpu:
                    gpu.load()
                    print(f"GPU stress: {gpu.name}, cycling tests", flush=True)
                    seconds = Settings().gpu_seconds
                    gpu.cycle(stop, seconds, False, lambda *args: _print_calc("GPU", *args))
            except Exception as exc:
                gpu_error.append(str(exc))

        gpu_thread = threading.Thread(target=gpu_target, daemon=True)
        gpu_thread.start()
        # Give the device line a moment to appear before the timer.
        time.sleep(0.4)
        if gpu_error and mode == "gpu":
            _stop_cpu(processes)
            raise SystemExit(gpu_error[0])
        if gpu_error:
            print(f"GPU stress did not start: {gpu_error[0]}", flush=True)

    print("Press Ctrl+C to stop.", flush=True)
    cpu_load = _CpuLoad() if use_cpu else None
    gpu_load = None
    if use_gpu and not gpu_error:
        gpu_load = _GpuLoad()
        try:
            gpu_load.open()
        except OSError as exc:
            print(f"GPU load is unavailable: {exc}", flush=True)
            gpu_load = None
    started = time.monotonic()
    gpu_utils: list[float] = []
    gpu_flops: list[float] = []
    cpu_clocks: list[float] = []
    try:
        while not stop.is_set():
            time.sleep(0.2)
            if gpu_error and not use_cpu:
                break
            if cpu_load is not None:
                clock = cpu_load.clock_ghz()
                if clock is not None:
                    cpu_clocks.append(clock)
            if gpu_load is not None:
                sample = gpu_load.sample()
                if sample is not None:
                    gpu_utils.append(sample[0])
                    if sample[1] is not None:
                        gpu_flops.append(sample[1])
            elapsed = int(time.monotonic() - started)
            if elapsed and elapsed % 5 == 0:
                cpu = cpu_load.percent() if cpu_load is not None else None
                cpu_ghz = None
                if cpu is not None and cpu_clocks:
                    cpu_ghz = (sum(cpu_clocks) / len(cpu_clocks)) * (cpu / 100.0)
                gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
                gpu_tflops = sum(gpu_flops) / len(gpu_flops) if gpu_flops else None
                cpu_clocks.clear()
                gpu_utils.clear()
                gpu_flops.clear()
                with _PRINT:
                    print(
                        _status_line(elapsed, cpu, cpu_ghz, gpu, gpu_tflops, cpu_load is not None, gpu_load is not None),
                        flush=True,
                    )
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        stop.set()
        if gpu_load is not None:
            gpu_load.close()
        if cpu_thread is not None:
            cpu_thread.join(timeout=20)
        if gpu_thread is not None:
            gpu_thread.join(timeout=20)
        _stop_cpu(processes)


def _format_tflops(tflops: float) -> str:
    if tflops >= 1:
        return f"{tflops:.2f} TFLOPS"
    return f"{tflops * 1000:.0f} GFLOPS"


def _status_line(
    elapsed: int,
    cpu: float | None,
    cpu_ghz: float | None,
    gpu: float | None,
    gpu_tflops: float | None,
    show_cpu: bool,
    show_gpu: bool,
) -> str:
    parts = [f"{elapsed // 60:02d}:{elapsed % 60:02d}"]
    if show_cpu:
        if cpu is None:
            parts.append("CPU  n/a")
        elif cpu_ghz is None:
            parts.append(f"CPU {cpu:3.0f}%")
        else:
            parts.append(f"CPU {cpu:3.0f}%  {cpu_ghz:.1f} GHz")
    if show_gpu:
        if gpu is None:
            parts.append("GPU  n/a")
        elif gpu_tflops is None:
            parts.append(f"GPU {gpu:3.0f}%")
        else:
            parts.append(f"GPU {gpu:3.0f}%  {_format_tflops(gpu_tflops)}")
    return "  ".join(parts)


def _stop_cpu(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)
