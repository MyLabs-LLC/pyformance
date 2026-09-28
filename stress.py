"""Pin the CPU, the GPU, or both until Ctrl+C."""

from __future__ import annotations

import ctypes
import multiprocessing
import threading
import time

from pyformance.kernels import integer_kernel


def cpu_burn(percent: int = 100) -> None:
    if percent >= 100:
        while True:
            integer_kernel(300_000)
    started = time.perf_counter()
    integer_kernel(4_000)
    probe = max(time.perf_counter() - started, 1e-4)
    chunk = max(200, int(4_000 * (0.02 / probe)))
    while True:
        started = time.perf_counter()
        integer_kernel(chunk)
        worked = time.perf_counter() - started
        idle = worked * (100 - percent) / percent
        if idle > 0:
            time.sleep(idle)


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _filetime(value: _FileTime) -> int:
    return (value.high << 32) | value.low


def _cpu_times() -> tuple[int, int]:
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return 0, 0
    idle_ticks = _filetime(idle)
    total = _filetime(kernel) + _filetime(user)
    return idle_ticks, total


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


class _GpuLoad:
    """NVIDIA utilization from NVML. This is the Compute engine, not the 3D graph."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._device = ctypes.c_void_p()

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

    def percent(self) -> float | None:
        if self._lib is None:
            return None
        sample = _NvmlUtilization()
        if self._lib.nvmlDeviceGetUtilizationRates(self._device, ctypes.byref(sample)) != 0:
            return None
        return float(sample.gpu)

    def close(self) -> None:
        if self._lib is not None:
            self._lib.nvmlShutdown()
            self._lib = None


def run_stress(mode: str, workers: int | None = None, percent: int = 100) -> None:
    use_cpu = mode in {"cpu", "both"}
    use_gpu = mode in {"gpu", "both"}
    stop = threading.Event()
    processes: list[multiprocessing.Process] = []
    gpu_thread: threading.Thread | None = None
    gpu_error: list[str] = []

    if use_cpu:
        logical = multiprocessing.cpu_count() or 1
        limit = workers or logical
        # Peg whole logical processors. A per-thread pause runs hot once every thread is scheduled.
        target_cores = min(float(limit), logical * percent / 100)
        full = int(target_cores)
        remainder = target_cores - full
        context = multiprocessing.get_context("spawn")
        for _ in range(full):
            process = context.Process(target=cpu_burn, args=(100,), daemon=True)
            process.start()
            processes.append(process)
        if remainder >= 0.02:
            process = context.Process(target=cpu_burn, args=(max(1, round(remainder * 100)),), daemon=True)
            process.start()
            processes.append(process)
        print(f"CPU stress: {percent}% of {logical} threads", flush=True)

    if use_gpu:
        def gpu_target() -> None:
            try:
                from pyformance.gpu import CudaSession

                with CudaSession() as gpu:
                    gpu.load()
                    print(f"GPU stress: {gpu.name} at {percent}%", flush=True)
                    gpu.stress(stop, percent)
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
    gpu_samples: list[float] = []
    paused: set[int] = set()
    logical = multiprocessing.cpu_count() or 1
    try:
        while not stop.is_set():
            time.sleep(0.2)
            if gpu_error and not use_cpu:
                break
            if gpu_load is not None:
                sample = gpu_load.percent()
                if sample is not None:
                    gpu_samples.append(sample)
            elapsed = int(time.monotonic() - started)
            if elapsed and elapsed % 5 == 0:
                cpu = cpu_load.percent() if cpu_load is not None else None
                gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else None
                gpu_samples.clear()
                print(_status_line(elapsed, cpu, gpu, cpu_load is not None, gpu_load is not None), flush=True)
                if percent < 100 and cpu is not None and elapsed >= 10:
                    _balance_cpu(processes, paused, cpu, percent, logical)
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        stop.set()
        if gpu_load is not None:
            gpu_load.close()
        if gpu_thread is not None:
            gpu_thread.join(timeout=5)
        _stop_cpu(processes)


def _status_line(elapsed: int, cpu: float | None, gpu: float | None, show_cpu: bool, show_gpu: bool) -> str:
    parts = [f"{elapsed // 60:02d}:{elapsed % 60:02d}"]
    if show_cpu:
        parts.append("CPU  n/a" if cpu is None else f"CPU {cpu:3.0f}%")
    if show_gpu:
        parts.append("GPU  n/a" if gpu is None else f"GPU {gpu:3.0f}%")
    return "  ".join(parts)


def _balance_cpu(
    processes: list[multiprocessing.Process],
    paused: set[int],
    measured: float,
    target: int,
    logical: int,
) -> None:
    """Suspend or resume workers so the whole machine settles on the requested load."""
    miss = measured - target
    steps = int(abs(miss) / 100 * logical)
    if steps < 1:
        return
    if miss > 4:
        running = [process for process in processes if process.pid not in paused and process.is_alive()]
        for process in running[:steps]:
            if _set_paused(process, True):
                paused.add(process.pid)
    elif miss < -4:
        held = [process for process in processes if process.pid in paused and process.is_alive()]
        for process in held[:steps]:
            if _set_paused(process, False):
                paused.discard(process.pid)


def _set_paused(process: multiprocessing.Process, paused: bool) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    ntdll.NtSuspendProcess.argtypes = [ctypes.c_void_p]
    ntdll.NtSuspendProcess.restype = ctypes.c_long
    ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    handle = kernel32.OpenProcess(0x0800, False, process.pid)
    if not handle:
        return False
    code = ntdll.NtSuspendProcess(handle) if paused else ntdll.NtResumeProcess(handle)
    kernel32.CloseHandle(handle)
    return code == 0


def _stop_cpu(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)
