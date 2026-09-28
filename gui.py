"""Desktop window for running the benchmark."""

from __future__ import annotations

import ctypes
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pyformance import __version__
from pyformance.engine import Benchmark, Report, Settings, TestResult, geometric_mean, system_info, write_json

BG = "#10141c"
CARD = "#1a2130"
INK = "#e8eef7"
MUTED = "#93a0b5"
ACCENT = "#4c8dff"
LINE = "#2a3346"


def _format_seconds(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def _enable_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class App:
    def __init__(self) -> None:
        _enable_dpi()
        self.root = tk.Tk()
        self.root.title(f"PyFormance {__version__}")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)
        self.root.configure(bg=BG)
        self.info = system_info()
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.report: Report | None = None
        self.results: list[TestResult] = []
        self.events: queue.Queue = queue.Queue()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(50, self._poll)

    def _build(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=CARD, foreground=INK, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=INK, font=("Segoe UI", 22, "bold"))
        style.configure("Score.TLabel", background=CARD, foreground=INK, font=("Segoe UI", 22, "bold"))
        style.configure("ScoreName.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 6))
        style.configure("TCheckbutton", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BG)], foreground=[("active", INK)])
        style.configure(
            "Treeview",
            background=CARD,
            fieldbackground=CARD,
            foreground=INK,
            borderwidth=0,
            rowheight=30,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#222a3c",
            foreground=INK,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#243656")], foreground=[("selected", INK)])
        style.configure(
            "Run.Horizontal.TProgressbar",
            troughcolor="#222a3c",
            background=ACCENT,
            bordercolor=BG,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="PyFormance", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="CPU, memory, disk, and GPU benchmark. Scores compare runs of this program.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        hardware_parts = [
            self.info["processor"],
            f"{self.info['logical_processors']} threads",
            f"{self.info['ram']} RAM",
        ]
        if self.info.get("gpu"):
            hardware_parts.append(self.info["gpu"])
        hardware_parts.append(f"Python {self.info['python']}")
        hardware = "   ·   ".join(hardware_parts)
        ttk.Label(header, text=hardware, style="Muted.TLabel").pack(anchor="w", pady=(2, 12))

        cards = ttk.Frame(outer)
        cards.pack(fill="x", pady=(0, 12))
        self.score_labels: dict[str, ttk.Label] = {}
        for index, (key, title) in enumerate(
            (("overall", "Overall"), ("cpu", "CPU"), ("memory", "Memory"), ("disk", "Disk"), ("gpu", "GPU"))
        ):
            card = tk.Frame(cards, bg=CARD, highlightbackground=LINE, highlightthickness=1)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            cards.columnconfigure(index, weight=1)
            ttk.Label(card, text=title, style="ScoreName.TLabel").pack(anchor="w", padx=14, pady=(10, 0))
            label = ttk.Label(card, text="—", style="Score.TLabel")
            label.pack(anchor="w", padx=14, pady=(0, 10))
            self.score_labels[key] = label

        table_wrap = tk.Frame(outer, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        table_wrap.pack(fill="both", expand=True)
        columns = ("suite", "test", "result", "score", "time")
        self.tree = ttk.Treeview(table_wrap, columns=columns, show="headings", selectmode="browse")
        headings = {
            "suite": ("Suite", 110),
            "test": ("Test", 220),
            "result": ("Result", 180),
            "score": ("Score", 90),
            "time": ("Time", 80),
        }
        for key, (text, width) in headings.items():
            anchor = "w" if key in {"suite", "test", "result"} else "e"
            self.tree.heading(key, text=text, anchor=anchor)
            self.tree.column(key, width=width, anchor=anchor, stretch=key == "test")
        self.tree.pack(fill="both", expand=True, padx=1, pady=1)
        self.tree.tag_configure("odd", background="#151c2b")
        self.tree.tag_configure("even", background=CARD)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(12, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate", style="Run.Horizontal.TProgressbar")
        self.progress.pack(fill="x")
        self.status = ttk.Label(footer, text="Ready", style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(6, 8))

        buttons = ttk.Frame(footer)
        buttons.pack(fill="x")
        self.cpu_on = tk.BooleanVar(value=True)
        self.memory_on = tk.BooleanVar(value=True)
        self.disk_on = tk.BooleanVar(value=True)
        self.gpu_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(buttons, text="CPU", variable=self.cpu_on).pack(side="left")
        ttk.Checkbutton(buttons, text="Memory", variable=self.memory_on).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(buttons, text="Disk", variable=self.disk_on).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(buttons, text="GPU", variable=self.gpu_on).pack(side="left", padx=(8, 16))
        self.run_button = ttk.Button(buttons, text="Run", command=lambda: self._start(False))
        self.run_button.pack(side="left")
        self.quick_button = ttk.Button(buttons, text="Quick run", command=lambda: self._start(True))
        self.quick_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.save_button = ttk.Button(buttons, text="Save JSON", command=self._save, state="disabled")
        self.save_button.pack(side="right")

    def _start(self, quick: bool) -> None:
        if self.worker and self.worker.is_alive():
            return
        suites = []
        if self.cpu_on.get():
            suites.append("cpu")
        if self.memory_on.get():
            suites.append("memory")
        if self.disk_on.get():
            suites.append("disk")
        if self.gpu_on.get():
            suites.append("gpu")
        if not suites:
            messagebox.showinfo("PyFormance", "Choose at least one suite.")
            return
        self.results.clear()
        self.tree.delete(*self.tree.get_children())
        for label in self.score_labels.values():
            label.configure(text="—")
        self.report = None
        self.save_button.configure(state="disabled")
        self.stop.clear()
        self._set_running(True)
        settings = Settings(quick=quick, suites=tuple(suites))
        self.worker = threading.Thread(target=self._execute, args=(settings,), daemon=True)
        self.worker.start()

    def _execute(self, settings: Settings) -> None:
        bench = Benchmark(settings, self.stop)
        self._bench = bench
        try:
            report = bench.run(progress=self._progress, on_result=self._result)
        except Exception as exc:
            self._post(lambda: self._finish_error(str(exc)))
            return
        self._post(lambda: self._finish(report))

    def _post(self, fn) -> None:
        self.events.put(fn)

    def _poll(self) -> None:
        try:
            while True:
                self.events.get_nowait()()
        except queue.Empty:
            pass
        try:
            self.root.after(50, self._poll)
        except tk.TclError:
            pass

    def _progress(self, done: int, total: int, message: str) -> None:
        def update() -> None:
            self.progress.configure(maximum=max(total, 1), value=done)
            if message == "Done":
                self.status.configure(text="Done")
            elif self.stop.is_set():
                self.status.configure(text="Stopping after the current test")
            else:
                self.status.configure(text=f"Running {message}")

        self._post(update)

    def _result(self, item: TestResult) -> None:
        self._post(lambda: self._add_result(item))

    def _add_result(self, item: TestResult) -> None:
        self.results.append(item)
        tag = "even" if len(self.results) % 2 == 0 else "odd"
        score = "" if item.score is None else f"{item.score:.0f}"
        suite = {"cpu": "CPU", "memory": "Memory", "disk": "Disk", "gpu": "GPU"}.get(item.suite, item.suite)
        self.tree.insert(
            "",
            "end",
            values=(suite, item.name, item.result_text(), score, _format_seconds(item.seconds)),
            tags=(tag,),
        )
        self.tree.yview_moveto(1)
        self._refresh_scores()

    def _refresh_scores(self) -> None:
        grouped: dict[str, list[float]] = {}
        for item in self.results:
            if item.score is not None:
                grouped.setdefault(item.suite, []).append(item.score)
        suite_scores = {suite: geometric_mean(values) for suite, values in grouped.items()}
        for key in ("cpu", "memory", "disk", "gpu"):
            value = suite_scores.get(key)
            self.score_labels[key].configure(text="—" if value is None else f"{value:.0f}")
        if suite_scores:
            self.score_labels["overall"].configure(text=f"{geometric_mean(list(suite_scores.values())):.0f}")

    def _finish(self, report: Report) -> None:
        self.report = report
        self._set_running(False)
        if self.stop.is_set():
            self.status.configure(text=f"Stopped after {report.elapsed:.1f}s")
            if report.results:
                self.save_button.configure(state="normal")
            return
        if report.overall is None:
            self.status.configure(text="Stopped" if self.stop.is_set() else "No scores")
            return
        self.score_labels["overall"].configure(text=f"{report.overall:.0f}")
        for key, value in report.suite_scores.items():
            self.score_labels[key].configure(text=f"{value:.0f}")
        self.save_button.configure(state="normal")
        self.status.configure(
            text=(
                f"Finished in {report.elapsed:.1f}s. "
                "100 matches the reference. 200 is twice as fast. "
                "Overall is the geometric mean of the suite scores."
            )
        )

    def _finish_error(self, message: str) -> None:
        self._set_running(False)
        self.status.configure(text="The run failed")
        messagebox.showerror("PyFormance", message)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.quick_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")
        for child in self._checkbuttons():
            child.configure(state=state)

    def _checkbuttons(self):
        found = []

        def walk(widget) -> None:
            for child in widget.winfo_children():
                if isinstance(child, ttk.Checkbutton):
                    found.append(child)
                walk(child)

        walk(self.root)
        return found

    def _stop(self) -> None:
        self.stop.set()
        self.status.configure(text="Stopping after the current test")

    def _save(self) -> None:
        if self.report is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="pyformance.json",
        )
        if not path:
            return
        write_json(self.report, path)
        self.status.configure(text=f"Saved {path}")

    def _close(self) -> None:
        self.stop.set()
        bench = getattr(self, "_bench", None)
        if bench is not None:
            bench.close()
        self.root.destroy()

    def mainloop(self) -> None:
        self.root.mainloop()


def launch() -> None:
    App().mainloop()
