"""Command line entry point.

Run a full pass:
    python benchmark.py --cli

Shorter pass:
    python benchmark.py --cli --quick

Open the window:
    python benchmark.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyformance.engine import (
    SCORE_NOTE,
    TESTS,
    Benchmark,
    Settings,
    format_banner,
    format_result_line,
    self_test,
    write_json,
)

_DEFAULT_TESTS = "cpu,memory,disk,gpu"


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep the suite list and examples on their own lines."""


def _choice_text() -> str:
    lines = [
        "suites and tests:",
        "  --tests accepts suite names or individual test names.",
        f"  default: {_DEFAULT_TESTS}",
        "",
    ]
    current = None
    for test_id, suite, name in TESTS:
        if suite != current:
            current = suite
            lines.append(f"  {suite}")
        lines.append(f"    {test_id:<12} {name}")
    lines.extend(
        [
            "",
            "examples:",
            "  python benchmark.py",
            "  python benchmark.py --cli",
            "  python benchmark.py --cli --quick",
            "  python benchmark.py --cli --tests cpu,gpu",
            "  python benchmark.py --cli --tests sha256,gpu_fp32",
            "  python benchmark.py --cli --workers 8 --json results.json",
            "  python benchmark.py --stress",
            "  python benchmark.py --stress 50",
            "  python benchmark.py --stress 25",
            "  python benchmark.py --stress cpu",
            "  python benchmark.py --stress gpu",
            "  python benchmark.py --self-test",
        ]
    )
    return "\n".join(lines)


def _stress_target(value: str) -> tuple[str, int]:
    if value in {"cpu", "gpu", "both"}:
        return value, 100
    try:
        percent = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected cpu, gpu, both, or a load from 1 to 100") from None
    if percent < 1 or percent > 100:
        raise argparse.ArgumentTypeError("load must be from 1 to 100")
    return "both", percent


def _resolve_tests(raw: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    suites_by_name = {suite for _test_id, suite, _name in TESTS}
    ids_by_name = {test_id: suite for test_id, suite, _name in TESTS}
    selected: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name in suites_by_name:
            for test_id, suite, _name in TESTS:
                if suite == name and test_id not in seen:
                    selected.append(test_id)
                    seen.add(test_id)
        elif name in ids_by_name:
            if name not in seen:
                selected.append(name)
                seen.add(name)
        else:
            unknown.append(name)
    if unknown or not selected:
        known = ", ".join([*sorted(suites_by_name), *ids_by_name])
        bad = ", ".join(unknown) if unknown else raw
        raise argparse.ArgumentTypeError(
            f"unknown --tests value {bad!r}. Choose from: {known}"
        )
    suites: list[str] = []
    for test_id, suite, _name in TESTS:
        if test_id in seen and suite not in suites:
            suites.append(suite)
    return tuple(suites), tuple(selected)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description=(
            "Benchmark CPU, memory, disk, and GPU.\n"
            "A score of 100 matches the built-in reference.\n"
            "With no mode flag, a window opens."
        ),
        epilog=_choice_text(),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="print the report in the terminal instead of opening the window",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="shorter pass of the selected tests",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="run in the terminal and write the report to FILE",
    )
    parser.add_argument(
        "--tests",
        default=_DEFAULT_TESTS,
        metavar="NAMES",
        help=(
            "comma-separated suite or test names to run "
            f"(default: {_DEFAULT_TESTS}). Names are listed below"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help="worker processes for the CPU tests (default: every logical processor)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the math kernels, worker pool, a small disk run, and the GPU",
    )
    parser.add_argument(
        "--stress",
        nargs="?",
        const=("both", 100),
        type=_stress_target,
        metavar="{cpu,gpu,both,PERCENT}",
        help="pin the machine until Ctrl+C. cpu, gpu, both, or a load from 1 to 100 (default: both at 100%%)",
    )
    args = parser.parse_args(argv)

    if args.stress:
        from pyformance.stress import run_stress

        mode, percent = args.stress
        run_stress(mode, args.workers, percent)
        return

    if args.self_test:
        self_test()
        return

    try:
        suites, test_ids = _resolve_tests(args.tests)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if not args.cli and not args.json:
        from pyformance.gui import launch

        launch()
        return

    settings = Settings(quick=args.quick, suites=suites, tests=test_ids, workers=args.workers)
    suite_title = {"cpu": "CPU", "memory": "Memory", "disk": "Disk", "gpu": "GPU"}

    def progress(done: int, total: int, message: str) -> None:
        if message == "Done":
            return
        print(f"[{done + 1}/{total}] {message}", flush=True)

    def on_start(info: dict) -> None:
        print(format_banner(info))
        print(flush=True)

    def on_result(item) -> None:
        print(format_result_line(item), flush=True)

    def on_suite(suite: str, score: float) -> None:
        print(flush=True)
        print(f"{suite_title.get(suite, suite)} score  {score:.0f}", flush=True)
        print(flush=True)

    report = Benchmark(settings).run(progress=progress, on_result=on_result, on_suite=on_suite, on_start=on_start)
    if report.overall is not None:
        print(f"Overall     {report.overall:.0f}")
        print()
    print(SCORE_NOTE)
    print(f"Finished in {report.elapsed:.1f}s")
    if args.json:
        write_json(report, args.json)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
