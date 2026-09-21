"""Reproducible scaling benchmarks for the Blackwell union algorithms.

Examples (from the repository root)::

    python benchmarks/benchmark_union.py --algorithm target --repeats 10 \
        --output benchmarks/results/union-target
    python benchmarks/benchmark_union.py --algorithm sources --sizes 3 4 \
        --counts 1 3 10 30 100 300 1000 --case-budget-seconds 60 \
        --output benchmarks/results/union-sources

A case runs in one isolated worker process with one native library thread.
Imports, input generation, one warmup, and one calibration call are excluded
from the recorded timings. Each repeat times exactly one public solver call.
The coordinator checkpoints every event to JSONL and can kill an over-budget
worker. A summary median is emitted only after all requested repeats succeed.
Larger cases on a censored/failed curve are skipped by default; this is a
budget policy, not a claim about their actual running times. Use
--continue-after-censor to attempt them. Use --resume to reuse completed
cases from the same implementation and configuration.

The binary-target inputs reproduce the conditional-weight protocol of the
original redundancy benchmarks: uniform prior, independent Dirichlet rows,
uint32 weights with row total 2**30. The default binary-source inputs have
independent interior Bernoulli rows; they deliberately exercise the general
union optimizer and differ from the exactly collinear redundancy benchmark.
The optional --source-family collinear reproduces that easier input family.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import selectors
import statistics
import subprocess
import sys
from time import perf_counter
import traceback
from uuid import uuid4
from typing import Any

# Set before importing NumPy or the package; timed solvers run serially.
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS",
)
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROW_TOTAL = 1 << 30
COUNTS = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000,
          30000, 100000, 300000, 1000000)
TARGET_SIZES = (2, 8, 32, 128)
SOURCE_SIZES = (2, 3, 4, 8)
SCHEMA_VERSION = 1


def git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *arguments], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def implementation() -> dict[str, Any]:
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO / "discrete_pid").glob("*.py"))
    }
    encoded = json.dumps(sources, sort_keys=True).encode()
    status = git("status", "--porcelain")
    return {
        "implementation_commit": git("rev-parse", "HEAD"),
        "implementation_dirty": None if status is None else bool(status),
        "implementation_source_sha256": sources,
        "implementation_fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return platform.processor() or platform.machine()


def versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "clarabel", "numba", "threadpoolctl"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def native_pools() -> list[dict[str, Any]] | None:
    try:
        from threadpoolctl import threadpool_info
        return threadpool_info()
    except ImportError:
        return None


def inputs(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Generate prefix-stable cases without retaining a maximum-size family."""
    algorithm, size, count = case["algorithm"], case["size"], case["sources"]
    rng = np.random.default_rng(np.random.SeedSequence(
        [case["seed"], size, algorithm == "sources"]
    ))
    if algorithm == "target":
        data = np.empty((count, 2, size), dtype=np.uint32)
        for start in range(0, count, 10000):
            stop = min(start + 10000, count)
            probabilities = rng.dirichlet(np.ones(size), size=(stop - start, 2))
            rows = np.floor(probabilities * ROW_TOTAL).astype(np.uint32).reshape(-1, size)
            remainder = ROW_TOTAL - rows.sum(axis=1, dtype=np.uint64)
            largest = np.argmax(probabilities.reshape(-1, size), axis=1)
            rows[np.arange(len(rows)), largest] += remainder.astype(np.uint32)
            data[start:stop] = rows.reshape(probabilities.shape)
        return np.full(2, 0.5), data

    data = np.empty((count, size, 2), dtype=np.uint32)
    if case["source_family"] == "collinear":
        direction = np.resize(np.array([1, -1]), size)
        ab = rng.integers(1, 41, size=(count, 2))
        for start in range(0, count, 10000):
            stop = min(start + 10000, count)
            a, b = ab[start:stop, 0, None], ab[start:stop, 1, None]
            data[start:stop, :, 0] = b * (100 - a * direction)
            data[start:stop, :, 1] = a * (100 + b * direction)
    else:
        for start in range(0, count, 10000):
            stop = min(start + 10000, count)
            probabilities = rng.uniform(0.05, 0.95, size=(stop - start, size))
            ones = np.rint(probabilities * ROW_TOTAL).astype(np.uint32)
            data[start:stop, :, 1] = ones
            data[start:stop, :, 0] = ROW_TOTAL - ones
    return np.full(size, 1.0 / size), data


def scalar(value: Any) -> Any:
    """Keep reproducibility diagnostics, but do not serialize large channels."""
    if isinstance(value, np.generic):
        return scalar(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()
                if item is None or isinstance(item, (str, bool, int, float, np.generic, dict))}
    return None


def result_diagnostics(result: Any) -> dict[str, Any]:
    names = (
        "union_nats", "lower_bound_nats", "upper_bound_nats", "gap_nats",
        "feasibility_residual", "iterations", "method", "max_garbling_residual",
    )
    report = {name: scalar(getattr(result, name, None)) for name in names}
    report["diagnostics"] = scalar(getattr(result, "diagnostics", None))
    if isinstance(report["diagnostics"], dict):
        for name in ("retained_sources", "support", "master_seconds", "pricing_seconds",
                     "clarabel_max_threads", "solver_threads", "linear_solver_threads"):
            report[name] = report["diagnostics"].get(name)
    return report


def validate_result(report: dict[str, Any], tolerance: float) -> None:
    value = report.get("union_nats")
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value < -1e-10:
        raise ArithmeticError(f"Invalid union value: {value!r}")
    lower, upper = report.get("lower_bound_nats"), report.get("upper_bound_nats")
    if lower is not None and upper is not None:
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in (lower, upper)):
            raise ArithmeticError("Nonfinite solver bounds")
        if lower > upper + 1e-9:
            raise ArithmeticError("Solver lower bound exceeds its upper bound")
    gap = report.get("gap_nats")
    if gap is not None and (not isinstance(gap, (int, float)) or not np.isfinite(gap)
                            or gap > tolerance * 1.05 + 1e-10):
        raise ArithmeticError(f"Solver did not reach tolerance {tolerance}: gap={gap!r}")
    residual = report.get("feasibility_residual")
    if residual is not None and (not isinstance(residual, (int, float))
                                or not np.isfinite(residual) or residual > 1e-6):
        raise ArithmeticError(f"Solver feasibility residual exceeds 1e-6: {residual!r}")


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def worker(payload: dict[str, Any]) -> int:
    case = payload["case"]
    started = perf_counter()
    common = {"case_id": case["case_id"], "attempt_id": payload["attempt_id"], "case": case}
    try:
        import discrete_pid
        solver = getattr(discrete_pid, "union_binary_" + case["algorithm"])
        kwargs: dict[str, Any] = {"return_channel": False, "return_garblings": False}
        if case["algorithm"] == "sources":
            kwargs.update(tolerance=case["tolerance"], max_iterations=case["max_iterations"])
        current = implementation()
        if current["implementation_fingerprint"] != case["implementation_fingerprint"]:
            raise RuntimeError("Implementation changed after the coordinator started")
        emit({**common, "event": "case_started", **current, "solver_kwargs": kwargs,
              "native_pools": native_pools(), "pid": os.getpid()})
        emit({**common, "event": "phase", "phase": "generation"})
        prior, data = inputs(case)
        emit({**common, "event": "input", "input_bytes": data.nbytes,
              "input_shape": list(data.shape), "input_dtype": str(data.dtype),
              "input_sha256": hashlib.sha256(memoryview(data).cast("B")).hexdigest()})
        emit({**common, "event": "phase", "phase": "warmup"})
        before = perf_counter()
        result = solver(prior, data, **kwargs)
        warmup_seconds = perf_counter() - before
        report = result_diagnostics(result)
        validate_result(report, case["tolerance"])
        pools_after_warmup = native_pools()
        emit({**common, "event": "warmup", "seconds": warmup_seconds,
              "native_pools_after_warmup": pools_after_warmup,
              "native_pool_introspection_available": pools_after_warmup is not None, **report})
        emit({**common, "event": "phase", "phase": "pilot"})
        before = perf_counter()
        result = solver(prior, data, **kwargs)
        pilot_seconds = perf_counter() - before
        report = result_diagnostics(result)
        validate_result(report, case["tolerance"])
        emit({**common, "event": "pilot", "seconds": pilot_seconds, **report})
        remaining = payload["case_budget_seconds"] - (perf_counter() - started)
        projected = pilot_seconds * case["repeats"]
        if projected > max(0.0, remaining) * 0.95:
            emit({**common, "event": "case_summary", "status": "censored_pilot_budget",
                  "pilot_seconds": pilot_seconds, "warmup_seconds": warmup_seconds,
                  "projected_repeat_seconds": projected, "remaining_budget_seconds": remaining,
                  "completed_repeats": 0, "median_seconds": None, **report})
            return 0
        samples = []
        for repeat in range(case["repeats"]):
            emit({**common, "event": "phase", "phase": "repeat", "repeat": repeat})
            before = perf_counter()
            result = solver(prior, data, **kwargs)
            seconds = perf_counter() - before
            report = result_diagnostics(result)
            # Persist the elapsed time even if the solver's accuracy check fails.
            emit({**common, "event": "repeat", "repeat": repeat, "seconds": seconds,
                  "calls_per_repeat": 1, **report})
            validate_result(report, case["tolerance"])
            samples.append(seconds)
        emit({**common, "event": "case_summary", "status": "ok", "completed_repeats": len(samples),
              "median_seconds": statistics.median(samples), "minimum_seconds": min(samples),
              "maximum_seconds": max(samples), "warmup_seconds": warmup_seconds,
              "pilot_seconds": pilot_seconds, "case_seconds": perf_counter() - started,
              **report})
        return 0
    except Exception as error:
        emit({**common, "event": "case_summary", "status": "failed", "median_seconds": None,
              "error_type": type(error).__name__, "error": str(error),
              "traceback": traceback.format_exc(), "case_seconds": perf_counter() - started})
        return 1


def append_event(stream: Any, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def run_case(case: dict[str, Any], args: argparse.Namespace, events: Any,
             timeout: float) -> dict[str, Any]:
    attempt_id = uuid4().hex
    payload = {"case": case, "case_budget_seconds": timeout, "attempt_id": attempt_id}
    error_path = args.output / "stderr" / f"{case['case_id']}.log"
    error_path.parent.mkdir(exist_ok=True)
    started = perf_counter()
    summary = None
    completed_repeats = 0
    phase = "imports"

    def record_line(line: bytes) -> None:
        nonlocal summary, completed_repeats, phase
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            event = {"case_id": case["case_id"], "event": "worker_output",
                     "text": line.decode(errors="replace").rstrip()}
        append_event(events, event)
        if event.get("event") == "phase":
            phase = event.get("phase", phase)
        if event.get("event") == "repeat":
            completed_repeats += 1
        if event.get("event") == "case_summary":
            summary = event

    with error_path.open("w") as errors:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(payload)],
            stdout=subprocess.PIPE, stderr=errors, bufsize=0, env=os.environ.copy(),
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffered = b""
        try:
            while True:
                remaining = timeout - (perf_counter() - started)
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    summary = {"case_id": case["case_id"], "attempt_id": attempt_id, "case": case,
                               "event": "case_summary", "status": "censored_timeout",
                               "phase": phase, "completed_repeats": completed_repeats,
                               "median_seconds": None, "case_seconds": perf_counter() - started}
                    append_event(events, summary)
                    break
                ready = selector.select(timeout=min(remaining, 0.5))
                if not ready:
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    if buffered:
                        record_line(buffered)
                    break
                buffered += chunk
                while b"\n" in buffered:
                    line, buffered = buffered.split(b"\n", 1)
                    record_line(line)
            process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            selector.close()
            process.stdout.close()
    if summary is None:
        summary = {"case_id": case["case_id"], "attempt_id": attempt_id, "case": case, "event": "case_summary",
                   "status": "failed_worker_exit", "exit_code": process.returncode,
                   "phase": phase, "completed_repeats": completed_repeats,
                   "median_seconds": None, "stderr_path": str(error_path)}
        append_event(events, summary)
    return summary


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    lines = path.read_text().splitlines()
    for number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # An interrupted final write is ignorable; a corrupt earlier event is not.
            if number != len(lines):
                raise
    return records


def write_csv_files(output: Path) -> None:
    records = read_events(output / "events.jsonl")
    summaries = {record["case_id"]: record for record in records
                 if record.get("event") == "case_summary"}
    columns = (
        "case_id", "attempt_id", "status", "algorithm", "size", "sources", "source_family", "seed",
        "repeats", "completed_repeats", "median_seconds", "minimum_seconds", "maximum_seconds",
        "pilot_seconds", "warmup_seconds", "union_nats", "lower_bound_nats", "upper_bound_nats",
        "gap_nats", "feasibility_residual", "iterations", "method", "retained_sources", "support",
        "master_seconds", "pricing_seconds", "clarabel_max_threads", "solver_threads",
        "linear_solver_threads", "tolerance", "implementation_commit",
        "implementation_fingerprint", "phase", "error_type", "error",
    )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in summaries.values():
            flat = {**record.get("case", {}), **record}
            writer.writerow({key: flat.get(key) for key in columns})
    raw_columns = (
        "case_id", "attempt_id", "algorithm", "size", "sources", "repeat", "seconds", "calls_per_repeat",
        "union_nats", "gap_nats", "feasibility_residual", "iterations", "method",
        "retained_sources", "support", "master_seconds", "pricing_seconds",
        "clarabel_max_threads", "solver_threads", "linear_solver_threads",
    )
    with (output / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_columns)
        writer.writeheader()
        for record in records:
            if record.get("event") == "repeat":
                flat = {**record.get("case", {}), **record}
                writer.writerow({key: flat.get(key) for key in raw_columns})


def int_list(values: list[str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    result = tuple(sorted(set(int(part) for token in values for part in token.split(","))))
    if not result or min(result) < 1:
        raise ValueError("sizes and counts must be positive integers")
    return result


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return worker(json.loads(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=("target", "sources", "source", "both"), default="both")
    parser.add_argument("--sizes", nargs="+", help="alphabet sizes for a single selected algorithm")
    parser.add_argument("--target-sizes", nargs="+", help="default: 2 8 32 128")
    parser.add_argument("--source-sizes", nargs="+", help="default: 2 3 4 8")
    parser.add_argument("--counts", nargs="+", help="source counts, as space- or comma-separated integers")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--source-family", choices=("independent", "collinear"), default="independent")
    parser.add_argument("--case-budget-seconds", type=float, default=60.0)
    parser.add_argument("--total-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--max-input-mib", type=float, default=2048.0)
    parser.add_argument("--continue-after-censor", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", default=cpu_name())
    parser.add_argument("--output", "--output-dir", type=Path, default=REPO / "benchmarks" / "results" / "union")
    args = parser.parse_args()
    if args.algorithm == "source":
        args.algorithm = "sources"
    if args.sizes is not None and args.algorithm == "both":
        parser.error("--sizes requires --algorithm target or sources; otherwise use --target-sizes/--source-sizes")
    if (args.repeats < 1 or args.tolerance <= 0 or args.max_iterations < 1
            or min(args.case_budget_seconds, args.total_budget_seconds, args.max_input_mib) <= 0):
        parser.error("repeat counts, tolerances, iteration limits, and budgets must be positive")
    try:
        counts = int_list(args.counts, COUNTS)
        target_sizes = int_list(args.sizes if args.algorithm == "target" else args.target_sizes, TARGET_SIZES)
        source_sizes = int_list(args.sizes if args.algorithm == "sources" else args.source_sizes, SOURCE_SIZES)
    except ValueError as error:
        parser.error(str(error))
    if min(target_sizes + source_sizes) < 2:
        parser.error("benchmark alphabet sizes must be at least two")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    event_path = args.output / "events.jsonl"
    if event_path.exists() and event_path.stat().st_size and not args.resume:
        parser.error(f"{event_path} already contains results; choose another output or pass --resume")
    impl = implementation()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now().astimezone().isoformat(),
        **impl, "hardware": {"cpu": args.cpu, "platform": platform.platform(),
                              "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "versions": versions(), "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "native_pool_introspection": "Recorded after warmup, once lazy solver dependencies are loaded; null explicitly means threadpoolctl is unavailable. Environment settings and solver thread settings are separate configuration records.",
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration": config, "counts": counts, "target_sizes": target_sizes, "source_sizes": source_sizes,
        "timing_protocol": "one warmup and one pilot excluded; each repeat is exactly one public solver call; median only if all repeats succeed",
        "target_input": "uniform prior; independent uniform-Dirichlet conditional rows; uint32 quantization with row total 2^30",
        "source_input": ("uniform prior; independent Bernoulli conditional probabilities uniform on [0.05,0.95]; uint32 quantization with row total 2^30"
                         if args.source_family == "independent" else
                         "uniform prior; exact asymmetric collinear posterior segments; integer conditional weights matching original redundancy protocol"),
        "comparison_note": "Default source-union input families differ from the collinear source-redundancy benchmark; timings do not establish a like-for-like algorithm comparison.",
        "output_flags": {"return_channel": False, "return_garblings": False},
        "budget_policy": "warmup/pilot excluded from measurements but included in wall budget; censor before repeats if pilot forecasts over-budget; hard timeout kills worker; subsequent larger cases on same curve skipped unless explicitly enabled",
    }
    # Each invocation keeps its own metadata; metadata.json is the latest invocation.
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    for filename in ("metadata.json", f"metadata-{stamp}.json"):
        (args.output / filename).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    # Recover only an incomplete final journal record before appending. Earlier
    # malformed records remain errors, protecting existing results from overwrite.
    if args.resume and event_path.exists():
        raw = event_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            last_start = raw.rfind(b"\n") + 1
            try:
                json.loads(raw[last_start:])
            except json.JSONDecodeError:
                with event_path.open("r+b") as recovery:
                    recovery.truncate(last_start)
            else:
                with event_path.open("ab") as recovery:
                    recovery.write(b"\n")
    old_records = read_events(event_path)
    done = {record["case_id"] for record in old_records
            if record.get("event") == "case_summary" and record.get("status") == "ok"}
    algorithms = ("target", "sources") if args.algorithm == "both" else (args.algorithm,)
    started = perf_counter()
    try:
        with event_path.open("a") as events:
            append_event(events, {"event": "run_started", **metadata})
            for algorithm in algorithms:
                for size in target_sizes if algorithm == "target" else source_sizes:
                    blocked_by = None
                    for count in counts:
                        case = {
                            "algorithm": algorithm, "size": size, "sources": count,
                            "source_family": args.source_family if algorithm == "sources" else "dirichlet",
                            "seed": args.seed, "repeats": args.repeats, "tolerance": args.tolerance,
                            "max_iterations": args.max_iterations,
                            "implementation_commit": impl["implementation_commit"],
                            "implementation_fingerprint": impl["implementation_fingerprint"],
                        }
                        case["case_id"] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]
                        if case["case_id"] in done:
                            print(f"resume {algorithm:7s} size={size:3d} k={count:7d}: completed", flush=True)
                            continue
                        remaining = args.total_budget_seconds - (perf_counter() - started)
                        input_bytes = count * size * 2 * np.dtype(np.uint32).itemsize
                        reason = None
                        if remaining <= 0:
                            reason = "skipped_total_budget"
                        elif input_bytes > args.max_input_mib * 1024 ** 2:
                            reason = "skipped_input_memory"
                        elif blocked_by is not None and not args.continue_after_censor:
                            reason = "skipped_curve_budget_policy"
                        if reason is not None:
                            summary = {"case_id": case["case_id"], "case": case, "event": "case_summary",
                                       "status": reason, "median_seconds": None, "input_bytes": input_bytes,
                                       "blocked_by_case_id": blocked_by}
                            append_event(events, summary)
                        else:
                            summary = run_case(case, args, events, min(args.case_budget_seconds, remaining))
                            if summary["status"] != "ok":
                                blocked_by = case["case_id"]
                        status = summary["status"]
                        elapsed = summary.get("median_seconds")
                        timing = f"{elapsed:.6g} s" if elapsed is not None else status
                        print(f"{algorithm:7s} size={size:3d} k={count:7d}: {timing}", flush=True)
                        write_csv_files(args.output)
            append_event(events, {"event": "run_finished", "seconds": perf_counter() - started})
    finally:
        write_csv_files(args.output)
    print(f"Wrote {args.output / 'summary.csv'}, timings.csv, events.jsonl, and metadata.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
