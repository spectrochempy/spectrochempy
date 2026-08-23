# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Reproducible local performance harness for the public ``Baseline`` API.

This script is intentionally opt-in developer tooling:

- it is not collected by pytest;
- it introduces no CI timing threshold;
- it does not change runtime behavior;
- it separates absolute timing, instrumented breakdown, and cProfile passes.

Example:
-------
micromamba run -n scpy-core python tests/benchmarks/baseline_fit_harness.py
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field
from time import perf_counter
from typing import Any

import numpy as np

import spectrochempy as scp
import spectrochempy.processing.baselineprocessing.baselineprocessing as baseline_module
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    factory: Callable[[], NDDataset]
    baseline_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class TimingStat:
    calls: int = 0
    total_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.calls += 1
        self.total_s += elapsed_s


def _make_probe_dataset(
    *,
    n_points: int,
    n_rows: int | None = None,
    descending: bool = False,
    masked: bool = False,
) -> NDDataset:
    x = np.linspace(1000.0, 2000.0, n_points)
    if descending:
        x = x[::-1]

    base = 0.002 * x + 0.5 + 0.08 * np.sin(np.linspace(0.0, 24.0, n_points))
    peak1 = 0.4 * np.exp(-((x - 1250.0) ** 2) / (2.0 * 35.0**2))
    peak2 = 0.7 * np.exp(-((x - 1710.0) ** 2) / (2.0 * 60.0**2))
    profile = base + peak1 + peak2

    if n_rows is None:
        data = profile
        coordset = [scp.Coord(x, title="wavenumber", units="cm^-1")]
        name = f"baseline_perf_1d_{n_points}"
    else:
        data = np.vstack([profile + 0.01 * i for i in range(n_rows)])
        coordset = [
            scp.Coord(np.arange(n_rows, dtype=float), title="row", units=None),
            scp.Coord(x, title="wavenumber", units="cm^-1"),
        ]
        name = f"baseline_perf_{n_rows}x{n_points}"

    dataset = scp.NDDataset(
        data=data,
        coordset=coordset,
        units="absorbance",
        title="baseline performance probe",
    )
    dataset.name = name
    if masked:
        dataset[1400.0:1500.0] = scp.MASKED
    return dataset


def _edge_ranges() -> list[list[float]]:
    return [[1000.0, 1125.0], [1875.0, 2000.0]]


def _default_scenarios() -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}

    for n_points in (1024, 4096, 16384):
        scenarios[f"poly1d_n{n_points}"] = Scenario(
            name=f"poly1d_n{n_points}",
            model="polynomial",
            factory=lambda n_points=n_points: _make_probe_dataset(n_points=n_points),
            baseline_kwargs={"order": 3, "include_limits": False},
            fit_kwargs={"ranges": _edge_ranges()},
            notes="strict 1D polynomial with explicit edge support ranges",
        )
        scenarios[f"asls1d_n{n_points}"] = Scenario(
            name=f"asls1d_n{n_points}",
            model="asls",
            factory=lambda n_points=n_points: _make_probe_dataset(n_points=n_points),
            baseline_kwargs={"lamb": 1e5, "asymmetry": 0.05},
            notes="strict 1D AsLS",
        )

    for n_rows in (1, 8, 32):
        scenarios[f"poly2d_m{n_rows}_n4096"] = Scenario(
            name=f"poly2d_m{n_rows}_n4096",
            model="polynomial",
            factory=lambda n_rows=n_rows: _make_probe_dataset(
                n_points=4096,
                n_rows=n_rows,
            ),
            baseline_kwargs={"order": 3, "include_limits": False},
            fit_kwargs={"ranges": _edge_ranges()},
            notes="2D polynomial with explicit edge support ranges",
        )

    scenarios["snip1d_n4096"] = Scenario(
        name="snip1d_n4096",
        model="snip",
        factory=lambda: _make_probe_dataset(n_points=4096),
        baseline_kwargs={"snip_width": 40},
        notes="strict 1D SNIP",
    )
    scenarios["rubberband1d_n4096_desc"] = Scenario(
        name="rubberband1d_n4096_desc",
        model="rubberband",
        factory=lambda: _make_probe_dataset(n_points=4096, descending=True),
        notes="strict descending 1D rubberband",
    )
    scenarios["asls1d_masked_n4096"] = Scenario(
        name="asls1d_masked_n4096",
        model="asls",
        factory=lambda: _make_probe_dataset(n_points=4096, masked=True),
        baseline_kwargs={"lamb": 1e5, "asymmetry": 0.05},
        notes="strict masked 1D AsLS",
    )
    return scenarios


def _build_baseline(scenario: Scenario) -> baseline_module.Baseline:
    blc = baseline_module.Baseline(
        model=scenario.model,
        **scenario.baseline_kwargs,
    )
    if "ranges" in scenario.fit_kwargs:
        blc.ranges = scenario.fit_kwargs["ranges"]
    return blc


def _patch_timed_method(
    stats: dict[str, TimingStat],
    obj: Any,
    attr: str,
    label: str,
    stack: ExitStack,
) -> None:
    original = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        start = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stats[label].add(perf_counter() - start)

    setattr(obj, attr, wrapped)
    stack.callback(setattr, obj, attr, original)


def _time_call(func: Callable[[], Any]) -> tuple[float, Any]:
    start = perf_counter()
    value = func()
    return perf_counter() - start, value


def _summary(values_s: list[float]) -> dict[str, float]:
    return {
        "mean_ms": 1000.0 * float(np.mean(values_s)),
        "median_ms": 1000.0 * float(np.median(values_s)),
        "min_ms": 1000.0 * float(np.min(values_s)),
        "max_ms": 1000.0 * float(np.max(values_s)),
    }


def _measure_uninstrumented_operations(
    scenario: Scenario,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    fit_times: list[float] = []
    transform_first_times: list[float] = []
    corrected_first_times: list[float] = []
    transform_repeat_times: list[float] = []
    corrected_repeat_times: list[float] = []
    shape = None

    for iteration in range(warmup + repeats):
        dataset = scenario.factory()
        shape = tuple(int(v) for v in dataset.shape)
        blc = _build_baseline(scenario)

        fit_elapsed, _ = _time_call(lambda blc=blc, dataset=dataset: blc.fit(dataset))
        transform_first_elapsed, transform_first = _time_call(blc.transform)
        corrected_first_elapsed, corrected_first = _time_call(
            lambda blc=blc: blc.corrected
        )
        transform_repeat_elapsed, transform_repeat = _time_call(blc.transform)
        corrected_repeat_elapsed, corrected_repeat = _time_call(
            lambda blc=blc: blc.corrected
        )

        assert transform_first.shape == corrected_first.shape == shape
        assert transform_repeat.shape == corrected_repeat.shape == shape

        if iteration >= warmup:
            fit_times.append(fit_elapsed)
            transform_first_times.append(transform_first_elapsed)
            corrected_first_times.append(corrected_first_elapsed)
            transform_repeat_times.append(transform_repeat_elapsed)
            corrected_repeat_times.append(corrected_repeat_elapsed)

    return {
        "shape": shape,
        "fit": _summary(fit_times),
        "transform_first": _summary(transform_first_times),
        "corrected_first": _summary(corrected_first_times),
        "transform_repeat": _summary(transform_repeat_times),
        "corrected_repeat": _summary(corrected_repeat_times),
    }


def _measure_instrumented_fit_breakdown(
    scenario: Scenario,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    fit_times: list[float] = []
    measured_stats: dict[str, TimingStat] = defaultdict(TimingStat)

    for iteration in range(warmup + repeats):
        dataset = scenario.factory()
        blc = _build_baseline(scenario)
        iteration_stats: dict[str, TimingStat] = defaultdict(TimingStat)

        with ExitStack() as stack:
            _patch_timed_method(
                iteration_stats, NDDataset, "copy", "nddataset.copy", stack
            )
            _patch_timed_method(
                iteration_stats, NDDataset, "sort", "nddataset.sort", stack
            )
            _patch_timed_method(
                iteration_stats,
                NDDataset,
                "remove_masks",
                "nddataset.remove_masks",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                Coord,
                "loc2index",
                "coord.loc2index",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                baseline_module,
                "concatenate",
                "baseline.concatenate",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                baseline_module,
                "trim_ranges",
                "baseline.trim_ranges",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                baseline_module.Baseline,
                "_fit",
                "baseline._fit",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                np.polynomial.polynomial,
                "polyfit",
                "numpy.polyfit",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                np.polynomial.polynomial,
                "polyval",
                "numpy.polyval",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                baseline_module,
                "spsolve",
                "baseline.spsolve",
                stack,
            )
            _patch_timed_method(iteration_stats, np, "interp", "numpy.interp", stack)
            _patch_timed_method(
                iteration_stats,
                baseline_module,
                "ConvexHull",
                "baseline.ConvexHull",
                stack,
            )

            fit_elapsed, _ = _time_call(
                lambda blc=blc, dataset=dataset: blc.fit(dataset)
            )

        if iteration >= warmup:
            fit_times.append(fit_elapsed)
            for label, stat in iteration_stats.items():
                measured_stats[label].calls += stat.calls
                measured_stats[label].total_s += stat.total_s

    fit_total_s = sum(fit_times)
    components = []
    for label, stat in sorted(
        measured_stats.items(),
        key=lambda item: item[1].total_s,
        reverse=True,
    ):
        components.append(
            {
                "label": label,
                "calls": stat.calls,
                "total_ms": 1000.0 * stat.total_s,
                "share_of_instrumented_fit_percent": (
                    100.0 * stat.total_s / fit_total_s if fit_total_s else 0.0
                ),
            }
        )

    return {
        "fit": _summary(fit_times),
        "components": components,
        "inclusive_component_share_percent": (
            100.0 * sum(stat.total_s for stat in measured_stats.values()) / fit_total_s
            if fit_total_s
            else 0.0
        ),
    }


def _profile_fit(
    scenario: Scenario,
    *,
    profile_top: int,
) -> str | None:
    if profile_top <= 0:
        return None

    dataset = scenario.factory()
    blc = _build_baseline(scenario)
    profiler = cProfile.Profile()
    profiler.enable()
    blc.fit(dataset)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(
        profile_top
    )
    return stream.getvalue()


def _characterize_scenario(
    scenario: Scenario,
    *,
    repeats: int,
    warmup: int,
    profile_top: int,
) -> dict[str, Any]:
    absolute = _measure_uninstrumented_operations(
        scenario,
        repeats=repeats,
        warmup=warmup,
    )
    instrumented_fit = _measure_instrumented_fit_breakdown(
        scenario,
        repeats=repeats,
        warmup=warmup,
    )
    result = {
        "scenario": scenario.name,
        "model": scenario.model,
        "notes": scenario.notes,
        "shape": absolute["shape"],
        "repeats": repeats,
        "warmup": warmup,
        "absolute_timings": {
            "fit": absolute["fit"],
            "transform_first": absolute["transform_first"],
            "corrected_first": absolute["corrected_first"],
            "transform_repeat": absolute["transform_repeat"],
            "corrected_repeat": absolute["corrected_repeat"],
        },
        "instrumented_fit_breakdown": instrumented_fit,
    }
    profile_output = _profile_fit(scenario, profile_top=profile_top)
    if profile_output is not None:
        result["cprofile_top_cumulative"] = profile_output
    return result


def _select_scenarios(
    all_scenarios: dict[str, Scenario],
    selected: list[str] | None,
) -> list[Scenario]:
    if not selected:
        return list(all_scenarios.values())

    missing = [name for name in selected if name not in all_scenarios]
    if missing:
        choices = ", ".join(sorted(all_scenarios))
        raise SystemExit(f"Unknown scenario(s): {missing}. Known scenarios: {choices}")
    return [all_scenarios[name] for name in selected]


def _print_text_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        absolute = result["absolute_timings"]
        instrumented = result["instrumented_fit_breakdown"]
        print(  # noqa: T201
            f"[{result['scenario']}] model={result['model']} shape={result['shape']} "
            f"fit median={absolute['fit']['median_ms']:.2f} ms "
            f"instrumented-fit median={instrumented['fit']['median_ms']:.2f} ms"
        )
        print(  # noqa: T201
            "  absolute timings: "
            f"transform1={absolute['transform_first']['median_ms']:.2f} ms, "
            f"corrected1={absolute['corrected_first']['median_ms']:.2f} ms, "
            f"transform2={absolute['transform_repeat']['median_ms']:.2f} ms, "
            f"corrected2={absolute['corrected_repeat']['median_ms']:.2f} ms"
        )
        for component in instrumented["components"][:8]:
            print(  # noqa: T201
                "  - "
                f"{component['label']}: {component['total_ms']:.2f} ms total "
                f"over {component['calls']} call(s) "
                f"({component['share_of_instrumented_fit_percent']:.1f}% of "
                "instrumented fit time)"
            )
        if "cprofile_top_cumulative" in result:
            print("  cProfile (top cumulative):")  # noqa: T201
            for line in result["cprofile_top_cumulative"].splitlines():
                print(f"    {line}")  # noqa: T201
        print()  # noqa: T201


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario name to run. Repeat to select multiple scenarios.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured repetitions per pass and per scenario (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up repetitions per pass and per scenario (default: 1).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=0,
        help="Include top cumulative cProfile rows from a separate fit-only pass.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = _select_scenarios(_default_scenarios(), args.scenario)
    results = [
        _characterize_scenario(
            scenario,
            repeats=args.repeats,
            warmup=args.warmup,
            profile_top=args.profile_top,
        )
        for scenario in scenarios
    ]
    if args.json:
        print(json.dumps(results, indent=2))  # noqa: T201
        return
    _print_text_summary(results)


if __name__ == "__main__":
    main()
