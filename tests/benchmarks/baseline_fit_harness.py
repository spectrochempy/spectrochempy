# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Reproducible local performance harness for the public ``Baseline`` fit path.

This script is intentionally opt-in developer tooling:

- it is not collected by pytest;
- it introduces no CI timing threshold;
- it does not change runtime behavior;
- it can emit lightweight timing summaries and optional cProfile output.

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
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    factory: Callable[[], NDDataset]
    baseline_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingStat:
    calls: int = 0
    total_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.calls += 1
        self.total_s += elapsed_s


def _make_probe_dataset(
    *,
    n_points: int = 4096,
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


def _default_scenarios() -> dict[str, Scenario]:
    return {
        "poly1d": Scenario(
            name="poly1d",
            model="polynomial",
            factory=lambda: _make_probe_dataset(n_points=4096, n_rows=None),
            baseline_kwargs={"order": 3, "include_limits": False},
            fit_kwargs={"ranges": [[1000.0, 1125.0], [1875.0, 2000.0]]},
        ),
        "poly2d": Scenario(
            name="poly2d",
            model="polynomial",
            factory=lambda: _make_probe_dataset(n_points=4096, n_rows=32),
            baseline_kwargs={"order": 3, "include_limits": False},
            fit_kwargs={"ranges": [[1000.0, 1125.0], [1875.0, 2000.0]]},
        ),
        "asls1d": Scenario(
            name="asls1d",
            model="asls",
            factory=lambda: _make_probe_dataset(n_points=4096, n_rows=None),
            baseline_kwargs={"lamb": 1e5, "asymmetry": 0.05},
        ),
        "snip1d": Scenario(
            name="snip1d",
            model="snip",
            factory=lambda: _make_probe_dataset(n_points=4096, n_rows=None),
            baseline_kwargs={"snip_width": 40},
        ),
        "rubberband1d": Scenario(
            name="rubberband1d",
            model="rubberband",
            factory=lambda: _make_probe_dataset(
                n_points=4096,
                n_rows=None,
                descending=True,
            ),
        ),
        "asls1d_masked": Scenario(
            name="asls1d_masked",
            model="asls",
            factory=lambda: _make_probe_dataset(
                n_points=4096,
                n_rows=None,
                masked=True,
            ),
            baseline_kwargs={"lamb": 1e5, "asymmetry": 0.05},
        ),
    }


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


def _timed_fit(
    scenario: Scenario,
    *,
    repeats: int,
    warmup: int,
    profile_top: int,
) -> dict[str, Any]:
    fit_times: list[float] = []
    measured_stats: dict[str, TimingStat] = defaultdict(TimingStat)
    profile_output = None
    dataset = scenario.factory()

    for iteration in range(warmup + repeats):
        iteration_stats: dict[str, TimingStat] = defaultdict(TimingStat)
        blc = Baseline(model=scenario.model, **scenario.baseline_kwargs)
        if "ranges" in scenario.fit_kwargs:
            blc.ranges = scenario.fit_kwargs["ranges"]

        with ExitStack() as stack:
            _patch_timed_method(
                iteration_stats,
                NDDataset,
                "copy",
                "nddataset.copy",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                NDDataset,
                "sort",
                "nddataset.sort",
                stack,
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
                Baseline,
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
            _patch_timed_method(
                iteration_stats,
                np,
                "interp",
                "numpy.interp",
                stack,
            )
            _patch_timed_method(
                iteration_stats,
                baseline_module,
                "ConvexHull",
                "baseline.ConvexHull",
                stack,
            )

            profiler = cProfile.Profile() if profile_top and iteration == warmup else None
            if profiler is not None:
                profiler.enable()

            start = perf_counter()
            blc.fit(dataset)
            elapsed = perf_counter() - start

            if profiler is not None:
                profiler.disable()
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(
                    profile_top
                )
                profile_output = stream.getvalue()

        if iteration >= warmup:
            fit_times.append(elapsed)
            for label, stat in iteration_stats.items():
                measured_stats[label].calls += stat.calls
                measured_stats[label].total_s += stat.total_s

    instrumented_total_s = sum(stat.total_s for stat in measured_stats.values())
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
                "share_of_fit_percent": (
                    100.0 * stat.total_s / fit_total_s if fit_total_s else 0.0
                ),
            }
        )

    result = {
        "scenario": scenario.name,
        "model": scenario.model,
        "shape": tuple(int(v) for v in dataset.shape),
        "repeats": repeats,
        "warmup": warmup,
        "mean_fit_ms": 1000.0 * float(np.mean(fit_times)),
        "median_fit_ms": 1000.0 * float(np.median(fit_times)),
        "min_fit_ms": 1000.0 * float(np.min(fit_times)),
        "max_fit_ms": 1000.0 * float(np.max(fit_times)),
        "inclusive_instrumented_share_percent": (
            100.0 * instrumented_total_s / fit_total_s if fit_total_s else 0.0
        ),
        "components": components,
    }
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
        print(  # noqa: T201
            f"[{result['scenario']}] model={result['model']} shape={result['shape']} "
            f"median={result['median_fit_ms']:.2f} ms "
            f"mean={result['mean_fit_ms']:.2f} ms "
            f"inclusive-instrumented={result['inclusive_instrumented_share_percent']:.1f}%"
        )
        for component in result["components"][:8]:
            print(  # noqa: T201
                "  - "
                f"{component['label']}: {component['total_ms']:.2f} ms total "
                f"over {component['calls']} call(s) "
                f"({component['share_of_fit_percent']:.1f}% of fit time)"
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
        help="Measured repetitions per scenario (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up repetitions per scenario (default: 1).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=0,
        help="Include top cumulative cProfile rows for the first measured run.",
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
        _timed_fit(
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
