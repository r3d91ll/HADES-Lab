"""Utilities for recording Conveyance-aligned benchmark metrics.

The helpers in this module provide a reusable API for translating raw
benchmark artefacts (e.g. JSON emitted by
``tests/benchmarks/arango_connection_test.py``) into structured
Conveyance records.  Each record captures the efficiency and capability
views, the contributing factors (W/R/H/T), the context components
(L/I/A/G), and a zero-propagation gate, making it easy to maintain
Consistent logs across scripts and workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping
import json


TIME_UNITS: Mapping[str, float] = {
    "seconds": 1.0,
    "s": 1.0,
    "milliseconds": 1e3,
    "ms": 1e3,
    "microseconds": 1e6,
    "us": 1e6,
}


@dataclass(frozen=True)
class ConveyanceContext:
    """Context components and weights for the Conveyance calculation."""

    L: float
    I: float
    A: float
    G: float
    weight_L: float = 0.25
    weight_I: float = 0.25
    weight_A: float = 0.25
    weight_G: float = 0.25

    def weighted_sum(self) -> float:
        """Return the scalar Ctx value (wL·L + wI·I + wA·A + wG·G)."""

        return (
            self.weight_L * self.L
            + self.weight_I * self.I
            + self.weight_A * self.A
            + self.weight_G * self.G
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the context components, weights, and total Ctx."""

        return {
            "components": {"L": self.L, "I": self.I, "A": self.A, "G": self.G},
            "weights": {
                "L": self.weight_L,
                "I": self.weight_I,
                "A": self.weight_A,
                "G": self.weight_G,
            },
            "Ctx": self.weighted_sum(),
        }


def load_metric(
    input_path: Path,
    benchmark_key: str,
    time_source: str,
    time_metric: str,
) -> float:
    """Extract a latency metric from a benchmark JSON artefact."""

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    benchmark: MutableMapping[str, Any] | None = data.get(benchmark_key)
    if benchmark is None:
        raise KeyError(f"Benchmark key '{benchmark_key}' not found in {input_path}")

    stats = benchmark.get("stats")
    if stats is None:
        raise KeyError(f"Stats not present under '{benchmark_key}' in {input_path}")

    source = stats.get(time_source)
    if source is None:
        raise KeyError(
            f"Time source '{time_source}' missing under stats for '{benchmark_key}'"
        )

    if time_metric not in source:
        raise KeyError(
            f"Metric '{time_metric}' missing under '{time_source}' in {input_path}"
        )

    return float(source[time_metric])


def compute_conveyance(
    what: float,
    where: float,
    who: float,
    time_seconds: float,
    ctx_value: float,
    alpha: float,
) -> Dict[str, Any]:
    """Return efficiency/capability Conveyance values and zero-gate."""

    zero_gate = (
        what <= 0
        or where <= 0
        or who <= 0
        or time_seconds <= 0
        or ctx_value <= 0
    )
    if zero_gate:
        return {
            "conveyance_efficiency": 0.0,
            "conveyance_capability": 0.0,
            "zero_propagation": True,
        }

    efficiency = (what * where * who / time_seconds) * (ctx_value**alpha)
    capability = (what * where * who) * (ctx_value**alpha)
    return {
        "conveyance_efficiency": efficiency,
        "conveyance_capability": capability,
        "zero_propagation": False,
    }


def build_record(
    *,
    input_path: Path,
    label: str,
    benchmark_key: str,
    time_source: str,
    time_metric: str,
    time_units: str,
    what: float,
    where: float,
    who: float,
    context: ConveyanceContext,
    alpha: float,
    notes: str = "",
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Construct a structured Conveyance record."""

    raw_value = load_metric(input_path, benchmark_key, time_source, time_metric)
    scale = TIME_UNITS.get(time_units)
    if scale is None:
        raise ValueError(f"Unsupported time unit '{time_units}'")

    time_seconds = raw_value / scale if scale != 0 else raw_value
    time_seconds = max(time_seconds, 1e-9)

    ctx_value = context.weighted_sum()
    conveyance_payload = compute_conveyance(
        what=what,
        where=where,
        who=who,
        time_seconds=time_seconds,
        ctx_value=ctx_value,
        alpha=alpha,
    )

    record = {
        "label": label,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "input": str(input_path),
        "benchmark_key": benchmark_key,
        "time_source": time_source,
        "time_metric": time_metric,
        "time_units": time_units,
        "time_value": raw_value,
        "time_seconds": time_seconds,
        "what": what,
        "where": where,
        "who": who,
        "alpha": alpha,
        "context": context.as_dict(),
        "notes": notes,
    }
    record.update(conveyance_payload)
    return record


def append_record(output_path: Path, record: Mapping[str, Any]) -> None:
    """Append a record to a JSONL file."""

    with output_path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False)
        fh.write("\n")


def log_conveyance(
    *,
    input_path: Path,
    label: str,
    benchmark_key: str,
    time_source: str,
    time_metric: str,
    time_units: str,
    what: float,
    where: float,
    who: float,
    context: ConveyanceContext,
    alpha: float = 1.7,
    notes: str = "",
    output_path: Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Build (and optionally persist) a Conveyance benchmark record."""

    record = build_record(
        input_path=input_path,
        label=label,
        benchmark_key=benchmark_key,
        time_source=time_source,
        time_metric=time_metric,
        time_units=time_units,
        what=what,
        where=where,
        who=who,
        context=context,
        alpha=alpha,
        notes=notes,
    )

    if output_path is not None and not dry_run:
        append_record(output_path, record)

    return record


__all__ = [
    "ConveyanceContext",
    "TIME_UNITS",
    "build_record",
    "compute_conveyance",
    "load_metric",
    "log_conveyance",
]
