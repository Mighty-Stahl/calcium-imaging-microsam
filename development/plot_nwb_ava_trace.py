#!/usr/bin/env python3
"""Extract and plot AVA/AVB calcium activity aligned to OP50 stimulus from NWB.

Pipeline:
- Find the first timepoint where the stimulus label contains the keyword (default: OP50).
- Extract a window from -pre_s .. +post_s seconds around that first occurrence ("cycle 1").
- Average left/right for each class:
    - AVA = mean(AVAL, AVAR)
    - AVB = mean(AVBL, AVBR)
- Either:
    - plot one NWB file (mean AVA/AVB for that animal), or
    - scan a directory of NWB files and plot sex-separated means.

Sex grouping when scanning a directory:
- If filename contains an "h" token (e.g. "-h2"), it is treated as hermaphrodite.
- If filename contains an "m" token (e.g. "-m3"), it is treated as male.
- Files where sex cannot be inferred are skipped.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NwbExtract:
    path: Path
    sex: str | None  # 'herm' | 'male' | None
    t0_s: float
    rel_t_s: np.ndarray  # common relative time axis
    aval: np.ndarray | None
    avar: np.ndarray | None
    avbl: np.ndarray | None
    avbr: np.ndarray | None
    ava_raw: np.ndarray
    avb_raw: np.ndarray


_H_TOKEN = re.compile(r"(^|[-_])h\d*([-_]|$)", re.IGNORECASE)
_M_TOKEN = re.compile(r"(^|[-_])m\d*([-_]|$)", re.IGNORECASE)


def _unwrap_one(value):
    """Unwrap (1,) object arrays or 1-element pandas Series to a scalar."""
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.Series):
            if len(value) == 1:
                return value.iloc[0]
            return value
    except Exception:
        pass

    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == (1,):
        return value[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _infer_sex_from_filename(path: Path) -> str | None:
    name = path.name
    if _H_TOKEN.search(name):
        return "herm"
    if _M_TOKEN.search(name):
        return "male"

    lowered = name.lower()
    if "herm" in lowered:
        return "herm"
    if "male" in lowered:
        return "male"

    return None


def _compress_stimulus_events(stim_events: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Keep only points where the label changes (reduces clutter)."""
    if not stim_events:
        return []
    out: list[tuple[float, str]] = []
    prev = None
    for t, label in sorted(stim_events, key=lambda x: x[0]):
        if prev is None or label != prev:
            out.append((t, label))
            prev = label
    return out


def _find_first_stimulus_time(stim_events: list[tuple[float, str]], keyword: str) -> float | None:
    kw = (keyword or "").strip().lower()
    if not kw:
        return None
    for t, label in _compress_stimulus_events(stim_events):
        if kw in str(label).lower():
            return float(t)
    return None


def _build_time_axis(n: int, *, rate_hz: float, starting_time_s: float) -> np.ndarray:
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError(f"Invalid sampling rate: {rate_hz!r}")
    return starting_time_s + np.arange(n, dtype=float) / rate_hz


def _interp_window(
    *,
    t_s: np.ndarray,
    y: np.ndarray,
    t0_s: float,
    rel_t_s: np.ndarray,
) -> np.ndarray:
    """Interpolate y(t) onto (t0 + rel_t) while ignoring NaNs."""
    y = np.asarray(y, dtype=float)
    if y.shape != t_s.shape:
        raise ValueError(f"Time/trace shape mismatch: t{t_s.shape} vs y{y.shape}")

    finite = np.isfinite(y) & np.isfinite(t_s)
    if finite.sum() < 2:
        return np.full(rel_t_s.shape, np.nan, dtype=float)

    x = t_s[finite]
    v = y[finite]

    # target absolute times
    xq = t0_s + rel_t_s

    # out-of-range -> NaN (np.interp would extrapolate)
    out = np.full(rel_t_s.shape, np.nan, dtype=float)
    in_range = (xq >= x.min()) & (xq <= x.max())
    if in_range.any():
        out[in_range] = np.interp(xq[in_range], x, v)
    return out


def _mean_traces(traces: Iterable[np.ndarray]) -> np.ndarray:
    arrs = [np.asarray(t, dtype=float) for t in traces if t is not None]
    if not arrs:
        raise ValueError("No traces to average")
    if len(arrs) == 1:
        return arrs[0]
    return np.nanmean(np.stack(arrs, axis=0), axis=0)


def _compute_dff0(
    *,
    rel_t_s: np.ndarray,
    trace: np.ndarray,
    baseline_window_s: tuple[float, float] = (-3.5, 0.0),
    response_window_s: tuple[float, float] = (0.0, 15.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ΔF/F0 from a stimulus-aligned trace.

    - F0 is the mean over rel_t in [baseline_start, baseline_end)
    - Ft is taken over rel_t in [response_start, response_end]
    """
    rel_t_s = np.asarray(rel_t_s, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if rel_t_s.shape != trace.shape:
        raise ValueError(f"rel_t/trace shape mismatch: {rel_t_s.shape} vs {trace.shape}")

    b0, b1 = baseline_window_s
    r0, r1 = response_window_s

    baseline_mask = (rel_t_s >= b0) & (rel_t_s < b1)
    response_mask = (rel_t_s >= r0) & (rel_t_s <= r1)

    baseline_vals = trace[baseline_mask]
    f0 = float(np.nanmean(baseline_vals)) if baseline_vals.size else np.nan
    if not np.isfinite(f0) or f0 == 0:
        t_resp = rel_t_s[response_mask]
        return t_resp, np.full(t_resp.shape, np.nan, dtype=float)

    t_resp = rel_t_s[response_mask]
    ft = trace[response_mask]
    return t_resp, (ft - f0) / f0


def _fmt_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3g}"


def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope from y = a*x + b via least squares, NaN-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")

    xv = x[m]
    yv = y[m]
    A = np.column_stack([xv, np.ones_like(xv)])
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    return float(coef[0])


def _auc_trapz(x: np.ndarray, y: np.ndarray, *, subtract: float | None = None) -> float:
    """Compute trapezoidal AUC, NaN-safe.

    If subtract is provided, integrates (y - subtract).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")

    xv = x[m]
    yv = y[m]
    if subtract is not None:
        yv = yv - float(subtract)
    return float(np.trapezoid(yv, xv))


def _extract_needed_traces(nwb_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Extract neuron->activity plus stimulus + timebase metadata."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()

        if "CalciumActivity" not in nwb.processing:
            raise KeyError("NWB has no processing module 'CalciumActivity'")
        mod = nwb.processing["CalciumActivity"]

        if "ActivityTraces" not in mod.data_interfaces:
            raise KeyError("CalciumActivity has no data interface 'ActivityTraces'")
        tbl = mod.data_interfaces["ActivityTraces"]

        required_cols = {"neuron", "activity"}
        if not hasattr(tbl, "colnames") or not required_cols.issubset(set(tbl.colnames)):
            raise KeyError("ActivityTraces missing required columns: neuron, activity")

        needed = {"AVAL", "AVAR", "AVBL", "AVBR"}
        out: dict[str, np.ndarray] = {}
        for i in range(len(tbl)):
            row = tbl[i]
            neuron = str(_unwrap_one(row["neuron"]))
            key = neuron.strip().upper()
            if key not in needed:
                continue
            activity = _unwrap_one(row["activity"])
            out[key] = np.asarray(activity, dtype=float)

        # timebase
        series = nwb.acquisition.get("CalciumImageSeries", None)
        if series is None:
            raise KeyError("NWB has no acquisition 'CalciumImageSeries'")
        rate = float(getattr(series, "rate", np.nan))
        starting_time = float(getattr(series, "starting_time", 0.0))

        # stimulus annotations
        stim = mod.data_interfaces.get("StimulusInfo", None)
        stim_events: list[tuple[float, str]] = []
        if stim is not None and hasattr(stim, "timestamps") and hasattr(stim, "data"):
            try:
                timestamps = np.asarray(stim.timestamps, dtype=float)
                labels = [str(x) for x in stim.data[:]]
                stim_events = list(zip(timestamps.tolist(), labels))
            except Exception:
                stim_events = []

        meta = {
            "rate_hz": rate,
            "starting_time_s": starting_time,
            "stim_events": stim_events,
        }

    return out, meta


def _extract_aligned(nwb_path: Path, *, keyword: str, rel_t_s: np.ndarray) -> NwbExtract | None:
    traces, meta = _extract_needed_traces(nwb_path)

    t0_s = _find_first_stimulus_time(meta.get("stim_events", []), keyword=keyword)
    if t0_s is None:
        print(f"[skip] {nwb_path.name}: no stimulus matching {keyword!r}")
        return None

    rate_hz = float(meta["rate_hz"])
    starting_time_s = float(meta["starting_time_s"])

    # Build time axis for each trace length.
    # Interpolate each neuron trace separately (handles slight length differences).
    def aligned(neuron: str) -> np.ndarray | None:
        y = traces.get(neuron)
        if y is None:
            return None
        t_s = _build_time_axis(len(y), rate_hz=rate_hz, starting_time_s=starting_time_s)
        return _interp_window(t_s=t_s, y=y, t0_s=t0_s, rel_t_s=rel_t_s)

    aval = aligned("AVAL")
    avar = aligned("AVAR")
    avbl = aligned("AVBL")
    avbr = aligned("AVBR")

    if aval is None and avar is None:
        print(f"[skip] {nwb_path.name}: missing AVAL/AVAR")
        return None
    if avbl is None and avbr is None:
        print(f"[skip] {nwb_path.name}: missing AVBL/AVBR")
        return None

    ava_raw = _mean_traces([t for t in (aval, avar) if t is not None])
    avb_raw = _mean_traces([t for t in (avbl, avbr) if t is not None])

    sex = _infer_sex_from_filename(nwb_path)

    return NwbExtract(
        path=nwb_path,
        sex=sex,
        t0_s=t0_s,
        rel_t_s=rel_t_s,
        aval=aval,
        avar=avar,
        avbl=avbl,
        avbr=avbr,
        ava_raw=ava_raw,
        avb_raw=avb_raw,
    )


def _scan_nwbs(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".nwb":
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted([p for p in root.rglob("*.nwb") if p.is_file()])


def _aggregate_by_sex(extracts: list[NwbExtract]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate traces by sex.

    Important: F/F0 is computed per organism first, then averaged.
    """
    out: dict[str, dict[str, np.ndarray]] = {}

    for sex in ("herm", "male"):
        subset = [e for e in extracts if e.sex == sex]
        if not subset:
            continue

        # Raw activity: average after alignment (already per organism)
        ava_stack = np.stack([e.ava_raw for e in subset], axis=0)
        avb_stack = np.stack([e.avb_raw for e in subset], axis=0)
        ava_raw_mean = np.nanmean(ava_stack, axis=0)
        avb_raw_mean = np.nanmean(avb_stack, axis=0)

        # ΔF/F0: compute per worm per neuron-side first, then compute metrics per side,
        # then combine L/R metrics within worm, and finally aggregate across worms.
        example_trace = None
        for cand in (subset[0].aval, subset[0].avar, subset[0].ava_raw):
            if cand is not None:
                example_trace = cand
                break
        if example_trace is None:
            raise ValueError("Unexpected: no AVA trace data available")

        t_dff, _ = _compute_dff0(rel_t_s=subset[0].rel_t_s, trace=example_trace)

        ava_dff_curves: list[np.ndarray] = []
        avb_dff_curves: list[np.ndarray] = []
        ava_mean_vals: list[float] = []
        ava_slope_vals: list[float] = []
        ava_auc_vals: list[float] = []
        avb_mean_vals: list[float] = []
        avb_slope_vals: list[float] = []
        avb_auc_vals: list[float] = []

        for e in subset:
            # per-side ΔF/F0 curves
            t_this: np.ndarray | None = None
            side_curves: dict[str, np.ndarray] = {}
            for name, trace in (
                ("AVAL", e.aval),
                ("AVAR", e.avar),
                ("AVBL", e.avbl),
                ("AVBR", e.avbr),
            ):
                if trace is None:
                    continue
                t_side, dff_side = _compute_dff0(rel_t_s=e.rel_t_s, trace=trace)
                if t_this is None:
                    t_this = t_side
                else:
                    if t_side.shape != t_this.shape or not np.allclose(t_side, t_this, equal_nan=True):
                        raise ValueError("Inconsistent per-worm ΔF/F0 grids; check --dt/pre/post settings")
                side_curves[name] = dff_side

            if t_this is None:
                raise ValueError("Unexpected: could not compute ΔF/F0 time axis")
            if t_this.shape != t_dff.shape or not np.allclose(t_this, t_dff, equal_nan=True):
                raise ValueError("Inconsistent ΔF/F0 time grids across worms; check --dt/pre/post settings")

            # per-side metrics
            def side_metrics(curve: np.ndarray | None) -> tuple[float, float, float]:
                if curve is None:
                    return float("nan"), float("nan"), float("nan")
                mean_v = float(np.nanmean(curve))
                slope_v = _linear_regression_slope(t_dff, curve)
                auc_v = _auc_trapz(t_dff, curve)
                return mean_v, slope_v, auc_v

            aval_m, aval_s, aval_a = side_metrics(side_curves.get("AVAL"))
            avar_m, avar_s, avar_a = side_metrics(side_curves.get("AVAR"))
            avbl_m, avbl_s, avbl_a = side_metrics(side_curves.get("AVBL"))
            avbr_m, avbr_s, avbr_a = side_metrics(side_curves.get("AVBR"))

            # combine bilateral within worm (metric-wise)
            ava_mean_vals.append(float(np.nanmean([aval_m, avar_m])))
            ava_slope_vals.append(float(np.nanmean([aval_s, avar_s])))
            ava_auc_vals.append(float(np.nanmean([aval_a, avar_a])))
            avb_mean_vals.append(float(np.nanmean([avbl_m, avbr_m])))
            avb_slope_vals.append(float(np.nanmean([avbl_s, avbr_s])))
            avb_auc_vals.append(float(np.nanmean([avbl_a, avbr_a])))

            # combine bilateral within worm (curve-wise) for plotting
            ava_dff_curves.append(_mean_traces([c for c in (side_curves.get("AVAL"), side_curves.get("AVAR")) if c is not None]))
            avb_dff_curves.append(_mean_traces([c for c in (side_curves.get("AVBL"), side_curves.get("AVBR")) if c is not None]))

        out[sex] = {
            "ava_raw": ava_raw_mean,
            "avb_raw": avb_raw_mean,
            "t_dff": t_dff,
            "ava_dff": np.nanmean(np.stack(ava_dff_curves, axis=0), axis=0),
            "avb_dff": np.nanmean(np.stack(avb_dff_curves, axis=0), axis=0),
            "ava_mean": np.asarray([float(np.nanmean(ava_mean_vals))], dtype=float),
            "ava_mean_sd": np.asarray([float(np.nanstd(ava_mean_vals, ddof=1))], dtype=float),
            "ava_slope": np.asarray([float(np.nanmean(ava_slope_vals))], dtype=float),
            "ava_slope_sd": np.asarray([float(np.nanstd(ava_slope_vals, ddof=1))], dtype=float),
            "ava_auc": np.asarray([float(np.nanmean(ava_auc_vals))], dtype=float),
            "ava_auc_sd": np.asarray([float(np.nanstd(ava_auc_vals, ddof=1))], dtype=float),
            "avb_mean": np.asarray([float(np.nanmean(avb_mean_vals))], dtype=float),
            "avb_mean_sd": np.asarray([float(np.nanstd(avb_mean_vals, ddof=1))], dtype=float),
            "avb_slope": np.asarray([float(np.nanmean(avb_slope_vals))], dtype=float),
            "avb_slope_sd": np.asarray([float(np.nanstd(avb_slope_vals, ddof=1))], dtype=float),
            "avb_auc": np.asarray([float(np.nanmean(avb_auc_vals))], dtype=float),
            "avb_auc_sd": np.asarray([float(np.nanstd(avb_auc_vals, ddof=1))], dtype=float),
            "n": np.asarray([len(subset)], dtype=int),
        }

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align AVA/AVB activity to first OP50 stimulus and plot -15..+15s windows"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to scan recursively for NWB files (if set, aggregates by sex)",
    )
    parser.add_argument(
        "--nwb",
        type=str,
        default=None,
        help="Single NWB file (used when --dir is not set)",
    )
    parser.add_argument(
        "--stimulus",
        type=str,
        default="OP50",
        help="Stimulus keyword to align to (substring match in StimulusInfo labels)",
    )
    parser.add_argument(
        "--pre",
        type=float,
        default=15.0,
        help="Seconds before stimulus to include",
    )
    parser.add_argument(
        "--post",
        type=float,
        default=15.0,
        help="Seconds after stimulus to include",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.25,
        help="Sampling step (seconds) for the extracted window (interpolation grid)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="op50_aligned_ava_avb_bysex.png",
        help="Output image path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    args = parser.parse_args(argv)

    pre_s = float(args.pre)
    post_s = float(args.post)
    if pre_s <= 0 or post_s <= 0:
        raise ValueError("--pre and --post must be > 0")

    if pre_s < 3.5:
        raise ValueError("--pre must be >= 3.5 to compute ΔF/F0 baseline over [-3.5, 0)")
    if post_s < 15.0:
        raise ValueError("--post must be >= 15.0 to compute ΔF/F0 over [0, 15]")

    dt = float(args.dt)
    if dt <= 0:
        raise ValueError("--dt must be > 0")

    rel_t_s = np.arange(-pre_s, post_s + 0.5 * dt, dt, dtype=float)

    if args.dir is not None:
        roots = _scan_nwbs(Path(args.dir))
        mode = "dir"
    else:
        if args.nwb is None:
            raise ValueError("Provide either --dir (scan) or --nwb (single file)")
        roots = _scan_nwbs(Path(args.nwb))
        mode = "single"

    extracts: list[NwbExtract] = []
    for p in roots:
        e = _extract_aligned(p, keyword=str(args.stimulus), rel_t_s=rel_t_s)
        if e is None:
            continue
        if mode == "dir" and e.sex is None:
            print(f"[skip] {p.name}: could not infer sex from filename")
            continue
        extracts.append(e)

    if not extracts:
        raise ValueError("No NWB files yielded aligned AVA/AVB traces")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 6.8), constrained_layout=True)
    axes_ff = axes[0]
    axes_raw = axes[1]

    if mode == "single":
        e = extracts[0]
        # ΔF/F0 (0..15s): compute per neuron (AVAL/AVAR/...) first, then average
        ava_sides: list[np.ndarray] = []
        avb_sides: list[np.ndarray] = []
        t_dff: np.ndarray | None = None
        for trace, bucket in (
            (e.aval, ava_sides),
            (e.avar, ava_sides),
            (e.avbl, avb_sides),
            (e.avbr, avb_sides),
        ):
            if trace is None:
                continue
            t_side, dff_side = _compute_dff0(rel_t_s=rel_t_s, trace=trace)
            if t_dff is None:
                t_dff = t_side
            bucket.append(dff_side)
        if t_dff is None:
            raise ValueError("Unexpected: could not compute ΔF/F0")
        ava_dff = _mean_traces(ava_sides)
        avb_dff = _mean_traces(avb_sides)
        ava_mean = float(np.nanmean(ava_dff))
        avb_mean = float(np.nanmean(avb_dff))
        ava_slope = _linear_regression_slope(t_dff, ava_dff)
        avb_slope = _linear_regression_slope(t_dff, avb_dff)
        ava_auc = _auc_trapz(t_dff, ava_dff)
        avb_auc = _auc_trapz(t_dff, avb_dff)
        axes_ff[0].plot(t_dff, ava_dff, color="black", linewidth=1.8, label="worm")
        axes_ff[1].plot(t_dff, avb_dff, color="black", linewidth=1.8, label="worm")

        axes_ff[0].text(
            0.02,
            0.98,
            f"mean={_fmt_float(ava_mean)}\nslope={_fmt_float(ava_slope)}/s\nAUC={_fmt_float(ava_auc)}",
            transform=axes_ff[0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        axes_ff[1].text(
            0.02,
            0.98,
            f"mean={_fmt_float(avb_mean)}\nslope={_fmt_float(avb_slope)}/s\nAUC={_fmt_float(avb_auc)}",
            transform=axes_ff[1].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        # Raw activity (-pre..+post)
        axes_raw[0].plot(rel_t_s, e.ava_raw, color="black", linewidth=1.5, label="AVA (avg L/R)")
        axes_raw[1].plot(rel_t_s, e.avb_raw, color="black", linewidth=1.5, label="AVB (avg L/R)")
        subtitle = e.path.name
        for ax in (*axes_ff, *axes_raw):
            ax.legend(loc="best", frameon=False, fontsize=9)
    else:
        agg = _aggregate_by_sex(extracts)
        colors = {"herm": "red", "male": "blue"}

        ava_metrics: dict[str, tuple[float, float, float, float, float, float, int]] = {}
        avb_metrics: dict[str, tuple[float, float, float, float, float, float, int]] = {}
        for sex in ("herm", "male"):
            if sex not in agg:
                continue
            n = int(agg[sex]["n"][0])
            # ΔF/F0 (0..15s): computed per worm per neuron-side, metrics per worm, then aggregated
            t_dff = agg[sex]["t_dff"]
            ava_dff = agg[sex]["ava_dff"]
            avb_dff = agg[sex]["avb_dff"]
            ava_mean = float(agg[sex]["ava_mean"][0])
            ava_mean_sd = float(agg[sex]["ava_mean_sd"][0])
            ava_slope = float(agg[sex]["ava_slope"][0])
            ava_slope_sd = float(agg[sex]["ava_slope_sd"][0])
            ava_auc = float(agg[sex]["ava_auc"][0])
            ava_auc_sd = float(agg[sex]["ava_auc_sd"][0])
            avb_mean = float(agg[sex]["avb_mean"][0])
            avb_mean_sd = float(agg[sex]["avb_mean_sd"][0])
            avb_slope = float(agg[sex]["avb_slope"][0])
            avb_slope_sd = float(agg[sex]["avb_slope_sd"][0])
            avb_auc = float(agg[sex]["avb_auc"][0])
            avb_auc_sd = float(agg[sex]["avb_auc_sd"][0])
            axes_ff[0].plot(t_dff, ava_dff, color=colors[sex], linewidth=2.0, label=f"{sex} (n={n})")
            axes_ff[1].plot(t_dff, avb_dff, color=colors[sex], linewidth=2.0, label=f"{sex} (n={n})")

            ava_metrics[sex] = (ava_mean, ava_mean_sd, ava_slope, ava_slope_sd, ava_auc, ava_auc_sd, n)
            avb_metrics[sex] = (avb_mean, avb_mean_sd, avb_slope, avb_slope_sd, avb_auc, avb_auc_sd, n)

            # Raw activity (-pre..+post)
            axes_raw[0].plot(rel_t_s, agg[sex]["ava_raw"], color=colors[sex], linewidth=1.8, label=f"{sex} (n={n})")
            axes_raw[1].plot(rel_t_s, agg[sex]["avb_raw"], color=colors[sex], linewidth=1.8, label=f"{sex} (n={n})")
        subtitle = Path(args.dir).name if args.dir else ""

        def _metrics_block(m: dict[str, tuple[float, float, float, float, float, float, int]]) -> str:
            lines: list[str] = []
            for sex in ("herm", "male"):
                if sex not in m:
                    continue
                mean_v, mean_sd_v, slope_v, slope_sd_v, auc_v, auc_sd_v, n_v = m[sex]
                lines.append(
                    f"{sex} (n={n_v}): mean={_fmt_float(mean_v)}±{_fmt_float(mean_sd_v)}, "
                    f"slope={_fmt_float(slope_v)}±{_fmt_float(slope_sd_v)}/s, "
                    f"AUC={_fmt_float(auc_v)}±{_fmt_float(auc_sd_v)}"
                )
            return "\n".join(lines)

        axes_ff[0].text(
            0.02,
            0.98,
            _metrics_block(ava_metrics),
            transform=axes_ff[0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        axes_ff[1].text(
            0.02,
            0.98,
            _metrics_block(avb_metrics),
            transform=axes_ff[1].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        for ax in (*axes_ff, *axes_raw):
            ax.legend(loc="best", frameon=False, fontsize=9)

    for ax, title in zip(axes_ff, ["AVA ΔF/F0 (0..15s)", "AVB ΔF/F0 (0..15s)"]):
        ax.set_title(title)
        ax.set_xlim(0.0, 15.0)
        ax.set_xlabel("Time relative to stimulus (s)")
        ax.grid(True, alpha=0.25)
    axes_ff[0].set_ylabel("ΔF/F0")

    for ax, title in zip(axes_raw, ["AVA activity (avg AVAL/AVAR)", "AVB activity (avg AVBL/AVBR)"]):
        ax.set_title(title)
        ax.axvline(0, color="k", linewidth=1, alpha=0.4)
        ax.set_xlim(rel_t_s.min(), rel_t_s.max())
        ax.set_xlabel("Time relative to stimulus (s)")
        ax.grid(True, alpha=0.25)
    axes_raw[0].set_ylabel("Activity (a.u.)")

    fig.suptitle(
        f"{args.stimulus} aligned: ΔF/F0 (baseline -3.5..0s, 0..15s) + raw activity (-{pre_s:.0f}s..+{post_s:.0f}s) — {subtitle}",
        fontsize=12,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
