#!/usr/bin/env python3
"""Plot sex-separated odor responses as raw fluorescence and ΔF/F0 from the wide CSV.

Expected input format (wide):
- Metadata columns like: odor, neuron, sex, dataset_name, animal, cycle, ...
- Then many timepoint columns named like "-5.62s", "0.00s", "14.98s", etc.

Computation:
- Baseline F0 per worm trace: mean fluorescence from -5 to 0 seconds (inclusive).
- Stimulus window: compute (Ft - F0) / F0 for times from 0 to 15 seconds (inclusive).
- Average across cycles within a worm, then average across worms within each sex.

Output:
- 2x2 grid:
    - Columns: AVA vs AVB
    - Rows: raw fluorescence (Ft) and ΔF/F0
- Each axis compares sexes:
    - Hermaphrodite = red
    - Male = blue
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_TIME_COL_RE = re.compile(r"^(-?\d+(?:\.\d+)?)s$")


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float


def _extract_time_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    time_cols: list[str] = []
    time_vals: list[float] = []
    for col in df.columns:
        m = _TIME_COL_RE.match(str(col))
        if not m:
            continue
        time_cols.append(col)
        time_vals.append(float(m.group(1)))

    if not time_cols:
        raise ValueError(
            "No timepoint columns found. Expected columns like '-5.62s' or '0.00s'."
        )

    # keep in ascending time order
    order = np.argsort(time_vals)
    time_cols = [time_cols[i] for i in order]
    time_vals = np.asarray([time_vals[i] for i in order], dtype=float)
    return time_cols, time_vals


def _pick_cols_in_window(time_cols: list[str], time_vals: np.ndarray, window: Window) -> list[str]:
    mask = (time_vals >= window.start_s) & (time_vals <= window.end_s)
    cols = [c for c, ok in zip(time_cols, mask) if ok]
    if not cols:
        raise ValueError(
            f"No time columns in window [{window.start_s}, {window.end_s}] seconds. "
            f"Available range is [{time_vals.min():.2f}, {time_vals.max():.2f}] seconds."
        )
    return cols


def _normalize_neuron_group(neuron: str) -> str:
    n = (neuron or "").strip().upper()
    if n.startswith("AVA"):
        return "AVA"
    if n.startswith("AVB"):
        return "AVB"
    return n


def _infer_worm_id(df: pd.DataFrame) -> pd.Series:
    # Prefer an explicit worm/animal identifier if present.
    for col in ("worm", "worm_id", "animal", "Animal", "animal_id"):
        if col in df.columns:
            # If it's numeric, cast to int-like string to avoid '2.0' etc.
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                return s.astype("Int64").astype(str)
            return s.astype(str)

    # Fallback: dataset_name if present, else row index.
    if "dataset_name" in df.columns:
        return df["dataset_name"].astype(str)

    return pd.Series(df.index.astype(str), index=df.index)


def compute_dff(
    df: pd.DataFrame,
    baseline_window: Window,
    stim_window: Window,
    odor: str | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    required = ["sex", "neuron", "odor"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    if odor is not None:
        available_odors = sorted(set(df["odor"].astype(str)))
        df = df[df["odor"].astype(str) == str(odor)].copy()
        if df.empty:
            raise ValueError(f"No rows for odor={odor!r}. Available odors: {available_odors}")

    time_cols, time_vals = _extract_time_columns(df)
    baseline_cols = _pick_cols_in_window(time_cols, time_vals, baseline_window)
    stim_cols = _pick_cols_in_window(time_cols, time_vals, stim_window)

    # numeric conversion for time columns
    df_time = df[time_cols].apply(pd.to_numeric, errors="coerce")

    f0 = df_time[baseline_cols].mean(axis=1, skipna=True)
    # avoid divide-by-zero explosions
    f0 = f0.replace(0, np.nan)

    stim = df_time[stim_cols]
    dff = (stim.sub(f0, axis=0)).div(f0, axis=0)

    out = df[["sex", "neuron", "odor"]].copy()
    out["neuron_group"] = out["neuron"].astype(str).map(_normalize_neuron_group)

    # worm identity for de-weighting multiple cycles per worm
    worm_id = _infer_worm_id(df)
    if "dataset_name" in df.columns and "animal" in df.columns:
        out["worm_id"] = df["dataset_name"].astype(str) + "_" + df["animal"].astype(str)
    else:
        out["worm_id"] = worm_id

    out = pd.concat([out, dff], axis=1)

    stim_times = np.asarray([float(_TIME_COL_RE.match(c).group(1)) for c in stim_cols], dtype=float)
    return out, stim_times, stim_cols


def compute_raw(
    df: pd.DataFrame,
    stim_window: Window,
    odor: str | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    required = ["sex", "neuron", "odor"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    if odor is not None:
        available_odors = sorted(set(df["odor"].astype(str)))
        df = df[df["odor"].astype(str) == str(odor)].copy()
        if df.empty:
            raise ValueError(f"No rows for odor={odor!r}. Available odors: {available_odors}")

    time_cols, time_vals = _extract_time_columns(df)
    stim_cols = _pick_cols_in_window(time_cols, time_vals, stim_window)

    df_time = df[time_cols].apply(pd.to_numeric, errors="coerce")
    stim = df_time[stim_cols]

    out = df[["sex", "neuron", "odor"]].copy()
    out["neuron_group"] = out["neuron"].astype(str).map(_normalize_neuron_group)

    worm_id = _infer_worm_id(df)
    if "dataset_name" in df.columns and "animal" in df.columns:
        out["worm_id"] = df["dataset_name"].astype(str) + "_" + df["animal"].astype(str)
    else:
        out["worm_id"] = worm_id

    out = pd.concat([out, stim], axis=1)

    stim_times = np.asarray([float(_TIME_COL_RE.match(c).group(1)) for c in stim_cols], dtype=float)
    return out, stim_times, stim_cols


def aggregate_by_sex_and_neuron(dff_df: pd.DataFrame, stim_cols: list[str]) -> pd.DataFrame:
    # 1) average within worm across cycles/replicates
    worm_mean = (
        dff_df.groupby(["sex", "neuron_group", "odor", "worm_id"], dropna=False)[stim_cols]
        .mean()
        .reset_index()
    )

    # 2) average across worms, compute SEM
    n_worms = (
        worm_mean.groupby(["sex", "neuron_group", "odor"], dropna=False)["worm_id"]
        .nunique()
        .rename("n_worms")
        .reset_index()
    )

    mean = worm_mean.groupby(["sex", "neuron_group", "odor"], dropna=False)[stim_cols].mean().reset_index()
    std = worm_mean.groupby(["sex", "neuron_group", "odor"], dropna=False)[stim_cols].std(ddof=1).reset_index()

    out = mean.merge(std, on=["sex", "neuron_group", "odor"], suffixes=("__mean", "__std"))
    out = out.merge(n_worms, on=["sex", "neuron_group", "odor"], how="left")

    # convert std -> sem
    for col in stim_cols:
        out[f"{col}__sem"] = out[f"{col}__std"] / np.sqrt(out["n_worms"].clip(lower=1))

    return out


def _plot_neuron_panel(
    ax,
    agg: pd.DataFrame,
    stim_times: np.ndarray,
    stim_cols: list[str],
    *,
    neuron: str,
    title: str,
    ylabel: str | None,
):
    sex_colors = {
        "herm": "red",
        "male": "blue",
    }

    ax.set_title(title)
    ax.axvspan(0, 15, alpha=0.08)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Time (s)")

    any_line = False
    for sex, color in sex_colors.items():
        sub = agg[agg["sex"].astype(str).str.lower() == sex]
        row = sub[sub["neuron_group"] == neuron]
        if row.empty:
            continue
        row = row.iloc[0]
        y = np.asarray([row[f"{c}__mean"] for c in stim_cols], dtype=float)
        yerr = np.asarray([row[f"{c}__sem"] for c in stim_cols], dtype=float)
        ax.plot(stim_times, y, color=color, label=f"{sex} (n={int(row['n_worms'])})")
        ax.fill_between(stim_times, y - yerr, y + yerr, color=color, alpha=0.18, linewidth=0)
        any_line = True

    if not any_line:
        ax.text(0.5, 0.5, f"No data for {neuron}", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(stim_times.min(), stim_times.max())
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_raw_and_dff(
    *,
    raw_agg: pd.DataFrame,
    dff_agg: pd.DataFrame,
    stim_times: np.ndarray,
    stim_cols: list[str],
    output: Path | None,
    odor: str | None,
    show: bool,
):
    import matplotlib.pyplot as plt

    neurons = ["AVA", "AVB"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)

    for col, neuron in enumerate(neurons):
        odor_title = f" — {odor}" if odor else ""
        _plot_neuron_panel(
            axes[0, col],
            raw_agg,
            stim_times,
            stim_cols,
            neuron=neuron,
            title=f"{neuron} (raw){odor_title}",
            ylabel="Fluorescence (a.u.)" if col == 0 else None,
        )
        _plot_neuron_panel(
            axes[1, col],
            dff_agg,
            stim_times,
            stim_cols,
            neuron=neuron,
            title=f"{neuron} (ΔF/F₀){odor_title}",
            ylabel="ΔF/F₀" if col == 0 else None,
        )

    axes[0, 0].legend(loc="best", frameon=False)
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot herm vs male ΔF/F0 (0–15s) from odor_response CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="odor_response_data (2).csv",
        help="Path to the input CSV (wide format)",
    )
    parser.add_argument(
        "--odor",
        type=str,
        default=None,
        help="Optional odor filter (e.g. OP50). If omitted, uses all odors combined.",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        help="If set, only include rows with this cycle (e.g. 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="odor_response_dff_herm_vs_male.png",
        help="Output image path (png/pdf/etc).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if args.cycle is not None:
        if "cycle" not in df.columns:
            raise ValueError("--cycle was provided but the CSV has no 'cycle' column")
        df_cycle = df.copy()
        df_cycle["cycle"] = pd.to_numeric(df_cycle["cycle"], errors="coerce")
        df = df_cycle[df_cycle["cycle"] == float(args.cycle)].copy()
        if df.empty:
            raise ValueError(f"No rows found for cycle={args.cycle}")

    if "odor" in df.columns and args.odor is None:
        odors = [o for o in df["odor"].astype(str).unique().tolist() if o and o != "nan"]
        if len(odors) > 1:
            # Avoid mixing conditions implicitly.
            args.odor = odors[0]
            print(f"[plot_odor_response_dff] Multiple odors found: {odors}. Using --odor {args.odor!r}.")

    dff_df, stim_times, stim_cols = compute_dff(
        df,
        baseline_window=Window(-5.0, 0.0),
        stim_window=Window(0.0, 15.0),
        odor=args.odor,
    )

    raw_df, raw_times, raw_cols = compute_raw(
        df,
        stim_window=Window(0.0, 15.0),
        odor=args.odor,
    )
    if not np.allclose(raw_times, stim_times) or raw_cols != stim_cols:
        raise RuntimeError("Stimulus time columns mismatch between raw and ΔF/F0 computations")

    agg = aggregate_by_sex_and_neuron(dff_df, stim_cols)
    raw_agg = aggregate_by_sex_and_neuron(raw_df, stim_cols)

    plot_raw_and_dff(
        raw_agg=raw_agg,
        dff_agg=agg,
        stim_times=stim_times,
        stim_cols=stim_cols,
        output=Path(args.output) if args.output else None,
        odor=args.odor,
        show=not args.no_show,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
