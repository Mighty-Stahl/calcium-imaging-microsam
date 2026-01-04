#!/usr/bin/env python3
"""
Extract fluorescence traces from segmented neurons in NPZ files.

Performs:
1. Raw fluorescence extraction (mean intensity per neuron)
2. Background subtraction (global per-frame)
3. You should define pre-stimulus baseline to be used as Fo.
4. ΔF/F₀ normalization

Usage:
    python extract_fluorescence.py <path_to_segmentation.npz>
    python extract_fluorescence.py  # Will prompt for file selection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter


# ============================================================
# SAVITZKY-GOLAY SMOOTHING CONFIGURATION
# ============================================================
APPLY_SAVGOL_SMOOTHING = True  # Set to False to disable smoothing
SAVGOL_WINDOW_SIZE = 5          # Must be odd number (e.g., 3, 5, 7, 9)
SAVGOL_POLYORDER = 2            # Polynomial order (must be < window_size)
# ============================================================

# ============================================================
# BASELINE (F₀) CALCULATION METHOD
# ============================================================
BASELINE_METHOD = "percentile"  # Options: "percentile" or "pre_stimulus"

# For percentile method:
BASELINE_PERCENTILE = 10.0      # Percentile to use for F₀ (e.g., 10 = 10th percentile)

# For pre-stimulus baseline method:
BASELINE_START_FRAME = 0        # Start of baseline window (inclusive)
BASELINE_END_FRAME = 10         # End of baseline window (exclusive)
BASELINE_STAT = "median"        # Statistic to use: "mean", "median", or "min"
# ============================================================


def extract_raw_fluorescence(image_4d: np.ndarray, segmentation_4d: np.ndarray) -> dict:
    """Extract raw fluorescence traces for each neuron.
    
    Args:
        image_4d: Raw calcium imaging data (T, Z, Y, X)
        segmentation_4d: Segmentation masks (T, Z, Y, X) with neuron IDs
        
    Returns:
        Dictionary mapping neuron_id -> raw fluorescence trace (array of length T)
    """
    neuron_ids = np.unique(segmentation_4d)
    neuron_ids = neuron_ids[neuron_ids > 0]  # Exclude background (0)
    
    n_timesteps = image_4d.shape[0]
    traces = {}
    
    print(f"\nExtracting raw fluorescence for {len(neuron_ids)} neurons...")
    
    for neuron_id in neuron_ids:
        fluorescence = np.zeros(n_timesteps)
        
        for t in range(n_timesteps):
            mask = segmentation_4d[t] == neuron_id
            
            if mask.any():
                # Mean intensity across all pixels in this neuron
                fluorescence[t] = image_4d[t][mask].mean()
            else:
                fluorescence[t] = np.nan
        
        traces[neuron_id] = fluorescence
        print(f"  Neuron {neuron_id}: {np.count_nonzero(~np.isnan(fluorescence))} timepoints")
    
    return traces


def subtract_background(image_4d: np.ndarray, segmentation_4d: np.ndarray, 
                       raw_traces: dict, background_percentile: float = 10.0) -> dict:
    """Subtract global per-frame background from raw traces.
    
    Background is computed as the percentile of all non-neuron pixels (ID=0) at each timepoint.
    Using percentile instead of mean makes it robust to unsegmented bright neurons.
    
    Args:
        image_4d: Raw calcium imaging data (T, Z, Y, X)
        segmentation_4d: Segmentation masks (T, Z, Y, X)
        raw_traces: Dictionary of raw fluorescence traces
        background_percentile: Percentile to use for background (default: 10)
        
    Returns:
        Dictionary mapping neuron_id -> background-subtracted trace
    """
    n_timesteps = image_4d.shape[0]
    corrected_traces = {}
    
    print(f"\nPerforming background subtraction (global per-frame, {background_percentile}th percentile)...")
    
    # Compute background for each timepoint
    background_trace = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        # Background mask = all pixels with ID 0
        background_mask = segmentation_4d[t] == 0
        
        if background_mask.any():
            # Use percentile instead of mean to be robust to unsegmented neurons
            background_trace[t] = np.percentile(image_4d[t][background_mask], background_percentile)
        else:
            # Fallback: use percentile of entire frame
            background_trace[t] = np.percentile(image_4d[t], background_percentile)
    
    print(f"  Background range: [{background_trace.min():.2f}, {background_trace.max():.2f}]")
    
    # Subtract background from each neuron
    for neuron_id, raw_trace in raw_traces.items():
        corrected_traces[neuron_id] = raw_trace - background_trace
        
    return corrected_traces, background_trace


def compute_dff(corrected_traces: dict, method: str = "percentile", 
                baseline_percentile: float = 10.0,
                baseline_start: int = 0, baseline_end: int = 10,
                baseline_stat: str = "median") -> dict:
    """Compute ΔF/F₀ normalization.
    
    Two methods available:
    1. Percentile: F₀ = percentile of entire trace (robust to outliers)
    2. Pre-stimulus: F₀ = mean/median of baseline frames (e.g., before stimulus)
    
    Args:
        corrected_traces: Background-subtracted traces
        method: "percentile" or "pre_stimulus"
        baseline_percentile: Percentile for F₀ if using percentile method (default: 10)
        baseline_start: Start frame for baseline window if using pre_stimulus method
        baseline_end: End frame (exclusive) for baseline window if using pre_stimulus method
        baseline_stat: Statistic to use for pre_stimulus baseline: "mean", "median", or "min"
        
    Returns:
        Dictionary mapping neuron_id -> ΔF/F₀ trace
    """
    dff_traces = {}
    
    print(f"\nComputing ΔF/F₀ using '{method}' method...")
    
    for neuron_id, trace in corrected_traces.items():
        # Remove NaN values for baseline calculation
        valid_mask = ~np.isnan(trace)
        valid_values = trace[valid_mask]
        
        if len(valid_values) == 0:
            dff_traces[neuron_id] = trace  # All NaN
            print(f"  Neuron {neuron_id}: Skipped (all NaN values)")
            continue
        
        # Compute F₀ based on method
        if method == "percentile":
            # Use percentile of entire trace
            f0 = np.percentile(valid_values, baseline_percentile)
            method_desc = f"{baseline_percentile}th percentile"
            
        elif method == "pre_stimulus":
            # Use baseline frames only
            baseline_frames = trace[baseline_start:baseline_end]
            baseline_valid = baseline_frames[~np.isnan(baseline_frames)]
            
            if len(baseline_valid) == 0:
                # Fallback to percentile if no valid baseline frames
                f0 = np.percentile(valid_values, baseline_percentile)
                method_desc = f"fallback to {baseline_percentile}th percentile (no valid baseline frames)"
            else:
                # Compute F₀ from baseline frames
                if baseline_stat == "mean":
                    f0 = np.mean(baseline_valid)
                elif baseline_stat == "median":
                    f0 = np.median(baseline_valid)
                elif baseline_stat == "min":
                    f0 = np.min(baseline_valid)
                else:
                    f0 = np.median(baseline_valid)  # Default to median
                
                method_desc = f"{baseline_stat} of frames {baseline_start}-{baseline_end}"
        else:
            # Invalid method - fallback to percentile
            f0 = np.percentile(valid_values, baseline_percentile)
            method_desc = f"fallback to {baseline_percentile}th percentile (invalid method)"
            print(f"  Warning: Unknown method '{method}', using percentile")
        
        # Avoid division by zero
        if f0 <= 0:
            f0 = valid_values.mean() if valid_values.mean() > 0 else 1.0
            method_desc += " (adjusted for zero baseline)"
        
        # Compute ΔF/F₀
        dff = (trace - f0) / f0
        
        # If using pre-stimulus method, set baseline frames to NaN (only analyze post-baseline)
        if method == "pre_stimulus":
            dff[:baseline_end] = np.nan
            method_desc += f" (baseline frames {baseline_start}-{baseline_end} set to NaN)"
        
        dff_traces[neuron_id] = dff
        
        print(f"  Neuron {neuron_id}: F₀={f0:.2f} ({method_desc}), ΔF/F₀ range=[{np.nanmin(dff):.3f}, {np.nanmax(dff):.3f}]")
    
    return dff_traces


def apply_savgol_smoothing(dff_traces: dict, window_size: int = 5, polyorder: int = 2) -> dict:
    """Apply Savitzky-Golay filter to smooth ΔF/F₀ traces.
    
    The Savitzky-Golay filter smooths data using a polynomial fit within a sliding window.
    It preserves features like peaks better than simple moving averages.
    
    Args:
        dff_traces: ΔF/F₀ traces to smooth
        window_size: Length of filter window (must be odd). Larger = smoother but may lose detail.
        polyorder: Order of polynomial to fit (must be < window_size). Usually 2 or 3.
        
    Returns:
        Dictionary mapping neuron_id -> smoothed ΔF/F₀ trace
    """
    smoothed_traces = {}
    
    print(f"\nApplying Savitzky-Golay smoothing (window={window_size}, polyorder={polyorder})...")
    
    # Validate parameters
    if window_size % 2 == 0:
        print(f"  Warning: window_size must be odd, adjusting {window_size} -> {window_size + 1}")
        window_size += 1
    
    if polyorder >= window_size:
        print(f"  Warning: polyorder must be < window_size, adjusting to {window_size - 1}")
        polyorder = window_size - 1
    
    for neuron_id, trace in dff_traces.items():
        # Handle NaN values
        valid_mask = ~np.isnan(trace)
        
        if np.sum(valid_mask) < window_size:
            # Not enough valid points for smoothing
            smoothed_traces[neuron_id] = trace.copy()
            print(f"  Neuron {neuron_id}: Skipped (insufficient valid points)")
            continue
        
        # Apply Savitzky-Golay filter
        try:
            smoothed = trace.copy()
            valid_indices = np.where(valid_mask)[0]
            
            # Only smooth if we have a contiguous block of valid data
            if len(valid_indices) == len(trace) and np.all(np.diff(valid_indices) == 1):
                # Contiguous data - apply filter directly
                smoothed = savgol_filter(trace, window_size, polyorder)
            else:
                # Sparse data - only smooth valid regions
                # For simplicity, skip smoothing if data has gaps
                print(f"  Neuron {neuron_id}: Skipped (data contains gaps/NaN values)")
            
            smoothed_traces[neuron_id] = smoothed
            
            # Show effect
            noise_reduction = np.std(trace[valid_mask]) - np.std(smoothed[valid_mask])
            print(f"  Neuron {neuron_id}: Noise reduced by {noise_reduction:.4f} std")
            
        except Exception as e:
            print(f"  Neuron {neuron_id}: Failed to smooth ({e}), keeping original")
            smoothed_traces[neuron_id] = trace.copy()
    
    return smoothed_traces


def plot_traces(raw_traces: dict, corrected_traces: dict, dff_traces: dict, 
                background_trace: np.ndarray, smoothed_traces: dict = None, output_path: Path = None):
    """Plot all trace processing steps.
    
    Args:
        raw_traces: Raw fluorescence traces
        corrected_traces: Background-subtracted traces
        dff_traces: ΔF/F₀ normalized traces
        background_trace: Per-frame background values
        smoothed_traces: Optional Savitzky-Golay smoothed ΔF/F₀ traces
        output_path: Optional path to save figure
    """
    neuron_ids = sorted(raw_traces.keys())
    n_neurons = len(neuron_ids)
    
    # ============================================================
    # ADJUST PLOT SIZE HERE (width, height in inches)
    # ============================================================
    PLOT_WIDTH = 20 if smoothed_traces else 15  # Wider if showing smoothed traces
    ROW_HEIGHT = 2           # Height per neuron row (inches)
    # ============================================================
    
    # Create figure with subplots
    n_cols = 4 if smoothed_traces else 3
    fig = plt.figure(figsize=(PLOT_WIDTH, ROW_HEIGHT * n_neurons + 2))
    gs = GridSpec(n_neurons + 1, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot background trace at the top
    ax_bg = fig.add_subplot(gs[0, :])
    timepoints = np.arange(len(background_trace))
    ax_bg.plot(timepoints, background_trace, 'k-', linewidth=1.5, label='Background')
    ax_bg.set_xlabel('Timepoint')
    ax_bg.set_ylabel('Intensity')
    ax_bg.set_title('Global Background (per-frame)', fontsize=12, fontweight='bold')
    ax_bg.grid(True, alpha=0.3)
    ax_bg.legend()
    
    # Plot traces for each neuron
    for idx, neuron_id in enumerate(neuron_ids):
        row = idx + 1
        
        # Raw fluorescence
        ax_raw = fig.add_subplot(gs[row, 0])
        ax_raw.plot(timepoints, raw_traces[neuron_id], 'b-', linewidth=1)
        ax_raw.set_ylabel('Raw F')
        ax_raw.set_title(f'Neuron {neuron_id}: Raw', fontsize=10)
        ax_raw.grid(True, alpha=0.3)
        if row == n_neurons:
            ax_raw.set_xlabel('Timepoint')
        
        # Background-subtracted
        ax_corr = fig.add_subplot(gs[row, 1])
        ax_corr.plot(timepoints, corrected_traces[neuron_id], 'g-', linewidth=1)
        ax_corr.set_ylabel('F - Background')
        ax_corr.set_title(f'Neuron {neuron_id}: Background Subtracted', fontsize=10)
        ax_corr.grid(True, alpha=0.3)
        ax_corr.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if row == n_neurons:
            ax_corr.set_xlabel('Timepoint')
        
        # ΔF/F₀
        ax_dff = fig.add_subplot(gs[row, 2])
        ax_dff.plot(timepoints, dff_traces[neuron_id], 'r-', linewidth=1)
        ax_dff.set_ylabel('ΔF/F₀')
        ax_dff.set_title(f'Neuron {neuron_id}: ΔF/F₀', fontsize=10)
        ax_dff.grid(True, alpha=0.3)
        ax_dff.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if row == n_neurons:
            ax_dff.set_xlabel('Timepoint')
        
        # Smoothed ΔF/F₀ (if available)
        if smoothed_traces:
            ax_smooth = fig.add_subplot(gs[row, 3])
            ax_smooth.plot(timepoints, smoothed_traces[neuron_id], 'm-', linewidth=1.5, label='Smoothed')
            ax_smooth.plot(timepoints, dff_traces[neuron_id], 'r-', linewidth=0.5, alpha=0.3, label='Original')
            ax_smooth.set_ylabel('ΔF/F₀')
            ax_smooth.set_title(f'Neuron {neuron_id}: Smoothed', fontsize=10)
            ax_smooth.grid(True, alpha=0.3)
            ax_smooth.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax_smooth.legend(fontsize=8)
            if row == n_neurons:
                ax_smooth.set_xlabel('Timepoint')
    
    title = 'Fluorescence Trace Processing Pipeline'
    if smoothed_traces:
        title += ' (with Savitzky-Golay Smoothing)'
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_path}")
    
    plt.show()


def plot_dff_only(dff_traces: dict, smoothed_traces: dict = None, output_path: Path = None, neuron_names: dict = None):
    """Plot only ΔF/F₀ traces for all neurons in a clean layout.
    
    Args:
        dff_traces: ΔF/F₀ normalized traces
        smoothed_traces: Optional Savitzky-Golay smoothed traces to overlay
        output_path: Optional path to save figure
        neuron_names: Optional dict mapping neuron_id -> neuron_name
    """
    neuron_ids = sorted(dff_traces.keys())
    n_neurons = len(neuron_ids)
    
    # Create figure - one subplot per neuron
    fig, axes = plt.subplots(n_neurons, 1, figsize=(12, 2.5 * n_neurons), sharex=True)
    
    # Handle single neuron case
    if n_neurons == 1:
        axes = [axes]
    
    # Get timepoints
    first_trace = dff_traces[neuron_ids[0]]
    timepoints = np.arange(len(first_trace))
    
    # Plot each neuron
    for idx, neuron_id in enumerate(neuron_ids):
        ax = axes[idx]
        trace = dff_traces[neuron_id]
        
        # Get neuron name
        if neuron_names and int(neuron_id) in neuron_names:
            neuron_label = neuron_names[int(neuron_id)]
            title = f'{neuron_label} (ID {neuron_id})'
        else:
            neuron_label = f'Neuron {neuron_id}'
            title = neuron_label
        
        # Plot trace
        if smoothed_traces and neuron_id in smoothed_traces:
            # Show both original and smoothed
            ax.plot(timepoints, trace, 'r-', linewidth=0.8, alpha=0.4, label=f'{neuron_label} (raw)')
            ax.plot(timepoints, smoothed_traces[neuron_id], 'm-', linewidth=2, label=f'{neuron_label} (smoothed)')
        else:
            # Show only original
            ax.plot(timepoints, trace, 'r-', linewidth=1.5, label=neuron_label)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Styling
        ax.set_ylabel('ΔF/F₀', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add statistics text
        valid_data = trace[~np.isnan(trace)]
        if len(valid_data) > 0:
            stats_text = f'Min: {np.min(valid_data):.3f}  Max: {np.max(valid_data):.3f}  Mean: {np.mean(valid_data):.3f}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # X-label only on bottom plot
    axes[-1].set_xlabel('Timepoint', fontsize=11)
    
    # Overall title
    title = 'ΔF/F₀ Fluorescence Traces'
    if smoothed_traces:
        title += ' (with Savitzky-Golay Smoothing)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 ΔF/F₀ plot saved to: {output_path}")
    
    plt.show()


def save_traces(raw_traces: dict, corrected_traces: dict, dff_traces: dict,
                background_trace: np.ndarray, smoothed_traces: dict = None, output_path: Path = None):
    """Save all traces to NPZ file.
    
    Args:
        raw_traces: Raw fluorescence traces
        corrected_traces: Background-subtracted traces
        dff_traces: ΔF/F₀ traces
        background_trace: Global background trace
        smoothed_traces: Optional Savitzky-Golay smoothed traces
        output_path: Path to save NPZ file
    """
    neuron_ids = sorted(raw_traces.keys())
    
    # Stack traces into arrays
    raw_array = np.array([raw_traces[nid] for nid in neuron_ids])
    corrected_array = np.array([corrected_traces[nid] for nid in neuron_ids])
    dff_array = np.array([dff_traces[nid] for nid in neuron_ids])
    
    save_dict = {
        'neuron_ids': np.array(neuron_ids),
        'raw_traces': raw_array,
        'corrected_traces': corrected_array,
        'dff_traces': dff_array,
        'background_trace': background_trace,
    }
    
    if smoothed_traces:
        smoothed_array = np.array([smoothed_traces[nid] for nid in neuron_ids])
        save_dict['smoothed_traces'] = smoothed_array
    
    np.savez(output_path, **save_dict)
    
    print(f"\n💾 Traces saved to: {output_path}")
    print(f"   - neuron_ids: {neuron_ids}")
    print(f"   - raw_traces: {raw_array.shape}")
    print(f"   - corrected_traces: {corrected_array.shape}")
    print(f"   - dff_traces: {dff_array.shape}")
    if smoothed_traces:
        print(f"   - smoothed_traces: {smoothed_array.shape}")
    print(f"   - background_trace: {background_trace.shape}")


def save_traces_csv(raw_traces: dict, corrected_traces: dict, dff_traces: dict,
                    background_trace: np.ndarray, smoothed_traces: dict = None, 
                    output_path: Path = None, neuron_names: dict = None):
    """Save all traces to CSV files.
    
    Creates separate CSV files for each processing stage.
    
    Args:
        raw_traces: Raw fluorescence traces
        corrected_traces: Background-subtracted traces
        dff_traces: ΔF/F₀ traces
        background_trace: Global background trace
        smoothed_traces: Optional Savitzky-Golay smoothed traces
        output_path: Base path for CSV files (will append suffixes)
        neuron_names: Optional dict mapping neuron_id -> neuron_name (e.g., {1: 'AVAL', 2: 'AVAR'})
    """
    import csv
    
    neuron_ids = sorted(raw_traces.keys())
    n_timesteps = len(background_trace)
    
    # Create column headers with neuron names if available
    if neuron_names:
        headers = [neuron_names.get(int(nid), f'Neuron_{nid}') for nid in neuron_ids]
    else:
        headers = [f'Neuron_{nid}' for nid in neuron_ids]
    
    # Create CSV paths
    base_path = output_path.with_suffix('')
    raw_csv = f"{base_path}_raw.csv"
    corrected_csv = f"{base_path}_corrected.csv"
    dff_csv = f"{base_path}_dff.csv"
    smoothed_csv = f"{base_path}_smoothed.csv"
    background_csv = f"{base_path}_background.csv"
    
    # Save raw traces
    with open(raw_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timepoint'] + headers)
        for t in range(n_timesteps):
            row = [t] + [raw_traces[nid][t] for nid in neuron_ids]
            writer.writerow(row)
    
    # Save background-subtracted traces
    with open(corrected_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timepoint'] + headers)
        for t in range(n_timesteps):
            row = [t] + [corrected_traces[nid][t] for nid in neuron_ids]
            writer.writerow(row)
    
    # Save ΔF/F₀ traces
    with open(dff_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timepoint'] + headers)
        for t in range(n_timesteps):
            row = [t] + [dff_traces[nid][t] for nid in neuron_ids]
            writer.writerow(row)
    
    # Save smoothed traces (if available)
    if smoothed_traces:
        with open(smoothed_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timepoint'] + headers)
            for t in range(n_timesteps):
                row = [t] + [smoothed_traces[nid][t] for nid in neuron_ids]
                writer.writerow(row)
    
    # Save background trace
    with open(background_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timepoint', 'Background'])
        for t in range(n_timesteps):
            writer.writerow([t, background_trace[t]])
    
    print(f"\n💾 CSV files saved:")
    print(f"   - Raw: {raw_csv}")
    print(f"   - Corrected: {corrected_csv}")
    print(f"   - ΔF/F₀: {dff_csv}")
    if smoothed_traces:
        print(f"   - Smoothed: {smoothed_csv}")
    print(f"   - Background: {background_csv}")
    if neuron_names:
        print(f"   ✓ Using neuron names: {list(neuron_names.values())}")


def load_neuron_names(segmentation_dir: Path) -> dict:
    """Load neuron names from neuron_names.json if it exists.
    
    Args:
        segmentation_dir: Directory to search for neuron_names.json
        
    Returns:
        Dictionary mapping neuron_id -> neuron_name, or empty dict if not found
    """
    import json
    
    # Check multiple possible locations
    possible_paths = [
        segmentation_dir / "neuron_names.json",
        segmentation_dir.parent / "neuron_names.json",
        segmentation_dir.parent / "neuropal_prompts" / "neuron_names.json",
    ]
    
    for names_path in possible_paths:
        if names_path.exists():
            try:
                with open(names_path, 'r') as f:
                    names_dict = json.load(f)
                # Convert keys to integers
                names_dict = {int(k): v for k, v in names_dict.items()}
                print(f"\n✓ Loaded neuron names from: {names_path}")
                print(f"  Found {len(names_dict)} named neurons: {list(names_dict.values())}")
                return names_dict
            except Exception as e:
                print(f"⚠️  Failed to load neuron names from {names_path}: {e}")
    
    return {}


def process_segmentation_file(npz_path: str) -> int:
    """Load segmentation NPZ and extract fluorescence traces.
    
    Args:
        npz_path: Path to segmentation NPZ file
        
    Returns:
        Exit code (0 for success)
    """
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return 1
    
    print(f"Loading segmentation from: {npz_path}")
    
    # Load data
    data = np.load(npz_path)
    
    if "image_4d" not in data or "segmentation_4d" not in data:
        print("Error: NPZ must contain 'image_4d' and 'segmentation_4d'")
        return 1
    
    image_4d = data["image_4d"]
    segmentation_4d = data["segmentation_4d"]
    
    print(f"Image shape: {image_4d.shape}")
    print(f"Segmentation shape: {segmentation_4d.shape}")
    
    # Diagnostic information about image data
    print(f"\nImage diagnostics:")
    print(f"  dtype: {image_4d.dtype}")
    print(f"  min: {image_4d.min():.4f}, max: {image_4d.max():.4f}")
    print(f"  mean: {image_4d.mean():.4f}, std: {image_4d.std():.4f}")
    
    background_pixels = segmentation_4d == 0
    print(f"\nBackground diagnostics:")
    print(f"  Background pixels (ID=0) count: {np.sum(background_pixels):,}")
    print(f"  Background pixels 10th percentile: {np.percentile(image_4d[background_pixels], 10):.4f}")
    print(f"  Background pixels mean: {image_4d[background_pixels].mean():.4f}")
    
    # Count neurons
    neuron_ids = np.unique(segmentation_4d)
    neuron_ids = neuron_ids[neuron_ids > 0]
    print(f"Found {len(neuron_ids)} neurons: {neuron_ids}")
    
    # Extract fluorescence traces
    raw_traces = extract_raw_fluorescence(image_4d, segmentation_4d)
    
    # Background subtraction (using 10th percentile to be robust to unsegmented neurons)
    corrected_traces, background_trace = subtract_background(
        image_4d, segmentation_4d, raw_traces, background_percentile=10.0
    )
    
    # ΔF/F₀ normalization
    dff_traces = compute_dff(
        corrected_traces, 
        method=BASELINE_METHOD,
        baseline_percentile=BASELINE_PERCENTILE,
        baseline_start=BASELINE_START_FRAME,
        baseline_end=BASELINE_END_FRAME,
        baseline_stat=BASELINE_STAT
    )
    
    # Optional: Savitzky-Golay smoothing
    smoothed_traces = None
    if APPLY_SAVGOL_SMOOTHING:
        smoothed_traces = apply_savgol_smoothing(dff_traces, SAVGOL_WINDOW_SIZE, SAVGOL_POLYORDER)
    else:
        print("\n⏭️  Savitzky-Golay smoothing disabled (APPLY_SAVGOL_SMOOTHING = False)")
    
    # Prepare output paths
    output_dir = npz_path.parent / "fluorescence_traces"
    output_dir.mkdir(exist_ok=True)
    
    traces_path = output_dir / f"{npz_path.stem}_traces.npz"
    plot_path = output_dir / f"{npz_path.stem}_traces.png"
    dff_plot_path = output_dir / f"{npz_path.stem}_dff_only.png"
    csv_base_path = output_dir / "traces"
    
    # Try to load neuron names
    neuron_names = load_neuron_names(npz_path.parent)
    
    # Save traces (NPZ format)
    save_traces(raw_traces, corrected_traces, dff_traces, background_trace, smoothed_traces, traces_path)
    
    # Save traces (CSV format with neuron names if available)
    save_traces_csv(raw_traces, corrected_traces, dff_traces, background_trace, smoothed_traces, csv_base_path, neuron_names)
    
    # Plot results - full pipeline
    print("\n📈 Generating plots...")
    plot_traces(raw_traces, corrected_traces, dff_traces, background_trace, smoothed_traces, plot_path)
    
    # Plot results - ΔF/F₀ only (with neuron names if available)
    plot_dff_only(dff_traces, smoothed_traces, dff_plot_path, neuron_names)
    
    print("\n✅ Processing complete!")
    return 0


def main(argv=None):
    """Main entry point."""
    
    # ============================================================
    # SET YOUR FILE PATH HERE (or leave as None for file dialog)
    # ⭐️ ============================================================ ⭐️
    DEFAULT_FILE = "/Users/arnlois/data/code/saved_segmentations/my_segmentation.npz"
    # DEFAULT_FILE = None  # Uncomment this to use file dialog or command line
    # ⭐️ ============================================================ ⭐️
    
    parser = argparse.ArgumentParser(
        description="Extract fluorescence traces from segmented neurons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my_segmentation.npz
  %(prog)s /path/to/segmentation.npz
        """
    )
    parser.add_argument(
        "npz_file",
        nargs="?",
        type=str,
        help="Path to segmentation NPZ file (with image_4d and segmentation_4d)",
    )
    
    args = parser.parse_args(argv)
    
    # Priority: command line arg > DEFAULT_FILE > file dialog
    npz_file = args.npz_file or DEFAULT_FILE
    
    if npz_file is None:
        try:
            from tkinter import Tk, filedialog
            print("No file specified. Opening file dialog...")
            root = Tk()
            root.withdraw()
            npz_file = filedialog.askopenfilename(
                title="Select Segmentation NPZ File",
                filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
            )
            root.destroy()
            
            if not npz_file:
                print("No file selected. Exiting.")
                return 1
                
        except ImportError:
            print("Error: No file specified and tkinter not available.")
            print("Usage: python extract_fluorescence.py <path_to_segmentation.npz>")
            return 1
    
    return process_segmentation_file(npz_file)


if __name__ == "__main__":
    sys.exit(main())
