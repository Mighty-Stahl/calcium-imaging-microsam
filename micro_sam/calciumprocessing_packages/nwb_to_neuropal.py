#!/usr/bin/env python3
"""
Extract NeuroPAL image from NWB file and save to NPZ.

NeuroPAL is typically stored as (T, Y, X, Z, C) in NWB files.
This script extracts a single timepoint (default: t=0) and saves as (Z, Y, X, C).

Usage:
    python nwb_to_neuropal.py <input.nwb>
    python nwb_to_neuropal.py  # Will prompt for file selection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


def extract_neuropal_from_nwb(nwb_path: str, output_path: str = None, timepoint: int = 0):
    """Extract NeuroPAL image from NWB file.
    
    Args:
        nwb_path: Path to input NWB file
        output_path: Path to output NPZ file (optional)
        timepoint: Which timepoint to extract (default: 0)
        
    Returns:
        Path to saved NPZ file
    """
    nwb_path = Path(nwb_path)
    
    if output_path is None:
        output_path = nwb_path.parent / f"{nwb_path.stem}_neuropal.npz"
    else:
        output_path = Path(output_path)
    
    print(f"Loading NWB file: {nwb_path}")
    
    try:
        io = NWBHDF5IO(str(nwb_path), "r")
        nwb = io.read()
    except Exception as e:
        print(f"Failed to open NWB file: {e}")
        return None
    
    # Check for NeuroPAL data
    if "NeuroPALImageRaw" not in nwb.acquisition:
        print(f"Error: 'NeuroPALImageRaw' not found in acquisition.")
        print(f"Available keys: {list(nwb.acquisition.keys())}")
        io.close()
        return None
    
    neuropal_series = nwb.acquisition["NeuroPALImageRaw"]
    
    print(f"NeuroPAL data shape: {neuropal_series.data.shape}")
    print(f"NeuroPAL data dtype: {neuropal_series.data.dtype}")
    
    # Expected formats:
    # - (T, Y, X, Z, C) - 5D with color
    # - (T, Z, Y, X) - 4D grayscale
    original_shape = neuropal_series.data.shape
    is_rgb = len(original_shape) == 5
    
    if len(original_shape) == 5:
        print(f"Detected RGB NeuroPAL: (T, Y, X, Z, C)")
    elif len(original_shape) == 4:
        print(f"Detected grayscale NeuroPAL: (T, Z, Y, X)")
    else:
        print(f"Warning: Unexpected shape {original_shape}")
    
    # Extract single timepoint
    print(f"Extracting timepoint {timepoint}...")
    try:
        neuropal_data = neuropal_series.data[timepoint]
    except IndexError:
        print(f"Error: Timepoint {timepoint} out of range. Using timepoint 0.")
        neuropal_data = neuropal_series.data[0]
    
    print(f"Extracted shape: {neuropal_data.shape}")
    
    # Transpose to napari format: (Z, Y, X) for grayscale or (Z, Y, X, C) for RGB
    if is_rgb:
        # (Y, X, Z, C) -> (Z, Y, X, C)
        neuropal_data = np.transpose(neuropal_data, (2, 0, 1, 3))
        print(f"Transposed to napari RGB format: {neuropal_data.shape} (Z, Y, X, C)")
    else:
        # Already (Z, Y, X) - no transpose needed
        print(f"Already in napari grayscale format: {neuropal_data.shape} (Z, Y, X)")
    
    # Convert to appropriate dtype
    if neuropal_data.dtype == np.uint16:
        # Keep uint16 but normalize to 0-1 range for napari
        print("Converting uint16 to float32 (normalized to 0-1)")
        neuropal_data = neuropal_data.astype(np.float32) / 65535.0
    elif neuropal_data.dtype == np.uint8:
        print("Converting uint8 to float32 (normalized to 0-1)")
        neuropal_data = neuropal_data.astype(np.float32) / 255.0
    
    print(f"Final dtype: {neuropal_data.dtype}")
    print(f"Value range: [{neuropal_data.min():.4f}, {neuropal_data.max():.4f}]")
    
    # Save to NPZ
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(
        output_path,
        neuropal=neuropal_data,
        timepoint=timepoint,
        original_shape=original_shape,
        is_rgb=is_rgb,
    )
    
    io.close()
    
    print(f"✓ NeuroPAL image saved successfully!")
    print(f"  Shape: {neuropal_data.shape}")
    print(f"  Type: {'RGB' if is_rgb else 'Grayscale'}")
    print(f"  File: {output_path}")
    print(f"\nTo load in annotator_minimal:")
    print(f"  python run_annotatorminimal.py --path <calcium_imaging.npz> --neuropal {output_path}")
    
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract NeuroPAL image from NWB file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.nwb
  %(prog)s input.nwb --output neuropal.npz
  %(prog)s input.nwb --timepoint 5
        """
    )
    parser.add_argument(
        "nwb_file",
        nargs="?",
        type=str,
        help="Path to input NWB file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output NPZ file (default: same directory as input with _neuropal.npz suffix)",
    )
    parser.add_argument(
        "--timepoint", "-t",
        type=int,
        default=0,
        help="Which timepoint to extract (default: 0)",
    )
    
    args = parser.parse_args(argv)
    
    nwb_file = args.nwb_file
    
    if nwb_file is None:
        try:
            from tkinter import Tk, filedialog
            print("No file specified. Opening file dialog...")
            root = Tk()
            root.withdraw()
            nwb_file = filedialog.askopenfilename(
                title="Select NWB File",
                filetypes=[("NWB files", "*.nwb"), ("All files", "*.*")],
            )
            root.destroy()
            
            if not nwb_file:
                print("No file selected. Exiting.")
                return 1
                
        except ImportError:
            print("Error: No file specified and tkinter not available.")
            print("Usage: python nwb_to_neuropal.py <input.nwb>")
            return 1
    
    result = extract_neuropal_from_nwb(nwb_file, args.output, args.timepoint)
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
