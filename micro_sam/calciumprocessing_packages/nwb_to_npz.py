#!/usr/bin/env python3
"""Convert NWB CalciumImageSeries to NPZ format for MicroSAM 4D Annotator.

Extracts calcium imaging data, handles channel selection, and converts to (T,Z,Y,X) format.
"""
import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO

# ===== HARDCODED PARAMETERS - EDIT THESE =====
NWB_PATH = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h2/sub-20220327-h2_ses-20220327_ophys.nwb"
OUTPUT_PATH = "calcium_extracted.npz"

fps = 2.67

stim_time = 342.0
stim_frame = int(stim_time * fps)

baseline = 10.0
response = 15  # seconds


START_FRAME = int((stim_time - baseline) * fps)
END_FRAME = int((stim_time + response) * fps)  # Set to None for all frames
CHANNEL = 2  # 0 - RED, 1 - GREEN, 3 - BLUE
SERIES_NAME = "CalciumImageSeries"
COMPRESS = True
NORMALIZE = False
# =============================================


def convert_nwb_to_npz(
    nwb_path: str,
    output_path: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    channel: str | int = "green",
    series_name: str = "CalciumImageSeries",
    compress: bool = True,
    normalize: bool = False,
):
    """Convert NWB calcium imaging to NPZ.
    
    Args:
        nwb_path: Path to input NWB file
        output_path: Path for output NPZ file
        start_frame: First frame to include (0-indexed)
        end_frame: Last frame to include (exclusive, None = all)
        channel: Channel selection - "green" (1), "red" (0), "blue" (2), "avg", or int index
        series_name: Name of acquisition series (default: CalciumImageSeries)
        compress: Use compression for NPZ output
        normalize: Apply percentile normalization (1-99%)
    """
    print(f"Loading NWB file: {nwb_path}")
    
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        
        # Get the calcium imaging series
        if series_name not in nwb.acquisition:
            available = list(nwb.acquisition.keys())
            raise ValueError(f"Series '{series_name}' not found. Available: {available}")
        
        series = nwb.acquisition[series_name]
        
        print(f"Series: {series_name}")
        print(f"  Shape: {series.data.shape}")
        print(f"  Rate: {series.rate} Hz")
        
        # Determine frame range
        total_frames = series.data.shape[0]
        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"Invalid start_frame {start_frame} (total frames: {total_frames})")
        if end_frame <= start_frame:
            raise ValueError(f"Invalid range: start={start_frame}, end={end_frame}")
        
        n_frames = end_frame - start_frame
        print(f"\nExtracting frames {start_frame} to {end_frame-1} ({n_frames} frames)")
        
        # Load data slice (memory-mapped if possible)
        print("Loading data...")
        arr = np.asarray(series.data[start_frame:end_frame])
        print(f"  Loaded shape: {arr.shape}")
        
    # Handle channel selection
    # Expected shape: (T, Y, X, Z, C) where C=3 (RGB channels)
    if arr.shape[-1] == 3:
        print(f"\nChannel selection: {channel}")
        
        if isinstance(channel, int):
            ch_idx = channel
        elif channel == "avg":
            print("  Averaging across channels...")
            arr = arr.mean(axis=-1)
            ch_idx = None
        else:
            # Map channel names to indices
            name_to_idx = {"red": 0, "green": 1, "blue": 2, "r": 0, "g": 1, "b": 1}
            ch_idx = name_to_idx.get(channel.lower(), 1)
        
        if ch_idx is not None:
            print(f"  Selecting channel {ch_idx}")
            arr = arr[..., ch_idx]
        
        print(f"  Shape after channel selection: {arr.shape}")
    
    # Detect axes and transpose to (T, Z, Y, X)
    # Current shape should be (T, Y, X, Z) after removing channel
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array after channel selection, got shape {arr.shape}")
    
    # Find Z axis (smallest dimension, typically Z=23)
    shape_list = list(arr.shape)
    
    # Time is axis 0 (already correct)
    # Find Z axis among remaining axes (smallest non-time axis)
    non_time_axes = list(range(1, arr.ndim))
    z_axis = min(non_time_axes, key=lambda i: shape_list[i])
    
    print(f"\nDetected axes:")
    print(f"  Time (T): axis 0, size {shape_list[0]}")
    print(f"  Z: axis {z_axis}, size {shape_list[z_axis]}")
    
    # Build transpose order: (T=0, Z, remaining axes for Y,X)
    other_axes = [i for i in range(1, arr.ndim) if i != z_axis]
    new_order = (0, z_axis) + tuple(other_axes)
    
    print(f"\nTransposing from {arr.shape} to (T, Z, Y, X)")
    print(f"  Transpose order: {new_order}")
    
    arr_tzyx = np.transpose(arr, axes=new_order)
    
    print(f"  Final shape: {arr_tzyx.shape} (T, Z, Y, X)")
    print(f"  Dtype: {arr_tzyx.dtype}")
    
    # Optional normalization
    if normalize:
        print("\nNormalizing (1-99 percentile)...")
        arr_tzyx = arr_tzyx.astype("float32")
        p1, p99 = np.percentile(arr_tzyx, (1, 99))
        arr_tzyx = np.clip(arr_tzyx, p1, p99)
        arr_tzyx = (arr_tzyx - p1) / (p99 - p1 + 1e-8)
        print(f"  Normalized to range [0, 1]")
    
    print(f"  Intensity range: [{arr_tzyx.min():.1f}, {arr_tzyx.max():.1f}]")
    
    # Calculate size
    size_mb = arr_tzyx.nbytes / (1024**2)
    print(f"  Size: {size_mb:.1f} MB")
    
    # Save to NPZ
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    
    if compress:
        np.savez_compressed(output_path, image_4d=arr_tzyx)
    else:
        np.savez(output_path, image_4d=arr_tzyx)
    
    print(f"✅ Conversion complete!")
    print(f"\nTo load in annotator:")
    print(f"  python run_annotator4d.py --path {output_path}")


def main():
    """Run conversion with hardcoded parameters."""
    convert_nwb_to_npz(
        nwb_path=NWB_PATH,
        output_path=OUTPUT_PATH,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        channel=CHANNEL,
        series_name=SERIES_NAME,
        compress=COMPRESS,
        normalize=NORMALIZE,
    )


if __name__ == "__main__":
    main()

