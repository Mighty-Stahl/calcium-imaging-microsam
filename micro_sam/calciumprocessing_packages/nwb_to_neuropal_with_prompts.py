#!/usr/bin/env python3
"""
Extract NeuroPAL image AND neuron coordinates from NWB file with proper alignment.

This script extracts both the NeuroPAL image and neuron centroid positions,
ensuring that the coordinate transformations match so that point prompts
align correctly with the image when loaded in napari.

Key difference from separate extraction:
- Applies the SAME coordinate transformation to both image and point coordinates
- Ensures alignment between NeuroPAL image (Z, Y, X, C) and points (Z, Y, X)

Usage:
    python nwb_to_neuropal_with_prompts.py <input.nwb>
    python nwb_to_neuropal_with_prompts.py  # Will prompt for file selection
"""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


# ===== HARDCODED PARAMETERS - EDIT THESE =====
NWB_PATH = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h2/sub-20220327-h2_ses-20220327_ophys.nwb"
OUTPUT_PATH = None  # None = auto-generate name, or specify custom path
TIMEPOINT = 0  # Which timepoint to extract from the NWB file

# Neuron selection - choose one option:
# Option 1: Extract specific neurons (uncomment and modify list)
NEURON_LIST = ['AWAL','AWAR','AWCR','AWCL']

# Option 2: Extract all neurons (uncomment line below and comment out NEURON_LIST above)
# NEURON_LIST = None  # None = extract all neurons
# =============================================


def extract_neuropal_with_prompts(
    nwb_path: str,
    output_path: str = None,
    timepoint: int = 0,
    neuron_list: list = None,
):
    """Extract NeuroPAL image and neuron coordinates from NWB file.
    
    The image is stored as (T, Y, X, Z, C) in NWB and transposed to (Z, Y, X, C) for napari.
    The neuron coordinates are stored as (x, y, z) in NWB and transformed to match:
    - Original NWB: (x, y, z) corresponds to (X, Y, Z) dimensions
    - After transpose: coordinates become (z, y, x) to match napari (Z, Y, X) order
    
    Args:
        nwb_path: Path to input NWB file
        output_path: Path to output NPZ file (optional)
        timepoint: Which timepoint to extract (default: 0)
        neuron_list: List of neuron IDs to extract (e.g., ['AVAL', 'AVAR']). 
                    If None, extracts all neurons.
        
    Returns:
        Path to saved NPZ file
    """
    nwb_path = Path(nwb_path)
    
    if output_path is None:
        output_path = nwb_path.parent / f"{nwb_path.stem}_neuropal_with_prompts.npz"
    else:
        output_path = Path(output_path)
    
    print(f"Loading NWB file: {nwb_path}")
    print("=" * 70)
    
    try:
        io = NWBHDF5IO(str(nwb_path), "r")
        nwb = io.read()
    except Exception as e:
        print(f"Failed to open NWB file: {e}")
        return None
    
    # ========================================
    # PART 1: Extract NeuroPAL Image
    # ========================================
    
    if "NeuroPALImageRaw" not in nwb.acquisition:
        print(f"Error: 'NeuroPALImageRaw' not found in acquisition.")
        print(f"Available keys: {list(nwb.acquisition.keys())}")
        io.close()
        return None
    
    neuropal_series = nwb.acquisition["NeuroPALImageRaw"]
    
    print(f"\n📷 NeuroPAL Image:")
    print(f"   Original shape: {neuropal_series.data.shape}")
    print(f"   Data type: {neuropal_series.data.dtype}")
    
    original_shape = neuropal_series.data.shape
    is_rgb = len(original_shape) == 5
    
    if len(original_shape) == 5:
        print(f"   Format: 5D - likely (T, Y, X, Z, C) or (T, Z, Y, X, C) - RGB")
    elif len(original_shape) == 4:
        print(f"   Format: 4D - likely (T, Z, Y, X) - Grayscale")
        print(f"   Interpreting as: T={original_shape[0]}, Z={original_shape[1]}, Y={original_shape[2]}, X={original_shape[3]}")
    else:
        print(f"   Warning: Unexpected shape {original_shape}")
    
    # Extract single timepoint
    print(f"\n   Extracting timepoint {timepoint}...")
    try:
        neuropal_data = neuropal_series.data[timepoint]
    except IndexError:
        print(f"   Error: Timepoint {timepoint} out of range. Using timepoint 0.")
        neuropal_data = neuropal_series.data[0]
        timepoint = 0
    
    print(f"   Extracted shape: {neuropal_data.shape}")
    
    # Transpose to napari format
    if is_rgb:
        # Need to determine if (Y, X, Z, C) or (Z, Y, X, C)
        # For now, assume (Y, X, Z, C) -> (Z, Y, X, C)
        neuropal_data = np.transpose(neuropal_data, (2, 0, 1, 3))
        print(f"   Transposed to: {neuropal_data.shape} (Z, Y, X, C)")
    else:
        # Format is (Z, Y, X) - already correct for napari!
        print(f"   Format (Z, Y, X): {neuropal_data.shape} - already correct for napari")
    
    # Convert to float32 for napari
    if neuropal_data.dtype == np.uint16:
        print("   Converting uint16 to float32 (normalized to 0-1)")
        neuropal_data = neuropal_data.astype(np.float32) / 65535.0
    elif neuropal_data.dtype == np.uint8:
        print("   Converting uint8 to float32 (normalized to 0-1)")
        neuropal_data = neuropal_data.astype(np.float32) / 255.0
    
    print(f"   Final dtype: {neuropal_data.dtype}")
    print(f"   Value range: [{neuropal_data.min():.4f}, {neuropal_data.max():.4f}]")
    
    # ========================================
    # PART 2: Extract Neuron Coordinates
    # ========================================
    
    print(f"\n🧠 Neuron Coordinates:")
    
    neuron_coords = []
    neuron_names = []
    neuron_info = {}
    
    try:
        neuropal_neurons = nwb.processing["NeuroPAL"]["NeuroPALSegmentation"]["NeuroPALNeurons"]
        voxel_mask = neuropal_neurons.voxel_mask[:]
        id_labels = neuropal_neurons.ID_labels[:]
        
        print(f"   Found {len(id_labels)} neurons in NWB file")
        
        # Determine which neurons to extract
        if neuron_list is None:
            # Extract all neurons
            indices_to_extract = list(range(len(id_labels)))
            print(f"   Extracting all {len(indices_to_extract)} neurons")
        else:
            # Extract specific neurons
            indices_to_extract = []
            for neuron_id in neuron_list:
                found = False
                for idx, label in enumerate(id_labels):
                    if str(label) == neuron_id:
                        indices_to_extract.append(idx)
                        found = True
                        break
                if not found:
                    print(f"   ⚠️  Neuron '{neuron_id}' not found")
            print(f"   Extracting {len(indices_to_extract)} specified neurons")
        
        # Extract coordinates for each neuron
        for idx in indices_to_extract:
            neuron_id = str(id_labels[idx])
            neuron_coords_raw = voxel_mask[idx]
            
            # Parse coordinates (could be single point or array)
            if hasattr(neuron_coords_raw, '__len__') and len(neuron_coords_raw) >= 3:
                # Single coordinate (x, y, z) or (x, y, z, value)
                x = float(neuron_coords_raw[0])
                y = float(neuron_coords_raw[1])
                z = float(neuron_coords_raw[2])
            else:
                # Array of coordinates - compute centroid
                coords_array = np.array(neuron_coords_raw)
                if len(coords_array.shape) == 2 and coords_array.shape[0] > 0:
                    xyz = coords_array[:, :3]
                    centroid = np.mean(xyz, axis=0)
                    x, y, z = float(centroid[0]), float(centroid[1]), float(centroid[2])
                else:
                    print(f"   ⚠️  Skipping '{neuron_id}': unexpected coordinate format")
                    continue
            
            # CRITICAL: Transform coordinates to match image format
            # NWB image format: (T, Z, Y, X) → after extracting timepoint: (Z, Y, X)
            # NWB coordinates are ACTUALLY stored as: (y, x, z) NOT (x, y, z)!
            # This is a quirk of how NeuroPAL coordinates are stored in this NWB format
            #   - First value (labeled 'x') actually corresponds to Y dimension
            #   - Second value (labeled 'y') actually corresponds to X dimension  
            #   - Third value (labeled 'z') corresponds to Z dimension
            
            # For napari (Z, Y, X) format, we need to swap x and y:
            # voxel_mask gives us (y_value, x_value, z_value), so:
            transformed_coords = [z, x, y]  # [z, Y, X] - note X and Y are swapped!
            
            neuron_coords.append(transformed_coords)
            neuron_names.append(neuron_id)
            neuron_info[neuron_id] = {
                'original_coords': [x, y, z],
                'napari_coords': transformed_coords,
                'index': idx
            }
            
            print(f"   ✓ {neuron_id:8s}: voxel_mask=({x:6.1f},{y:6.1f},{z:6.1f}) → napari(z,y,x)=({z:6.1f},{x:6.1f},{y:6.1f})")
        
        if not neuron_coords:
            print("   ⚠️  No valid neuron coordinates extracted")
            neuron_coords = None
            neuron_names = None
        else:
            neuron_coords = np.array(neuron_coords)
            neuron_names = np.array(neuron_names)
            print(f"\n   ✅ Extracted {len(neuron_coords)} neuron coordinates")
            print(f"   Coordinate format: (Z, Y, X) to match napari image axes")
    
    except KeyError as e:
        print(f"   ⚠️  NeuroPAL neuron data not found in NWB file: {e}")
        print(f"   Continuing without neuron coordinates...")
        neuron_coords = None
        neuron_names = None
        neuron_info = {}
    
    # ========================================
    # PART 3: Save to NPZ
    # ========================================
    
    print(f"\n💾 Saving to: {output_path}")
    
    save_dict = {
        'neuropal': neuropal_data,
        'timepoint': timepoint,
        'original_shape': original_shape,
        'is_rgb': is_rgb,
    }
    
    if neuron_coords is not None:
        save_dict['neuron_coords'] = neuron_coords
        save_dict['neuron_names'] = neuron_names
    
    np.savez_compressed(output_path, **save_dict)
    
    # Also save JSON with detailed neuron info
    if neuron_info:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(neuron_info, f, indent=2)
        print(f"   Neuron info saved to: {json_path}")
    
    io.close()
    
    print("\n" + "=" * 70)
    print("✅ Extraction completed successfully!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   NeuroPAL shape: {neuropal_data.shape}")
    print(f"   Format: {'RGB' if is_rgb else 'Grayscale'}")
    if neuron_coords is not None:
        print(f"   Neurons: {len(neuron_coords)}")
        print(f"   Neuron names: {', '.join(neuron_names[:10])}" + 
              (f"... (+{len(neuron_names)-10} more)" if len(neuron_names) > 10 else ""))
    print(f"   Output file: {output_path}")
    
    print(f"\n📖 How to load in Python:")
    print(f"   ```python")
    print(f"   import numpy as np")
    print(f"   import napari")
    print(f"   ")
    print(f"   data = np.load('{output_path.name}')")
    print(f"   neuropal = data['neuropal']")
    if neuron_coords is not None:
        print(f"   neuron_coords = data['neuron_coords']  # Shape: {neuron_coords.shape}")
        print(f"   neuron_names = data['neuron_names']    # {len(neuron_names)} neurons")
    print(f"   ")
    print(f"   viewer = napari.Viewer()")
    print(f"   viewer.add_image(neuropal, name='NeuroPAL')")
    if neuron_coords is not None:
        print(f"   viewer.add_points(neuron_coords, name='Neurons', size=5, face_color='red')")
    print(f"   napari.run()")
    print(f"   ```")
    
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract NeuroPAL image and neuron coordinates from NWB file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.nwb
  %(prog)s input.nwb --output neuropal_with_prompts.npz
  %(prog)s input.nwb --timepoint 5
  %(prog)s input.nwb --neurons AVAL AVAR AWCL AWCR
  %(prog)s input.nwb --all-neurons
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
        help="Path to output NPZ file (default: same directory with _neuropal_with_prompts.npz suffix)",
    )
    parser.add_argument(
        "--timepoint", "-t",
        type=int,
        default=0,
        help="Which timepoint to extract (default: 0)",
    )
    parser.add_argument(
        "--neurons", "-n",
        nargs="+",
        type=str,
        default=None,
        help="Specific neuron IDs to extract (e.g., AVAL AVAR AWCL). If not specified, extracts all neurons.",
    )
    parser.add_argument(
        "--all-neurons",
        action="store_true",
        help="Extract all neurons (default if --neurons not specified)",
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
            print("Usage: python nwb_to_neuropal_with_prompts.py <input.nwb>")
            return 1
    
    # Determine neuron list
    neuron_list = args.neurons if args.neurons else None
    
    result = extract_neuropal_with_prompts(
        nwb_file, 
        args.output, 
        args.timepoint,
        neuron_list
    )
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    print("=" * 70)
    print("NeuroPAL Image + Neuron Coordinates Extractor")
    print("=" * 70)
    print(f"NWB file: {NWB_PATH}")
    print(f"Output: {OUTPUT_PATH if OUTPUT_PATH else 'Auto-generated'}")
    print(f"Timepoint: {TIMEPOINT}")
    print(f"Neurons: {NEURON_LIST if NEURON_LIST else 'All neurons'}")
    print("=" * 70)
    print()
    
    result = extract_neuropal_with_prompts(
        NWB_PATH,
        OUTPUT_PATH,
        TIMEPOINT,
        NEURON_LIST
    )
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)
