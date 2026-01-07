#!/usr/bin/env python3
"""
Extract neuron point prompts from NWB and transform to calcium imaging coordinates.

This script:
1. Extracts neuron centroids from NWB file (with XY swap correction)
2. Scales coordinates from NeuroPAL dimensions to calcium imaging dimensions
3. Saves as point prompts compatible with the calcium imaging viewer
"""

import numpy as np
from pynwb import NWBHDF5IO
from pathlib import Path

# ===== HARDCODED PARAMETERS - EDIT THESE =====
NWB_PATH = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h2/sub-20220327-h2_ses-20220327_ophys.nwb"
CALCIUM_NPZ_PATH = "calcium_extracted_gaussian2.npz"  # To get target dimensions
OUTPUT_NPZ_PATH = "calcium_neuron_prompts2.npz"

# Neuron selection
NEURON_LIST = None
# NEURON_LIST = None  # Uncomment to extract all neurons

# NeuroPAL dimensions (from the NWB file) - will auto-detect
# Calcium imaging dimensions - will auto-detect from calcium file
# =============================================


def main():
    print("=" * 70)
    print("NWB → Calcium Imaging Point Prompts Extractor")
    print("=" * 70)
    
    # ========================================
    # Step 1: Load calcium imaging to get target dimensions
    # ========================================
    print("\n1. LOADING CALCIUM IMAGING DIMENSIONS")
    print("-" * 70)
    
    calcium_data = np.load(CALCIUM_NPZ_PATH)
    if 'image_4d' in calcium_data:
        calcium_4d = calcium_data['image_4d']
    else:
        calcium_4d = calcium_data[list(calcium_data.files)[0]]
    
    calcium_shape = calcium_4d.shape  # (T, Z, Y, X)
    print(f"Calcium imaging shape: {calcium_shape}")
    print(f"  T={calcium_shape[0]}, Z={calcium_shape[1]}, Y={calcium_shape[2]}, X={calcium_shape[3]}")
    
    calcium_dims = {
        'T': calcium_shape[0],
        'Z': calcium_shape[1],
        'Y': calcium_shape[2],
        'X': calcium_shape[3],
    }
    
    # ========================================
    # Step 2: Load NWB file and get NeuroPAL dimensions
    # ========================================
    print("\n2. LOADING NWB FILE")
    print("-" * 70)
    
    io = NWBHDF5IO(NWB_PATH, 'r')
    nwb = io.read()
    
    # Get NeuroPAL dimensions
    neuropal_series = nwb.acquisition["NeuroPALImageRaw"]
    neuropal_shape = neuropal_series.data.shape  # (T, Z, Y, X)
    print(f"NeuroPAL shape: {neuropal_shape}")
    print(f"  T={neuropal_shape[0]}, Z={neuropal_shape[1]}, Y={neuropal_shape[2]}, X={neuropal_shape[3]}")
    
    neuropal_dims = {
        'T': neuropal_shape[0],
        'Z': neuropal_shape[1],
        'Y': neuropal_shape[2],
        'X': neuropal_shape[3],
    }
    
    # Calculate scaling factors
    print("\n3. CALCULATING SCALING FACTORS")
    print("-" * 70)
    
    scale_z = calcium_dims['Z'] / neuropal_dims['Z']
    scale_y = calcium_dims['Y'] / neuropal_dims['Y']
    scale_x = calcium_dims['X'] / neuropal_dims['X']
    
    print(f"Scaling factors (calcium/neuropal):")
    print(f"  Z: {calcium_dims['Z']}/{neuropal_dims['Z']} = {scale_z:.4f}")
    print(f"  Y: {calcium_dims['Y']}/{neuropal_dims['Y']} = {scale_y:.4f}")
    print(f"  X: {calcium_dims['X']}/{neuropal_dims['X']} = {scale_x:.4f}")
    
    # ========================================
    # Step 4: Extract neuron coordinates
    # ========================================
    print("\n4. EXTRACTING NEURON COORDINATES")
    print("-" * 70)
    
    neuropal_neurons = nwb.processing["NeuroPAL"]["NeuroPALSegmentation"]["NeuroPALNeurons"]
    voxel_mask = neuropal_neurons.voxel_mask[:]
    id_labels = neuropal_neurons.ID_labels[:]
    
    print(f"Found {len(id_labels)} total neurons in NWB file")
    
    # Determine which neurons to extract
    if NEURON_LIST is None:
        indices_to_extract = list(range(len(id_labels)))
        print(f"Extracting all {len(indices_to_extract)} neurons")
    else:
        indices_to_extract = []
        for neuron_id in NEURON_LIST:
            found = False
            for idx, label in enumerate(id_labels):
                if str(label) == neuron_id:
                    indices_to_extract.append(idx)
                    found = True
                    break
            if not found:
                print(f"  ⚠️  Neuron '{neuron_id}' not found")
        print(f"Extracting {len(indices_to_extract)} specified neurons")
    
    # Extract and transform coordinates
    neuron_coords_neuropal = []
    neuron_coords_calcium = []
    neuron_names = []
    
    print("\nExtracting and transforming coordinates:")
    for idx in indices_to_extract:
        neuron_id = str(id_labels[idx])
        neuron_coords_raw = voxel_mask[idx]
        
        # Parse coordinates
        if hasattr(neuron_coords_raw, '__len__') and len(neuron_coords_raw) >= 3:
            # CRITICAL: NWB voxel_mask stores coordinates in a swapped format!
            # Position [0] = value that should map to Y in image
            # Position [1] = value that should map to X in image
            # Position [2] = value that should map to Z in image
            val0 = float(neuron_coords_raw[0])  # This is the Y-like value
            val1 = float(neuron_coords_raw[1])  # This is the X-like value
            val2 = float(neuron_coords_raw[2])  # This is the Z-like value
        else:
            print(f"  ⚠️  Skipping '{neuron_id}': unexpected coordinate format")
            continue
        
        # Correct mapping discovered through testing:
        # val0 goes to Y position in NeuroPAL image
        # val1 goes to X position in NeuroPAL image
        # val2 goes to Z position in NeuroPAL image
        
        # For napari (Z, Y, X) format in NeuroPAL space:
        z_neuropal = val2
        y_neuropal = val0
        x_neuropal = val1
        
        neuropal_coord = [z_neuropal, y_neuropal, x_neuropal]  # [Z, Y, X]
        
        # Scale to calcium imaging dimensions
        z_calcium = z_neuropal * scale_z
        y_calcium = y_neuropal * scale_y
        x_calcium = x_neuropal * scale_x
        
        calcium_coord = [z_calcium, y_calcium, x_calcium]
        
        neuron_coords_neuropal.append(neuropal_coord)
        neuron_coords_calcium.append(calcium_coord)
        neuron_names.append(neuron_id)
        
        print(f"  ✓ {neuron_id:8s}: voxel_mask({val0:6.1f},{val1:6.1f},{val2:6.1f}) "
              f"→ NeuroPAL(z,y,x)=({z_neuropal:6.1f},{y_neuropal:6.1f},{x_neuropal:6.1f}) "
              f"→ Calcium(z,y,x)=({z_calcium:6.1f},{y_calcium:6.1f},{x_calcium:6.1f})")
    
    if not neuron_coords_calcium:
        print("\n❌ No valid neuron coordinates extracted!")
        io.close()
        return
    
    neuron_coords_neuropal = np.array(neuron_coords_neuropal)
    neuron_coords_calcium = np.array(neuron_coords_calcium)
    neuron_names = np.array(neuron_names)
    
    print(f"\n✅ Extracted {len(neuron_coords_calcium)} neuron coordinates")
    
    # ========================================
    # Step 5: Save to NPZ
    # ========================================
    print("\n5. SAVING POINT PROMPTS")
    print("-" * 70)
    
    output_path = Path(OUTPUT_NPZ_PATH)
    
    np.savez_compressed(
        output_path,
        neuron_coords_calcium=neuron_coords_calcium,  # Scaled to calcium dimensions
        neuron_coords_neuropal=neuron_coords_neuropal,  # Original NeuroPAL dimensions
        neuron_names=neuron_names,
        calcium_dims=np.array([calcium_dims['T'], calcium_dims['Z'], calcium_dims['Y'], calcium_dims['X']]),
        neuropal_dims=np.array([neuropal_dims['T'], neuropal_dims['Z'], neuropal_dims['Y'], neuropal_dims['X']]),
        scale_factors=np.array([scale_z, scale_y, scale_x]),
    )
    
    print(f"Saved to: {output_path}")
    
    io.close()
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Neurons extracted: {len(neuron_names)}")
    print(f"Neuron names: {', '.join(neuron_names)}")
    print(f"\nCoordinate ranges in calcium imaging space:")
    print(f"  Z: [{neuron_coords_calcium[:, 0].min():.1f}, {neuron_coords_calcium[:, 0].max():.1f}] (image: {calcium_dims['Z']})")
    print(f"  Y: [{neuron_coords_calcium[:, 1].min():.1f}, {neuron_coords_calcium[:, 1].max():.1f}] (image: {calcium_dims['Y']})")
    print(f"  X: [{neuron_coords_calcium[:, 2].min():.1f}, {neuron_coords_calcium[:, 2].max():.1f}] (image: {calcium_dims['X']})")
    
    print(f"\nOutput file: {output_path}")
    
    print("\n📖 To load in Python:")
    print(f"   data = np.load('{output_path.name}')")
    print(f"   neuron_coords = data['neuron_coords_calcium']  # Shape: {neuron_coords_calcium.shape}")
    print(f"   neuron_names = data['neuron_names']  # {len(neuron_names)} neurons")
    
    print("\n✅ Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
