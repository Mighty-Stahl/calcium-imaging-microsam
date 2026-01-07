#!/usr/bin/env python3
"""Napari viewer for calcium imaging with neuron point prompts."""

import numpy as np
import napari

# ===== HARDCODED PARAMETERS - EDIT THESE =====
CALCIUM_NPZ_PATH = "calcium_extracted_gaussian_full.npz"
PROMPTS_NPZ_PATH = "calcium_neuron_prompts2.npz"
# =============================================

def main():
    """Load and view calcium imaging with neuron point prompts."""
    print(f"Loading calcium imaging: {CALCIUM_NPZ_PATH}")
    print(f"Loading neuron prompts: {PROMPTS_NPZ_PATH}")
    
    # Load calcium imaging
    calcium_data = np.load(CALCIUM_NPZ_PATH)
    if 'image_4d' in calcium_data:
        calcium_4d = calcium_data['image_4d']
    else:
        calcium_4d = calcium_data[list(calcium_data.files)[0]]
    
    print(f"\nCalcium imaging shape: {calcium_4d.shape} (T, Z, Y, X)")
    print(f"Data type: {calcium_4d.dtype}")
    print(f"Intensity range: [{calcium_4d.min()}, {calcium_4d.max()}]")
    
    # Load neuron prompts
    prompts_data = np.load(PROMPTS_NPZ_PATH)
    neuron_coords = prompts_data['neuron_coords_calcium']
    neuron_names = prompts_data['neuron_names']
    
    print(f"\nNeuron prompts: {len(neuron_names)} neurons")
    print(f"Neuron names: {', '.join(neuron_names)}")
    print(f"Coordinate ranges:")
    print(f"  Z: [{neuron_coords[:, 0].min():.1f}, {neuron_coords[:, 0].max():.1f}]")
    print(f"  Y: [{neuron_coords[:, 1].min():.1f}, {neuron_coords[:, 1].max():.1f}]")
    print(f"  X: [{neuron_coords[:, 2].min():.1f}, {neuron_coords[:, 2].max():.1f}]")
    
    # Create Napari viewer
    viewer = napari.Viewer()
    
    # Calculate contrast limits for better visualization
    nonzero = calcium_4d[calcium_4d > 0]
    if len(nonzero) > 0:
        vmin = 0
        vmax = np.percentile(nonzero, 99.5)
    else:
        vmin = float(calcium_4d.min())
        vmax = float(calcium_4d.max())
        if vmin == vmax:
            vmax = vmin + 1
    
    print(f"\nContrast limits: [{vmin}, {vmax}]")
    
    # Add calcium imaging (4D: T, Z, Y, X)
    layer = viewer.add_image(
        calcium_4d,
        name="Calcium Imaging",
        colormap="gray",
        contrast_limits=[vmin, vmax],
        scale=(1, 1, 1, 1),  # (T, Z, Y, X)
        gamma=0.8,
        rendering="mip",
        blending="translucent",
    )
    
    layer.interpolation2d = "linear"
    layer.interpolation3d = "linear"
    
    # Add neuron point prompts
    # Points are 3D (Z, Y, X), but we need to show them across all timepoints
    # We'll add them as 3D points that appear in all time slices
    
    print(f"\nAdding {len(neuron_coords)} neuron point prompts...")
    
    properties = {
        'neuron_name': neuron_names,
    }
    
    text = {
        'string': '{neuron_name}',
        'size': 10,
        'color': 'yellow',
    }
    
    # Add points layer (3D coordinates for points in the 4D viewer)
    # Napari will show these points across all timepoints
    points_layer = viewer.add_points(
        neuron_coords,
        name='Neuron IDs',
        size=8,
        face_color='red',
        properties=properties,
        text=text,
        opacity=0.8,
        ndim=3,  # 3D points (Z, Y, X)
    )
    
    print("\n✅ Calcium imaging and neuron prompts loaded successfully!")
    print("\nControls:")
    print("  - Use time slider (T) to move through frames")
    print("  - Use Z slider to move through Z-slices")
    print("  - Scroll to zoom, drag to pan")
    print("  - Toggle layer visibility with eye icon")
    print("  - Hover over points to see neuron names")
    print("  - Select points layer and press 'T' to toggle text labels")
    print("\nNOTE: Points are displayed as 3D locations, so they appear")
    print("      at the same Z position across all timepoints.")
    
    # Start the viewer
    napari.run()


if __name__ == "__main__":
    main()
