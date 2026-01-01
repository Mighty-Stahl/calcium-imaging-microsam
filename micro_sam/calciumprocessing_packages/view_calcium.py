#!/usr/bin/env python3
"""Simple Napari viewer for calcium imaging NPZ files."""

import numpy as np
import napari

# ===== HARDCODED PARAMETERS - EDIT THESE =====
NPZ_PATH = "calcium_extracted1.npz"
# =============================================

def main():
    """Load and view calcium imaging data in Napari."""
    print(f"Loading NPZ file: {NPZ_PATH}")
    
    # Load the NPZ file
    npz = np.load(NPZ_PATH)
    
    # Get the image data (key should be 'image_4d')
    if 'image_4d' in npz:
        image_4d = npz['image_4d']
    else:
        # If key is different, try to find it
        keys = list(npz.files)
        print(f"Available keys: {keys}")
        image_4d = npz[keys[0]]
    
    print(f"Loaded shape: {image_4d.shape} (T, Z, Y, X)")
    print(f"Data type: {image_4d.dtype}")
    print(f"Intensity range: [{image_4d.min()}, {image_4d.max()}]")
    
    # Create Napari viewer
    viewer = napari.Viewer()
    
    # Calculate contrast limits for better visualization
    # Use a tighter percentile range to enhance dim features
    nonzero = image_4d[image_4d > 0]
    if len(nonzero) > 0:
        vmin = 0
        vmax = np.percentile(nonzero, 99.5)  # Focus on bright features
    else:
        vmin = float(image_4d.min())
        vmax = float(image_4d.max())
        if vmin == vmax:
            vmax = vmin + 1
    
    print(f"Contrast limits: [{vmin}, {vmax}]")
    
    # Add the 4D image
    layer = viewer.add_image(
        image_4d,
        name="Calcium Imaging",
        colormap="gray",  # Gray is better for low-contrast calcium data
        contrast_limits=[vmin, vmax],
        scale=(1, 1, 1, 1),  # (T, Z, Y, X) - adjust if needed
        gamma=0.8,  # Slight gamma adjustment to brighten dim features
        rendering="mip",  # Maximum intensity projection for 3D view
        blending="translucent",
    )
    
    # Set smooth interpolation after layer creation
    layer.interpolation2d = "linear"
    layer.interpolation3d = "linear"
    
    print("\n✅ Viewer loaded successfully!")
    print("Controls:")
    print("  - Use time slider (T) to move through frames")
    print("  - Use Z slider to move through Z-slices")
    print("  - Scroll to zoom, drag to pan")
    
    # Start the viewer
    napari.run()


if __name__ == "__main__":
    main()
