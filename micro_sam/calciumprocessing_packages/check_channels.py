#!/usr/bin/env python3
"""Diagnostic script to identify which channel index corresponds to which wavelength."""

import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt

# ===== HARDCODED PARAMETERS - EDIT THESE =====
NWB_PATH = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h2/sub-20220327-h2_ses-20220327_ophys.nwb"
SERIES_NAME = "CalciumImageSeries"
# =============================================

def main():
    print(f"Opening NWB file: {NWB_PATH}")
    
    # Open NWB file
    io = NWBHDF5IO(NWB_PATH, "r")
    nwb = io.read()
    series = nwb.acquisition[SERIES_NAME]
    
    print(f"Series shape: {series.data.shape}")
    print(f"Rate: {series.rate} Hz")
    
    # Get one frame (middle timestep, middle Z-slice)
    t = series.data.shape[0] // 2
    z = series.data.shape[3] // 2
    
    print(f"\nExtracting frame at t={t}, z={z}")
    frame = np.asarray(series.data[t, :, :, z, :])  # Shape: (Y, X, 3)
    
    print(f"Frame shape: {frame.shape} (Y, X, Channels)")
    
    # Display all 3 channels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channel_names = ['Channel 0 (Red in RGB)', 'Channel 1 (Green in RGB)', 'Channel 2 (Blue in RGB)']
    
    print("\n" + "="*60)
    print("CHANNEL STATISTICS:")
    print("="*60)
    
    for i in range(3):
        ch_data = frame[:, :, i]
        axes[i].imshow(ch_data, cmap='gray', vmin=0, vmax=np.percentile(ch_data, 99))
        axes[i].set_title(
            f'{channel_names[i]}\n'
            f'Max: {ch_data.max()}, Mean: {ch_data.mean():.1f}',
            fontsize=10
        )
        axes[i].axis('off')
        
        # Print detailed stats
        print(f"\n{channel_names[i]}:")
        print(f"  Min:  {ch_data.min()}")
        print(f"  Max:  {ch_data.max()}")
        print(f"  Mean: {ch_data.mean():.2f}")
        print(f"  Std:  {ch_data.std():.2f}")
        print(f"  Median: {np.median(ch_data):.2f}")
        print(f"  99th percentile: {np.percentile(ch_data, 99):.2f}")
    
    print("\n" + "="*60)
    
    # Find brightest channel
    means = [frame[:,:,i].mean() for i in range(3)]
    brightest_idx = np.argmax(means)
    print(f"\n✨ BRIGHTEST CHANNEL: {brightest_idx} ({channel_names[brightest_idx]})")
    print(f"   → This is likely your calcium signal (488nm GCaMP)")
    
    plt.tight_layout()
    plt.savefig('channel_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved channel_comparison.png")
    plt.show()
    
    io.close()
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print(f"Use CHANNEL = {brightest_idx} in nwb_to_npz.py")
    print("(This is the channel with highest mean intensity)")
    print("="*60)


if __name__ == "__main__":
    main()
