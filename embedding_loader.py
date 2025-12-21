"""
embedding_loader.py

Standalone script to compute SAM embeddings for 4D data (T,Z,Y,X).
Works on Windows/Mac/Linux with progress bar in terminal.

Usage:
    python embedding_loader.py --input data.npz --output embeddings.zarr --model vit_h --start 0 --end 5

Or run interactively:
    python embedding_loader.py
"""

import argparse
import numpy as np
from pathlib import Path
import zarr
from tqdm import tqdm
import sys


def load_image_data(filepath):
    """Load 4D image data from .npz or .npy file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        with np.load(filepath) as data:
            # Try common keys
            if 'image_4d' in data:
                image_4d = data['image_4d']
            elif 'image' in data:
                image_4d = data['image']
            elif 'data' in data:
                image_4d = data['data']
            else:
                # Take first array
                key = list(data.keys())[0]
                print(f"Using key '{key}' from npz file")
                image_4d = data[key]
    elif filepath.suffix == '.npy':
        image_4d = np.load(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .npz or .npy")
    
    # Ensure 4D
    if image_4d.ndim != 4:
        raise ValueError(f"Expected 4D data (T,Z,Y,X), got shape {image_4d.shape}")
    
    return image_4d


def compute_embeddings(image_4d, output_path, input_path=None, model_type='vit_h', device=None, 
                       start_t=None, end_t=None, tile_shape=None, halo=None):
    """
    Compute SAM embeddings for 4D data and save to zarr.
    
    Parameters:
    -----------
    input_path : str or Path, optional
        Input file path (for display purposes)
    image_4d : np.ndarray
        4D array (T, Z, Y, X)
    output_path : str or Path
        Output zarr file path
    model_type : str
        SAM model type: 'vit_h', 'vit_l', or 'vit_b'
    device : str, optional
        'cuda' or 'cpu'. Auto-detect if None.
    start_t : int, optional
        Start timestep (inclusive). Default: 0
    end_t : int, optional
        End timestep (exclusive). Default: n_timesteps
    tile_shape : tuple, optional
        Tile shape for large images (Z, Y, X). Auto if None.
    halo : tuple, optional
        Halo for tiling (Z, Y, X). Auto if None.
    """
    from micro_sam import util as sam_util
    import torch
    
    T, Z, Y, X = image_4d.shape
    
    # Set timestep range
    if start_t is None:
        start_t = 0
    if end_t is None:
        end_t = T
    
    start_t = max(0, start_t)
    end_t = min(T, end_t)
    
    if start_t >= end_t:
        raise ValueError(f"Invalid timestep range: start={start_t}, end={end_t}")
    
    n_timesteps = end_t - start_t
    
    print(f"\n{'='*60}")
    print(f"🔧 Embedding Computation Setup")
    print(f"{'='*60}")
    print(f"  Input shape (T,Z,Y,X): {image_4d.shape}")
    print(f"  Timestep range: {start_t} to {end_t-1} ({n_timesteps} timesteps)")
    print(f"  SAM model: {model_type}")
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Initialize SAM predictor
    print(f"\n📥 Loading SAM model '{model_type}'...")
    predictor = sam_util.get_sam_model(model_type=model_type, device=device)
    
    # Create zarr store
    output_path = Path(output_path)
    if output_path.exists():
        print(f"⚠️  Output file exists, will overwrite: {output_path}")
    
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    
    print(f"\n🔄 Computing embeddings...")
    print(f"{'='*60}\n")
    
    # Compute embeddings per timestep with progress bar
    for t in tqdm(range(start_t, end_t), desc="Timesteps", unit="t", ncols=80):
        image_3d = image_4d[t]  # (Z, Y, X)
        
        # Create timestep-specific output path
        timestep_path = output_path / f"t{t:04d}.zarr"
        
        # Compute embeddings for this timestep
        sam_util.precompute_image_embeddings(
            predictor=predictor,
            input_=image_3d,
            save_path=str(timestep_path),
            ndim=3,
            tile_shape=tile_shape,
            halo=halo,
            verbose=False  # Suppress per-slice output
        )
    
    print(f"\n{'='*60}")
    print(f"✅ Embedding computation complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_path}")
    print(f"  Timesteps: {start_t} to {end_t-1}")
    print(f"  Files: t{start_t:04d}.zarr to t{end_t-1:04d}.zarr")
    print(f"\nTo use in annotator:")
    input_name = Path(input_path).name if input_path else 'your_data.npz'
    print(f"  1. Load image: {input_name}")
    print(f"  2. Select embedding folder: {output_path.name}")
    print(f"  3. Start segmentation")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute SAM embeddings for 4D data (T,Z,Y,X)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute all timesteps
  python embedding_loader.py --input data.npz --output embeddings.zarr
  
  # Compute timesteps 0-4 (5 timesteps)
  python embedding_loader.py --input data.npz --output embeddings.zarr --start 0 --end 5
  
  # Use smaller model (faster, less accurate)
  python embedding_loader.py --input data.npz --output embeddings.zarr --model vit_b
  
  # Force CPU (no GPU)
  python embedding_loader.py --input data.npz --output embeddings.zarr --device cpu
  
  # Interactive mode (prompts for input)
  python embedding_loader.py
        """
    )
    
    parser.add_argument('--input', type=str, help='Input .npz or .npy file (T,Z,Y,X)')
    parser.add_argument('--output', type=str, help='Output .zarr file for embeddings')
    parser.add_argument('--model', type=str, default='vit_h', 
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type (default: vit_h)')
    parser.add_argument('--device', type=str, default=None, 
                       choices=['cuda', 'cpu'],
                       help='Device (auto-detect if not specified)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start timestep (inclusive, default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End timestep (exclusive, default: all)')
    parser.add_argument('--tile-z', type=int, default=None,
                       help='Tile Z size for large images (optional)')
    parser.add_argument('--tile-y', type=int, default=None,
                       help='Tile Y size for large images (optional)')
    parser.add_argument('--tile-x', type=int, default=None,
                       help='Tile X size for large images (optional)')
    
    args = parser.parse_args()
    
    # Interactive mode if no input provided
    if args.input is None:
        print("\n" + "="*60)
        print("🔧 SAM Embedding Loader - Interactive Mode")
        print("="*60)
        
        args.input = input("\nInput file path (.npz or .npy): ").strip()
        if not args.input:
            print("❌ No input file specified")
            sys.exit(1)
        
        args.output = input("Output zarr path (default: embeddings.zarr): ").strip()
        if not args.output:
            args.output = "embeddings.zarr"
        
        model_choice = input("SAM model [h/l/b] (default: h for vit_h): ").strip().lower()
        if model_choice == 'l':
            args.model = 'vit_l'
        elif model_choice == 'b':
            args.model = 'vit_b'
        else:
            args.model = 'vit_h'
        
        start_input = input("Start timestep (default: 0): ").strip()
        args.start = int(start_input) if start_input else None
        
        end_input = input("End timestep [EXCLUDED] (default: all): ").strip()
        args.end = int(end_input) if end_input else None
        
        print()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Load image data
    print(f"📂 Loading image data from {input_path}...")
    try:
        image_4d = load_image_data(input_path)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # Prepare tile shape
    tile_shape = None
    if args.tile_z is not None and args.tile_y is not None and args.tile_x is not None:
        tile_shape = (args.tile_z, args.tile_y, args.tile_x)
    
    # Compute embeddings
    try:
        compute_embeddings(
            image_4d=image_4d,
            output_path=args.output,
            input_path=args.input,
            model_type=args.model,
            device=args.device,
            start_t=args.start,
            end_t=args.end,
            tile_shape=tile_shape
        )
    except Exception as e:
        print(f"\n❌ Error during embedding computation:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
