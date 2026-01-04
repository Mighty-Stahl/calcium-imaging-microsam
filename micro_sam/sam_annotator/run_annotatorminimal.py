from __future__ import annotations

import argparse
import sys
import numpy as np

try:
    import napari
except Exception:
    napari = None

from micro_sam.sam_annotator.annotator_minimal import AnnotatorMinimal


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MicroSAM minimal annotator")
    parser.add_argument("--fake-data", action="store_true", help="Use synthetic 4D data")
    parser.add_argument("--path", type=str, default=None, help="Path to .npz file containing a 4D array (T,Z,Y,X)")
    parser.add_argument("--neuropal", type=str, default=None, help="Path to NeuroPAL .npz file (optional)")
    parser.add_argument("--T", type=int, default=6, help="timesteps")
    parser.add_argument("--Z", type=int, default=8, help="z slices")
    parser.add_argument("--Y", type=int, default=128, help="height")
    parser.add_argument("--X", type=int, default=128, help="width")
    parser.add_argument("--seed", type=int, default=0, help="random seed for fake data")
    args = parser.parse_args(argv)

    if napari is None:
        print("napari is required to run this demo. Install it with `pip install napari`")
        return 1

    image4d = None

    if args.path is not None:
        path = args.path
        print(f"Loading NPZ from: {path}")
        try:
            npz = np.load(path, mmap_mode='r')
        except TypeError:
            npz = np.load(path)
        except Exception as e:
            print(f"Failed to open NPZ: {e}")
            return 2

        if isinstance(npz, np.lib.npyio.NpzFile):
            keys = list(npz.files)
            chosen_key = None
            for candidate in ("image_4d", "data", "calcium"):
                if candidate in keys:
                    chosen_key = candidate
                    break
            if chosen_key is None and keys:
                chosen_key = keys[0]
            try:
                image4d = npz[chosen_key]
            except Exception as e:
                print(f"Failed to read array '{chosen_key}' from NPZ: {e}")
                return 2
        else:
            image4d = npz

    default_npz_path = "calcium_extracted_gaussian.npz"

    if image4d is None:
        print(f"Loading default NPZ from: {default_npz_path}")
        try:
            npz = np.load(default_npz_path, mmap_mode='r')
            if isinstance(npz, np.lib.npyio.NpzFile):
                keys = list(npz.files)
                print(f"NPZ keys found: {keys}")
                key = "data" if "data" in keys else keys[0]
                image4d = npz[key]
            else:
                image4d = npz
        except Exception as e:
            print(f"Failed to load default NPZ file: {e}")
            return 2

    # Validate shape
    if getattr(image4d, 'shape', None) is None or len(image4d.shape) != 4:
        print("Input array must be 4D (T,Z,Y,X)")
        return 2

    viewer = napari.Viewer()
    annot = AnnotatorMinimal(viewer)
    try:
        viewer.window.add_dock_widget(annot, area="right")
    except Exception:
        pass

    annot.update_image(image4d)
    print("📥 Minimal annotator loaded")
    
    # Load NeuroPAL if provided
    if args.neuropal:
        print(f"Loading NeuroPAL from: {args.neuropal}")
        try:
            neuropal_npz = np.load(args.neuropal)
            if "neuropal" in neuropal_npz:
                neuropal_data = neuropal_npz["neuropal"]
                annot.add_neuropal_layer(neuropal_data)
            else:
                print(f"Warning: 'neuropal' key not found in NPZ. Available keys: {list(neuropal_npz.keys())}")
        except Exception as e:
            print(f"Failed to load NeuroPAL: {e}")

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
