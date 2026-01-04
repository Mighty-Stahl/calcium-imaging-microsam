import sys
from pathlib import Path

import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QFileDialog
from napari.utils.notifications import show_info


class AnnotatorMinimal(QtWidgets.QWidget):
    """A minimal annotator dock for Napari.

    Features:
    - Single image layer (the calcium imaging 4D array is added to the viewer)
    - Sidebar with a single button: "Load NeuroPAL Prompts"
    - When prompts are loaded, creates point layers `point_prompts` and `Neuron_Names`
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self.current_timestep = 0

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        title = QtWidgets.QLabel("<b>Annotator (Minimal)</b>")
        layout.addWidget(title)

        # Load NeuroPAL layer button
        btn_row_neuropal = QtWidgets.QWidget()
        btn_layout_neuropal = QtWidgets.QHBoxLayout()
        btn_row_neuropal.setLayout(btn_layout_neuropal)
        btn_load_neuropal_layer = QtWidgets.QPushButton("Load NeuroPAL Layer")
        btn_layout_neuropal.addWidget(btn_load_neuropal_layer)
        layout.addWidget(btn_row_neuropal)

        # Load neuropal prompts button
        btn_row = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout()
        btn_row.setLayout(btn_layout)
        btn_load_neuropal = QtWidgets.QPushButton("Load NeuroPAL Prompts")
        btn_layout.addWidget(btn_load_neuropal)
        layout.addWidget(btn_row)

        # Storage for prompts
        self.point_prompts_4d = {}
        self.neuron_names_map = {}

        def _load_neuropal_layer():
            """Load NeuroPAL NPZ file as an image layer."""
            try:
                npz_path = QFileDialog.getOpenFileName(
                    None,
                    "Select NeuroPAL NPZ File",
                    str(Path.home()),
                    "NPZ files (*.npz);;All files (*.*)",
                )[0]
                
                if not npz_path:
                    return
                
                print(f"Loading NeuroPAL from: {npz_path}")
                neuropal_npz = np.load(npz_path)
                
                if "neuropal" in neuropal_npz:
                    neuropal_data = neuropal_npz["neuropal"]
                    self.add_neuropal_layer(neuropal_data)
                else:
                    show_info(f"'neuropal' key not found. Available: {list(neuropal_npz.keys())}")
                    
            except Exception as e:
                show_info(f"Failed to load NeuroPAL layer: {e}")

        btn_load_neuropal_layer.clicked.connect(_load_neuropal_layer)

        def _load_neuropal_prompts():
            try:
                neuropal_path = Path(__file__).parent.parent / "Neuropal Coordinate Matching"
                if str(neuropal_path) not in sys.path:
                    sys.path.insert(0, str(neuropal_path))

                from load_neuropal_prompts import load_neuropal_prompts, create_neuron_name_layer_data  # type: ignore

                directory = QFileDialog.getExistingDirectory(
                    None,
                    "Select Directory Containing NeuroPAL Prompts",
                    str(Path.home()),
                )
                if not directory:
                    return

                result = load_neuropal_prompts(directory, timestep=0)
                if not result['success']:
                    show_info(f"Failed to load NeuroPAL prompts:\n{result['error']}")
                    return

                t = result['timestep']
                prompts = result['prompts']  # Shape: (N, 3) as [Z, Y, X]
                ids = result['ids']
                names = result['names']

                self.point_prompts_4d[t] = prompts
                self.neuron_names_map = names

                # Switch viewer to timestep 0 if dims available
                try:
                    self._viewer.dims.current_step = (0,) + self._viewer.dims.current_step[1:]
                    self.current_timestep = 0
                except Exception:
                    pass

                # Convert 3D coordinates to 4D by prepending timestep dimension
                # prompts: (N, 3) [Z, Y, X] -> prompts_4d: (N, 4) [T, Z, Y, X]
                timestep_col = np.full((len(prompts), 1), t)
                
                # Apply manual offset correction (diagnostic-determined)
                # NeuroPAL and calcium have different field of view
                MANUAL_OFFSET = np.array([0, 13, 83])  # [ΔZ, ΔY, ΔX]
                prompts_corrected = prompts + MANUAL_OFFSET
                
                prompts_4d = np.hstack([timestep_col, prompts_corrected])  # Shape: (N, 4)
                
                print(f"   Applied offset correction: {MANUAL_OFFSET}")
                print(f"   Original prompts: {prompts}")
                print(f"   Corrected prompts: {prompts_corrected}")

                # Create text labels for neuron names
                text_labels = [names.get(int(pid), f"ID_{pid}") for pid in ids]
                
                # Create properties for hover info (using 3D coordinates for properties)
                properties = {
                    'neuron_id': [names.get(int(pid), f"ID_{pid}") for pid in ids],
                    'point_id': ids.tolist(),
                    'z': prompts[:, 0].tolist(),
                    'y': prompts[:, 1].tolist(),
                    'x': prompts[:, 2].tolist(),
                }
                
                # Text parameters for napari (4D translation)
                text_params = {
                    'string': text_labels,
                    'size': 12,
                    'color': 'yellow',
                    'anchor': 'center',
                    'translation': np.array([0, 0, 0, 10])  # 4D: [T, Z, Y, X offset]
                }
                
                if "point_prompts" in self._viewer.layers:
                    layer = self._viewer.layers["point_prompts"]
                    layer.data = prompts_4d
                    layer.text = text_params
                    layer.properties = properties
                    layer.visible = True
                else:
                    self._viewer.add_points(
                        prompts_4d,
                        name="point_prompts",
                        size=10,
                        face_color='red',
                        text=text_params,
                        properties=properties,
                        ndim=4,
                    )

                # Focus the prompts layer
                try:
                    self._viewer.layers.selection.active = self._viewer.layers["point_prompts"]
                    self._viewer.reset_view()
                except Exception:
                    pass

                show_info(f"✓ Loaded {result['num_prompts']} NeuroPAL prompts (timestep {t})")

            except ImportError as e:
                show_info(
                    f"Failed to import NeuroPAL loader:\n{str(e)}\n\n"
                    "Make sure load_neuropal_prompts.py exists in:\n"
                    "micro_sam/Neuropal Coordinate Matching/"
                )
            except Exception as e:
                show_info(f"Failed to load NeuroPAL prompts:\n{str(e)}")

        btn_load_neuropal.clicked.connect(_load_neuropal_prompts)

    def update_image(self, image4d: np.ndarray):
        """Add the provided 4D image array to the Napari viewer as a single layer."""
        # Remove existing image if present
        try:
            if "image_4d" in self._viewer.layers:
                self._viewer.layers.remove(self._viewer.layers["image_4d"])
        except Exception:
            pass

        # Add image (Napari supports 4D arrays). Let Napari handle axes.
        try:
            self._viewer.add_image(image4d, name="image_4d", colormap='gray')
        except Exception:
            # Fallback: try adding a max projection if add_image fails
            try:
                proj = np.nanmax(image4d, axis=1)
                self._viewer.add_image(proj, name="image_2d_maxproj", colormap='gray')
            except Exception:
                show_info("Failed to add image to viewer")
    
    def add_neuropal_layer(self, neuropal_data: np.ndarray):
        """Add NeuroPAL image as a separate layer (RGB or grayscale)."""
        try:
            if "NeuroPAL" in self._viewer.layers:
                self._viewer.layers.remove(self._viewer.layers["NeuroPAL"])
        except Exception:
            pass
        
        try:
            # Take single timepoint if 4D
            if neuropal_data.ndim == 5:
                neuropal_data = neuropal_data[0]  # (T, Z, Y, X, C) -> (Z, Y, X, C)
            elif neuropal_data.ndim == 4 and neuropal_data.shape[0] > 50:
                # Likely (T, Z, Y, X) - take first timepoint
                neuropal_data = neuropal_data[0]
            
            # Determine if RGB or grayscale
            is_rgb = neuropal_data.ndim == 4 and neuropal_data.shape[-1] in [3, 4]
            
            # Add as RGB or grayscale layer
            self._viewer.add_image(
                neuropal_data,
                name="NeuroPAL",
                rgb=is_rgb,
                opacity=0.7,
                blending='additive',
                colormap='gray' if not is_rgb else None,
            )
            layer_type = "RGB" if is_rgb else "Grayscale"
            show_info(f"✓ Loaded NeuroPAL layer ({layer_type}, shape: {neuropal_data.shape})")
        except Exception as e:
            show_info(f"Failed to add NeuroPAL layer: {e}")
