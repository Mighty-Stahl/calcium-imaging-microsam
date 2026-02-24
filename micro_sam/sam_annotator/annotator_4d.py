import numpy as np
import threading
from qtpy import QtWidgets
from qtpy.QtGui import QKeySequence, QCursor
from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QFileDialog
from napari.utils.notifications import show_info
from micro_sam.sam_annotator.annotator_3d import Annotator3d
from micro_sam.sam_annotator._state import AnnotatorState
from pathlib import Path
import json
from micro_sam import instance_segmentation
from .util import _load_amg_state, _load_is_state
from . import util as _vutil
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from skimage.transform import resize as _sk_resize
from scipy.ndimage import distance_transform_edt, maximum_filter, label, center_of_mass, gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

class ObjectCommitWidget(QtWidgets.QWidget):
    """Widget to list current objects and commit them individually."""
    def __init__(self, annotator, parent=None):
        super().__init__(parent)
        self._annotator = annotator
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QtWidgets.QLabel("<b>Current Objects - Commit Individual IDs</b>")
        layout.addWidget(title)
        
        # Refresh button
        btn_refresh = QtWidgets.QPushButton("Refresh Object List")
        btn_refresh.clicked.connect(self.refresh_object_list)
        layout.addWidget(btn_refresh)
        
        # Scrollable area for object list
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)  # Set minimum height for better visibility
        scroll_widget = QtWidgets.QWidget()
        self._objects_layout = QtWidgets.QVBoxLayout()
        scroll_widget.setLayout(self._objects_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Store object entry widgets
        self._object_entries = []
        
    def refresh_object_list(self):
        """Refresh the list of objects from current_object_4d layer."""
        # Clear existing entries
        for entry in self._object_entries:
            try:
                self._objects_layout.removeWidget(entry["widget"])
                entry["widget"].deleteLater()
            except Exception:
                pass
        self._object_entries.clear()
        
        # Get current timestep and objects
        try:
            t = int(getattr(self._annotator, "current_timestep", 0) or 0)
            if self._annotator.current_object_4d is None:
                return
            
            objects = self._annotator.current_object_4d[t]
            unique_ids = np.unique(objects)
            unique_ids = unique_ids[unique_ids > 0]  # Exclude background
            
            if len(unique_ids) == 0:
                label = QtWidgets.QLabel("No objects in current timestep")
                self._objects_layout.addWidget(label)
                self._object_entries.append({"widget": label})
                return
            
            # Create entry for each object
            for obj_id in unique_ids:
                self._add_object_entry(int(obj_id), t)
                
        except Exception as e:
            print(f"Failed to refresh object list: {e}")
    
    def _add_object_entry(self, obj_id: int, timestep: int):
        """Add a single object entry with commit button."""
        try:
            container = QtWidgets.QWidget()
            hlayout = QtWidgets.QHBoxLayout()
            container.setLayout(hlayout)
            
            label = QtWidgets.QLabel(f"Object ID {obj_id}:")
            btn_commit = QtWidgets.QPushButton(f"Commit ID {obj_id}")
            
            # Create a proper slot function to avoid lambda issues
            def commit_callback():
                self._commit_single_object(obj_id, timestep)
            
            # Connect commit button
            btn_commit.clicked.connect(commit_callback)
            
            hlayout.addWidget(label)
            hlayout.addWidget(btn_commit)
            
            self._objects_layout.addWidget(container)
            self._object_entries.append({"widget": container, "id": obj_id})
            
        except Exception as e:
            print(f"Failed to add object entry: {e}")
    
    def _commit_single_object(self, obj_id: int, timestep: int):
        """Commit a single object ID from ALL timesteps in current_object_4d to committed_objects_4d."""
        try:
            if self._annotator.current_object_4d is None:
                show_info("No current objects to commit")
                return
            
            # Check if object exists in ANY timestep
            n_timesteps = self._annotator.n_timesteps
            found_in_any = False
            for t in range(n_timesteps):
                if np.any(self._annotator.current_object_4d[t] == obj_id):
                    found_in_any = True
                    break
            
            if not found_in_any:
                show_info(f"Object ID {obj_id} not found in any timestep")
                return
            
            # Ensure committed_objects_4d exists
            if self._annotator.segmentation_4d is None:
                self._annotator.segmentation_4d = np.zeros_like(self._annotator.image_4d, dtype=np.uint32)
            
            # Find next available ID in committed layer
            max_id = int(self._annotator.segmentation_4d.max())
            new_id = max_id + 1
            
            # Copy this object from ALL timesteps to committed layer
            timesteps_committed = []
            for t in range(n_timesteps):
                mask = self._annotator.current_object_4d[t] == obj_id
                if np.any(mask):
                    self._annotator.segmentation_4d[t][mask] = new_id
                    self._annotator.current_object_4d[t][mask] = 0
                    timesteps_committed.append(t)
            
            # Remove point prompts associated with this object ID from ALL timesteps
            total_prompts_removed = 0
            try:
                # Remove from all timesteps where this object exists
                for t in timesteps_committed:
                    # Remove from point_prompts_4d storage
                    if hasattr(self._annotator, 'point_prompts_4d') and t in self._annotator.point_prompts_4d:
                        pts = self._annotator.point_prompts_4d[t]
                        if pts is not None and len(pts) > 0:
                            point_ids = self._annotator._get_point_ids_for_timestep(t, pts)
                            points_to_keep = []
                            for point, pid in zip(pts, point_ids):
                                if pid != obj_id:
                                    points_to_keep.append(point)
                                else:
                                    # Remove from coordinate map
                                    z, y, x = int(point[0]), int(point[1]), int(point[2])
                                    key = (t, z, y, x)
                                    if key in self._annotator.point_id_map:
                                        del self._annotator.point_id_map[key]
                                    total_prompts_removed += 1
                            
                            self._annotator.point_prompts_4d[t] = np.array(points_to_keep) if points_to_keep else np.empty((0, 3))
                
                # Also update current visible layer if on a committed timestep
                current_t = getattr(self._annotator, 'current_timestep', 0)
                if current_t in timesteps_committed and "point_prompts" in self._annotator._viewer.layers:
                    layer = self._annotator._viewer.layers["point_prompts"]
                    pts = self._annotator.point_prompts_4d.get(current_t, np.empty((0, 3)))
                    layer.data = pts
                
                if total_prompts_removed > 0:
                    print(f"Removed {total_prompts_removed} point prompt(s) for object ID {obj_id} across {len(timesteps_committed)} timestep(s)")
            except Exception as e:
                print(f"Warning: Could not remove point prompts: {e}")
            
            # Update layers
            try:
                committed_layer = self._annotator._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._annotator._viewer.layers else None
                if committed_layer is not None:
                    committed_layer.refresh()
                
                current_layer = self._annotator._viewer.layers["current_object_4d"] if "current_object_4d" in self._annotator._viewer.layers else None
                if current_layer is not None:
                    current_layer.refresh()
            except Exception:
                pass
            
            show_info(f"Committed object ID {obj_id} as new ID {new_id} across {len(timesteps_committed)} timestep(s)")
            
            # Refresh both the object list and point list
            self.refresh_object_list()
            if hasattr(self._annotator, "_point_manager_widget"):
                try:
                    self._annotator._point_manager_widget.refresh_point_list()
                except Exception:
                    pass
            
        except Exception as e:
            print(f"Failed to commit object {obj_id}: {e}")
            show_info(f"Failed to commit object: {e}")


class PointPromptManagerWidget(QtWidgets.QWidget):
    """Widget to manage point prompt IDs with dropdown list for each point."""
    def __init__(self, annotator, parent=None):
        super().__init__(parent)
        self._annotator = annotator
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QtWidgets.QLabel("<b>Point Prompts - Assign IDs</b>")
        layout.addWidget(title)
        
        # Refresh button
        btn_refresh = QtWidgets.QPushButton("Refresh Point List")
        btn_refresh.clicked.connect(self.refresh_point_list)
        layout.addWidget(btn_refresh)
        
        # Scrollable area for point list
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)  # Set minimum height for better visibility
        scroll_widget = QtWidgets.QWidget()
        self._points_layout = QtWidgets.QVBoxLayout()
        scroll_widget.setLayout(self._points_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Store point entry widgets
        self._point_entries = []
    
    def refresh_point_list(self):
        """Refresh the list of point prompts for current timestep."""
        # Clear existing entries
        for entry in self._point_entries:
            try:
                self._points_layout.removeWidget(entry["widget"])
                entry["widget"].deleteLater()
            except Exception:
                pass
        self._point_entries.clear()
        
        # Get current timestep and points
        try:
            t = int(getattr(self._annotator, "current_timestep", 0) or 0)
            
            # Get points for this timestep
            if "point_prompts" not in self._annotator._viewer.layers:
                label = QtWidgets.QLabel("No point prompts layer found")
                self._points_layout.addWidget(label)
                self._point_entries.append({"widget": label})
                return
            
            layer = self._annotator._viewer.layers["point_prompts"]
            points = np.array(layer.data)
            
            if len(points) == 0:
                label = QtWidgets.QLabel("No points in current timestep")
                self._points_layout.addWidget(label)
                self._point_entries.append({"widget": label})
                return
            
            # Get IDs for all points based on their coordinates
            point_ids = self._annotator._get_point_ids_for_timestep(t, points)
            
            # Create entry for each point
            for idx, point in enumerate(points):
                self._add_point_entry(idx, point, point_ids[idx], t)
                
        except Exception as e:
            print(f"Failed to refresh point list: {e}")
    
    def _add_point_entry(self, idx: int, point, current_id: int, timestep: int):
        """Add a single point entry with ID dropdown."""
        try:
            container = QtWidgets.QWidget()
            hlayout = QtWidgets.QHBoxLayout()
            container.setLayout(hlayout)
            
            # Point label with coordinates
            z, y, x = int(point[0]), int(point[1]), int(point[2])
            label = QtWidgets.QLabel(f"Point {idx+1} ({z},{y},{x}):")
            label.setMinimumWidth(150)
            
            # ID dropdown
            id_label = QtWidgets.QLabel("ID:")
            id_spinbox = QtWidgets.QSpinBox()
            id_spinbox.setRange(1, 1000)
            id_spinbox.setValue(current_id)
            id_spinbox.setMinimumWidth(60)
            
            # Create callback for this specific point
            def on_id_changed(value):
                self._update_point_id(idx, value, timestep)
            
            id_spinbox.valueChanged.connect(on_id_changed)
            
            hlayout.addWidget(label)
            hlayout.addWidget(id_label)
            hlayout.addWidget(id_spinbox)
            hlayout.addStretch()
            
            self._points_layout.addWidget(container)
            self._point_entries.append({
                "widget": container, 
                "idx": idx, 
                "spinbox": id_spinbox
            })
            
        except Exception as e:
            print(f"Failed to add point entry: {e}")
    
    def _update_point_id(self, point_idx: int, new_id: int, timestep: int):
        """Update the ID for a specific point."""
        try:
            # Get point coordinates from the layer
            if "point_prompts" not in self._annotator._viewer.layers:
                return
            
            layer = self._annotator._viewer.layers["point_prompts"]
            points = np.array(layer.data)
            
            if point_idx >= len(points):
                return
            
            point = points[point_idx]
            z, y, x = int(point[0]), int(point[1]), int(point[2])
            
            # Store ID using coordinates
            self._annotator._set_point_id_from_coords(timestep, z, y, x, new_id)
            
            # Update colors for all points in the layer
            all_point_ids = self._annotator._get_point_ids_for_timestep(timestep, points)
            self._annotator._update_point_colors(layer, all_point_ids)
            
            print(f"Updated point {point_idx} at ({z},{y},{x}) to ID {new_id}")
            
        except Exception as e:
            print(f"Failed to update point ID: {e}")


class TimestepToolsWidget(QtWidgets.QWidget):
    """Simple UI widget providing 4D timestep operations for manual workflows."""
    def __init__(self, annotator, parent=None):
        super().__init__(parent)
        self._annotator = annotator
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        btn_segment = QtWidgets.QPushButton("Segment Object (Shift+S)")
        btn_commit = QtWidgets.QPushButton("Commit Object (C)")
        btn_propagate_points = QtWidgets.QPushButton("Propagate point prompts to all timesteps")

        btn_segment.clicked.connect(lambda: self._safe_call(self._annotator.segment_all_timesteps))
        btn_commit.clicked.connect(lambda: self._safe_call(self._annotator.commit_all_timesteps))
        btn_propagate_points.clicked.connect(lambda: self._safe_call(self._annotator.propagate_point_prompts_to_all_timesteps))

        layout.addWidget(btn_segment)
        layout.addWidget(btn_commit)
        layout.addWidget(btn_propagate_points)
        
        # Add copy point prompts UI
        copy_label = QtWidgets.QLabel("Copy points from T:")
        copy_row = QtWidgets.QWidget()
        copy_layout = QtWidgets.QHBoxLayout()
        copy_row.setLayout(copy_layout)
        
        self._copy_timestep_spinbox = QtWidgets.QSpinBox()
        self._copy_timestep_spinbox.setMinimum(0)
        self._copy_timestep_spinbox.setMaximum(999)
        self._copy_timestep_spinbox.setValue(0)
        self._copy_timestep_spinbox.setToolTip("Source timestep to copy points from")
        
        btn_copy_points = QtWidgets.QPushButton("Copy to Current T")
        btn_copy_points.clicked.connect(self._copy_point_prompts)
        
        copy_layout.addWidget(copy_label)
        copy_layout.addWidget(self._copy_timestep_spinbox)
        copy_layout.addWidget(btn_copy_points)
        
        layout.addWidget(copy_row)
        
        # Add crop object UI
        crop_label = QtWidgets.QLabel("<b>Crop Object by Bounds:</b>")
        layout.addWidget(crop_label)
        
        # Object ID to crop
        id_row = QtWidgets.QWidget()
        id_layout = QtWidgets.QHBoxLayout()
        id_row.setLayout(id_layout)
        id_layout.addWidget(QtWidgets.QLabel("Object ID:"))
        self._crop_id_spinbox = QtWidgets.QSpinBox()
        self._crop_id_spinbox.setMinimum(1)
        self._crop_id_spinbox.setMaximum(9999)
        self._crop_id_spinbox.setValue(1)
        id_layout.addWidget(self._crop_id_spinbox)
        layout.addWidget(id_row)
        
                # Z range
        z_row = QtWidgets.QWidget()
        z_layout = QtWidgets.QHBoxLayout()
        z_row.setLayout(z_layout)
        z_layout.addWidget(QtWidgets.QLabel("Z range:"))
        self._crop_z_min = QtWidgets.QSpinBox()
        self._crop_z_min.setMinimum(0)
        self._crop_z_min.setMaximum(9999)
        self._crop_z_min.setValue(0)
        z_layout.addWidget(self._crop_z_min)
        z_layout.addWidget(QtWidgets.QLabel("to"))
        self._crop_z_max = QtWidgets.QSpinBox()
        self._crop_z_max.setMinimum(0)
        self._crop_z_max.setMaximum(9999)
        self._crop_z_max.setValue(9999)
        z_layout.addWidget(self._crop_z_max)
        layout.addWidget(z_row)
        
        # Crop button
        btn_crop = QtWidgets.QPushButton("Crop Object")
        btn_crop.clicked.connect(self._crop_object)
        layout.addWidget(btn_crop)
    
    def _crop_object(self):
        """Crop an object in current_object_4d by Z axis bounds."""
        try:
            obj_id = int(self._crop_id_spinbox.value())
            z_min = int(self._crop_z_min.value())
            z_max = int(self._crop_z_max.value())
            
            if z_min > z_max:
                show_info("Invalid Z range: min value must be <= max value")
                return
            
            current_t = int(getattr(self._annotator, "current_timestep", 0) or 0)
            
            if self._annotator.current_object_4d is None:
                show_info("No current objects to crop")
                return
            
            # Get current timestep data
            seg_t = self._annotator.current_object_4d[current_t]
            
            # Check if object ID exists
            if not np.any(seg_t == obj_id):
                show_info(f"Object ID {obj_id} not found in current timestep")
                return
            
            # Create a mask for pixels to keep (within Z bounds)
            shape = seg_t.shape  # (Z, Y, X)
            
            # Create Z coordinate grid
            z_coords = np.arange(shape[0])[:, None, None]  # Shape (Z, 1, 1)
            z_coords = np.broadcast_to(z_coords, shape)
            
            # Mask for pixels within Z bounds
            in_bounds = (z_coords >= z_min) & (z_coords <= z_max)
            
            # Mask for this object
            obj_mask = (seg_t == obj_id)
            
            # Pixels to remove: object pixels outside bounds
            remove_mask = obj_mask & (~in_bounds)
            
            # Count pixels before and after
            pixels_before = np.count_nonzero(obj_mask)
            pixels_removed = np.count_nonzero(remove_mask)
            pixels_after = pixels_before - pixels_removed
            
            if pixels_removed == 0:
                show_info(f"Object ID {obj_id} is already within the specified bounds")
                return
            
            # Remove pixels outside bounds
            seg_t[remove_mask] = 0
            
            # Update the layer
            if "current_object_4d" in self._annotator._viewer.layers:
                layer = self._annotator._viewer.layers["current_object_4d"]
                try:
                    layer.refresh()
                except Exception:
                    pass
            
            show_info(f"Cropped object ID {obj_id}: removed {pixels_removed} pixels, kept {pixels_after} pixels")
            
        except Exception as e:
            show_info(f"Failed to crop object: {e}")
            print(f"Error cropping object: {e}")
    
    def _copy_point_prompts(self):
        """Copy point prompts from source timestep to current timestep."""
        try:
            source_t = int(self._copy_timestep_spinbox.value())
            current_t = int(getattr(self._annotator, "current_timestep", 0) or 0)
            
            if source_t == current_t:
                show_info("Source and current timestep are the same!")
                return
            
            if source_t >= self._annotator.n_timesteps:
                show_info(f"Source timestep {source_t} is out of range (max: {self._annotator.n_timesteps - 1})")
                return
            
            # Get points from source timestep
            if not hasattr(self._annotator, "point_prompts_4d"):
                self._annotator.point_prompts_4d = {}
            
            source_points = self._annotator.point_prompts_4d.get(source_t, np.empty((0, 3)))
            
            if len(source_points) == 0:
                show_info(f"No point prompts in timestep {source_t}")
                return
            
            # Copy points to current timestep
            copied_points = source_points.copy()
            self._annotator.point_prompts_4d[current_t] = copied_points
            
            # Copy IDs based on coordinates
            for point in source_points:
                z, y, x = int(point[0]), int(point[1]), int(point[2])
                source_id = self._annotator._get_point_id_from_coords(source_t, z, y, x)
                self._annotator._set_point_id_from_coords(current_t, z, y, x, source_id)
            
            # Update the points layer
            if "point_prompts" in self._annotator._viewer.layers:
                layer = self._annotator._viewer.layers["point_prompts"]
                if hasattr(layer, 'events') and hasattr(layer.events, 'data'):
                    with layer.events.data.blocker():
                        layer.data = copied_points
                else:
                    layer.data = copied_points
                
                # Update colors
                point_ids = self._annotator._get_point_ids_for_timestep(current_t, copied_points)
                self._annotator._update_point_colors(layer, point_ids)
            
            # Refresh point manager widget
            if hasattr(self._annotator, "_point_manager_widget"):
                self._annotator._point_manager_widget.refresh_point_list()
            
            show_info(f"Copied {len(copied_points)} point prompts from T={source_t} to T={current_t}")
            
        except Exception as e:
            show_info(f"Failed to copy point prompts: {e}")
            print(f"Error copying point prompts: {e}")

    def _safe_call(self, fn):
        try:
            fn()
        except Exception as e:
            try:
                show_info(f"Operation failed: {e}")
            except Exception:
                print(f"Operation failed: {e}")

def _select_array_from_zarr_group(f):
    """Pick a zarr.Array-like child from a zarr.Group.

    Preference order:
      - dataset named 'features'
      - common alternate names
      - first array-like child at depth 1
      - first array-like child inside a top-level group at depth 2

    Returns the array-like object or None if none found.
    """
    try:
        # Prefer explicit 'features'
        if "features" in f:
            return f["features"]
    except Exception:
        pass

    # common alternative names
    for alt in ("feats", "features_0", "features0", "arr", "data"):
        try:
            if alt in f:
                return f[alt]
        except Exception:
            pass

    # first pass: find first array-like child at depth 1
    try:
        for name, obj in f.items():
            try:
                if hasattr(obj, "ndim") or hasattr(obj, "shape"):
                    return obj
            except Exception:
                # if obj is a Group, try depth-2
                try:
                    for cname, cobj in obj.items():
                        if hasattr(cobj, "ndim") or hasattr(cobj, "shape"):
                            return cobj
                except Exception:
                    continue
    except Exception:
        pass

    return None



class TimestepEmbeddingManager:
    """Manage lazy loading of per-timestep embeddings stored as zarr files.

    This keeps only one materialized embedding in memory (the most recently
    used timestep). When the timestep changes we load the matching
    `embeddings_t{t}.zarr` lazily (in a background thread) and activate it on
    the parent annotator by calling `_ensure_embeddings_active_for_t(t)`.
    """

    def __init__(self, annotator, embeddings_dir: str | Path | None = None, lazy: bool = True):
        self.annotator = annotator
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir is not None else None
        self.lazy = bool(lazy)
        self._lock = threading.Lock()
        # cache for the currently materialized timestep
        self.cached_t = None
        self.cached_entry = None
        self.cached_path = None

    def set_embeddings_dir(self, path: str | Path):
        self.embeddings_dir = Path(path)

    def get_current_embedding(self):
        return self.cached_entry

    def on_timestep_changed(self, t: int):
        """Callback for timestep changes; triggers embedding loading and 4D-aware point updates."""
        self.current_timestep = t

        # --- update 4D point prompts ---
        try:
            if hasattr(self.annotator, "point_prompts_4d"):
                pts_t = np.array(self.annotator.point_prompts_4d.get(t, np.empty((0, 3))))
                if "point_prompts" in self.annotator._viewer.layers:
                    self.annotator._viewer.layers["point_prompts"].data = pts_t
        except Exception as e:
            print(f"[WARN] Failed updating 4D point prompts at timestep {t}: {e}")

        # --- background embedding loading ---
        try:
            thread = threading.Thread(
                target=self.load_embedding_for_timestep,
                args=(int(t),),
                daemon=True,
            )
            thread.start()
        except Exception:
            try:
                self.load_embedding_for_timestep(int(t))
            except Exception:
                pass

    def _make_zarr_path_for_t(self, t: int):
        # prefer explicit last dir from annotator
        p = None
        if getattr(self.annotator, "_last_embeddings_dir", None):
            p = Path(self.annotator._last_embeddings_dir)
        if p is None and self.embeddings_dir is not None:
            p = self.embeddings_dir
        if p is None:
            return None
        cand = p / f"embeddings_t{t}.zarr"
        if cand.exists():
            return cand
        # also accept alternative naming
        cand2 = p / f"t{t}.zarr"
        if cand2.exists():
            return cand2
        return None

    def load_embedding_for_timestep(self, t: int):
        """Load (lazily) the embedding for timestep `t` and activate it.

        This will materialize the zarr store (open it) and call
        `annotator._ensure_embeddings_active_for_t(t)` so the global state uses
        the newly-loaded embeddings. Only one embedding is kept materialized;
        the previous one is released (replaced with a path placeholder) if it
        was loaded from disk by this manager.
        """
        with self._lock:
            # If we already have the requested timestep cached, nothing to do
            if self.cached_t == int(t) and self.cached_entry is not None:
                return self.cached_entry

            # If annotator already knows about an embedding entry for t, prefer that
            try:
                existing = self.annotator.embeddings_4d.get(int(t))
                if existing is not None and isinstance(existing, dict) and "features" in existing:
                    # materialized already by other code; adopt as cache
                    self._release_cached_if_needed(exclude_t=int(t))
                    self.cached_t = int(t)
                    self.cached_entry = existing
                    self.cached_path = existing.get("path") if isinstance(existing, dict) else None
                    # ensure annotator binds it
                    try:
                        self.annotator._ensure_embeddings_active_for_t(int(t))
                    except Exception:
                        pass
                    return self.cached_entry
            except Exception:
                pass

            # Try to resolve zarr path
            zpath = self._make_zarr_path_for_t(int(t))
            if zpath is None:
                # Fallback: if annotator.embeddings_4d contains an entry (e.g., mapping produced by set_embeddings_folder), use it
                entry = self.annotator.embeddings_4d.get(int(t))
                if entry is not None:
                    # trigger annotator's activation path (it will materialize if needed)
                    try:
                        self.annotator._ensure_embeddings_active_for_t(int(t))
                    except Exception:
                        pass
                    return entry
                return None

            # Mark annotator mapping with a lazy path so other parts can see it
            try:
                self.annotator.embeddings_4d[int(t)] = {"path": str(zpath)}
            except Exception:
                pass

            # Materialize (open) the zarr in background thread context (we're already in a thread)
            try:
                import zarr as _zarr
                f = _zarr.open(str(zpath), mode="r")
                # Pick a suitable array-like dataset from the zarr group.
                feats = _select_array_from_zarr_group(f)
                if feats is None:
                    try:
                        show_info(f"No suitable array dataset found inside {zpath}; cannot load embeddings for timestep {t}.")
                    except Exception:
                        pass
                    return None
                attrs = getattr(feats, "attrs", {}) or {}
                input_size = attrs.get("input_size")
                original_size = attrs.get("original_size")
                if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                    try:
                        inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                        input_size = input_size or inferred
                        original_size = original_size or inferred
                    except Exception:
                        input_size = input_size or None
                        original_size = original_size or None

                entry = {"features": feats, "input_size": input_size, "original_size": original_size, "path": str(zpath)}

                # release old cached if it was loaded by manager
                self._release_cached_if_needed()

                # cache this one
                self.cached_t = int(t)
                self.cached_entry = entry
                self.cached_path = str(zpath)

                # store in annotator mapping and activate
                try:
                    self.annotator.embeddings_4d[int(t)] = entry
                except Exception:
                    pass
                try:
                    # call annotator activation so AnnotatorState binds this embedding
                    self.annotator._ensure_embeddings_active_for_t(int(t))
                except Exception:
                    pass
                return entry
            except Exception:
                try:
                    show_info(f"Failed to load embeddings for timestep {t} from {zpath}")
                except Exception:
                    pass
                return None

    def _release_cached_if_needed(self, exclude_t: int | None = None):
        """Release the currently cached embedding if it was loaded from disk by this manager.

        If the previous entry was materialized from a zarr on-disk store, replace
        it in `annotator.embeddings_4d` with a lightweight {'path': ...} mapping
        so memory can be reclaimed.
        """
        try:
            if self.cached_t is None:
                return
            if exclude_t is not None and int(self.cached_t) == int(exclude_t):
                return
            # Only replace if we have a cached_path (i.e., loaded from disk)
            if self.cached_path and self.cached_t is not None:
                try:
                    # replace materialized entry with a path-only placeholder
                    self.annotator.embeddings_4d[int(self.cached_t)] = {"path": str(self.cached_path)}
                except Exception:
                    try:
                        del self.annotator.embeddings_4d[int(self.cached_t)]
                    except Exception:
                        pass
            # clear local references so zarr can be freed by GC
            self.cached_t = None
            self.cached_entry = None
            self.cached_path = None
        except Exception:
            pass



class MicroSAM4DAnnotator(Annotator3d):
    """
    4D annotator for (T, Z, Y, X) time-series data.

    This class keeps a persistent 4D image layer (`raw_4d`) and a persistent
    4D labels layer (`committed_objects_4d`). Per-timestep interactive 3D
    layers (editable views) are created once and updated in-place when the
    Napari time slider (dims.current_step[0]) changes. Committing a
    segmentation writes the result back into the 4D label array and refreshes
    the visible 3D view.
    """

    def __init__(self, viewer):
        # mark this instance as a 4D annotator so base class will create _4d
        # container layers instead of per-timestep 3D layers during init
        try:
            # set flag before calling base init so _AnnotatorBase can detect 4D
            self._is_4d = True
        except Exception:
            pass
        super().__init__(viewer)
        self.current_timestep = 0
        self.image_4d = None
        self.use_preview = True
        # 4D arrays (T, Z, Y, X)
        self.segmentation_4d = None
        self.auto_segmentation_4d = None
        self.current_object_4d = None
        self.committed_objects_4d = None
        # 4D-aware point prompts: dict mapping timestep -> points array
        self.point_prompts_4d = {}
        # Persistent point-to-ID mapping: key=(t,z,y,x) -> value=ID
        self.point_id_map = {}
        # Counter for next point ID to assign
        self.next_point_id = 1
        self.n_timesteps = 0
        # small per-timestep cache (optional)
        self._segmentation_cache = None
        # per-timestep embeddings cache: mapping t -> embedding dict or lazy entry
        self.embeddings_4d = {}
        # flags for background materialization (t -> bool)
        self._embedding_loading = {}
        # currently-active timestep whose embeddings are bound to AnnotatorState
        self._active_embedding_t = None
        # track which timesteps we've shown an embedding info message for
        self._reported_embedding_info_t = set()

        # remember last directory where embeddings were saved/loaded
        self._last_embeddings_dir = None

        # Desired layer order (bottom to top)
        self._desired_layer_order = [
            "raw_4d",
            "current_object_4d",
            "auto_segmentation_4d",
            "committed_objects_4d",
            "point_prompts",
            "remap_points"
        ]

        # Timestep embedding manager for lazy per-timestep zarr loading
        try:
            self.timestep_embedding_manager = TimestepEmbeddingManager(self)
        except Exception:
            self.timestep_embedding_manager = None

        # Hide/remove unwanted base class UI widgets
        try:
            # The base class creates self._annotator_widget with various UI sections
            # We'll hide the ones we don't need for 4D annotation
            if hasattr(self, '_annotator_widget'):
                layout = self._annotator_widget.layout()
                if layout is not None:
                    # Remove all existing widgets from base class
                    while layout.count():
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
        except Exception as e:
            print(f"Failed to clean base class widgets: {e}")

        # Add small embedding controls to the annotator dock (Compute embeddings current/all T)
        try:
            emb_widget = QtWidgets.QGroupBox("EMBEDDINGS TOOLS")
            emb_widget.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #8000ff; 
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            emb_layout = QtWidgets.QVBoxLayout()
            emb_widget.setLayout(emb_layout)

            # Row with two compute buttons
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout()
            row.setLayout(row_layout)
            btn_current = QtWidgets.QPushButton("Compute embeddings (current T)")
            btn_all = QtWidgets.QPushButton("Compute embeddings (all T)")
            row_layout.addWidget(btn_current)
            row_layout.addWidget(btn_all)

            # Only add the compute buttons row
            emb_layout.addWidget(row)

            # Removed timestep list widget and activation controls

            def _compute_current():
                try:
                    t = int(getattr(self, "current_timestep", 0) or 0)
                    show_info(f"Computing embeddings for timestep {t} — this may take a while.")
                    self.compute_embeddings_for_timestep(t)
                    show_info("Embeddings computed for current timestep.")
                except Exception as e:
                    print(f"Failed to compute embeddings for timestep {t}: {e}")

            def _compute_all():
                try:
                    show_info("Computing embeddings for all timesteps — this may take a long time.")
                    self.compute_embeddings_for_all_timesteps()
                    show_info("Embeddings computed for all timesteps.")
                except Exception as e:
                    print(f"Failed to compute embeddings for all timesteps: {e}")

            btn_current.clicked.connect(_compute_current)
            btn_all.clicked.connect(_compute_all)

            # Add save and load embeddings buttons
            save_load_row = QtWidgets.QWidget()
            save_load_layout = QtWidgets.QHBoxLayout()
            save_load_row.setLayout(save_load_layout)
            btn_save = QtWidgets.QPushButton("Save embeddings")
            btn_save_current = QtWidgets.QPushButton("Save embeddings (current T)")
            btn_load = QtWidgets.QPushButton("Load embeddings from directory")
            save_load_layout.addWidget(btn_save)
            save_load_layout.addWidget(btn_save_current)
            save_load_layout.addWidget(btn_load)
            emb_layout.addWidget(save_load_row)

            def _save_embeddings():
                try:
                    # Select directory to save embeddings
                    directory = QFileDialog.getExistingDirectory(
                        None, 
                        "Select Directory to Save Embeddings",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    show_info(f"Computing and saving embeddings for all timesteps to {directory} — this may take a long time.")
                    
                    # Compute embeddings for all timesteps and save them
                    save_path = Path(directory)
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    for t in range(self.n_timesteps):
                        filename = save_path / f"t{t}.zarr"
                        show_info(f"Computing embeddings for timestep {t}...")
                        self.compute_embeddings_for_timestep(
                            t=t, 
                            save_path=str(filename)
                        )
                    
                    # Store the directory path for future reference
                    self._last_embeddings_dir = str(save_path)
                    show_info(f"All embeddings saved to {directory}")
                    
                except Exception as e:
                    print(f"Failed to save embeddings: {e}")
                    show_info(f"Failed to save embeddings: {e}")

            def _save_embeddings_current():
                try:
                    # Get current timestep
                    t = int(getattr(self, "current_timestep", 0) or 0)
                    
                    # Select directory to save embeddings
                    directory = QFileDialog.getExistingDirectory(
                        None, 
                        f"Select Directory to Save Embedding for Timestep {t}",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    save_path = Path(directory)
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    filename = save_path / f"t{t}.zarr"
                    show_info(f"Computing and saving embedding for timestep {t}...")
                    
                    self.compute_embeddings_for_timestep(
                        t=t, 
                        save_path=str(filename)
                    )
                    
                    # Store the directory path for future reference
                    self._last_embeddings_dir = str(save_path)
                    show_info(f"Embedding for timestep {t} saved to {filename}")
                    
                except Exception as e:
                    print(f"Failed to save embedding for current timestep: {e}")
                    show_info(f"Failed to save embedding: {e}")

            def _load_embeddings():
                try:
                    # Select directory containing embeddings
                    directory = QFileDialog.getExistingDirectory(
                        None,
                        "Select Directory Containing Embeddings",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    directory_path = Path(directory)
                    
                    # Check for t0, t1, t2, ... files
                    embedding_files = sorted(directory_path.glob("t*.zarr"))
                    if not embedding_files:
                        show_info(f"No embedding files (t0.zarr, t1.zarr, ...) found in {directory}")
                        return
                    
                    # Store lazy loading information
                    self._last_embeddings_dir = str(directory_path)
                    
                    # Initialize embeddings_4d with lazy entries (just store paths)
                    if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                        self.embeddings_4d = {}
                    
                    # Only load embeddings for timesteps that exist in the viewer
                    loaded_count = 0
                    for t in range(min(self.n_timesteps, len(embedding_files))):
                        filename = directory_path / f"t{t}.zarr"
                        if filename.exists():
                            # Store path for lazy loading
                            self.embeddings_4d[t] = {"path": str(filename)}
                            loaded_count += 1
                    
                    # Load embeddings for current timestep immediately
                    current_t = getattr(self, "current_timestep", 0)
                    if current_t in self.embeddings_4d:
                        self._load_embedding_for_timestep(current_t)
                    
                    show_info(f"Embeddings loaded for {loaded_count} timesteps")
                    
                except Exception as e:
                    print(f"Failed to load embeddings: {e}")
                    show_info(f"Failed to load embeddings: {e}")

            btn_save.clicked.connect(_save_embeddings)
            btn_save_current.clicked.connect(_save_embeddings_current)
            btn_load.clicked.connect(_load_embeddings)

            # Removed save/load embeddings folder functions and handlers

            # Insert the embedding widget at the top of the annotator layout
            try:
                self._annotator_widget.layout().insertWidget(0, emb_widget)
            except Exception:
                # fallback: add at end
                self._annotator_widget.layout().addWidget(emb_widget)

        except Exception as e:
            print(f"Failed to create embeddings tools: {e}")

        # Add POINT PROMPT TOOLS section
        try:
            point_widget = QtWidgets.QGroupBox("POINT PROMPT TOOLS")
            point_widget.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #00bfff; 
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            point_layout = QtWidgets.QVBoxLayout()
            point_widget.setLayout(point_layout)

            # Save and load buttons
            save_load_row = QtWidgets.QWidget()
            save_load_layout = QtWidgets.QHBoxLayout()
            save_load_row.setLayout(save_load_layout)
            btn_save_points = QtWidgets.QPushButton("Save point prompts")
            btn_load_points = QtWidgets.QPushButton("Load point prompts")
            save_load_layout.addWidget(btn_save_points)
            save_load_layout.addWidget(btn_load_points)
            point_layout.addWidget(save_load_row)

            # NeuroPAL prompts button (separate row)
            neuropal_row = QtWidgets.QWidget()
            neuropal_layout = QtWidgets.QHBoxLayout()
            neuropal_row.setLayout(neuropal_layout)
            btn_load_neuropal = QtWidgets.QPushButton("Load NeuroPAL Prompts")
            neuropal_layout.addWidget(btn_load_neuropal)
            point_layout.addWidget(neuropal_row)

            # Copy to all timesteps button (separate row)
            copy_row = QtWidgets.QWidget()
            copy_layout = QtWidgets.QHBoxLayout()
            copy_row.setLayout(copy_layout)
            btn_copy_all_t = QtWidgets.QPushButton("Copy Prompts to All Timesteps")
            btn_copy_all_t.setToolTip(
                "Copy current timestep's point prompts to all timesteps\n"
                "Preserves IDs and coordinates"
            )
            copy_layout.addWidget(btn_copy_all_t)
            point_layout.addWidget(copy_row)

            def _save_point_prompts():
                try:
                    # Select directory to save point prompts
                    directory = QFileDialog.getExistingDirectory(
                        None, 
                        "Select Directory to Save Point Prompts",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    save_path = Path(directory)
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save point prompts for each timestep that has them
                    saved_count = 0
                    for t in range(self.n_timesteps):
                        points = self.point_prompts_4d.get(t, np.empty((0, 3)))
                        if points is not None and len(points) > 0:
                            # Save points coordinates
                            points_file = save_path / f"point_prompts_{t}.npy"
                            np.save(points_file, points)
                            
                            # Save point IDs for this timestep
                            ids_for_timestep = []
                            for point in points:
                                z, y, x = int(point[0]), int(point[1]), int(point[2])
                                key = (t, z, y, x)
                                point_id = self.point_id_map.get(key, 1)
                                ids_for_timestep.append(point_id)
                            
                            ids_file = save_path / f"point_ids_{t}.npy"
                            np.save(ids_file, np.array(ids_for_timestep))
                            saved_count += 1
                    
                    if saved_count > 0:
                        show_info(f"Saved point prompts for {saved_count} timestep(s) to {directory}")
                    else:
                        show_info("No point prompts to save")
                    
                except Exception as e:
                    print(f"Failed to save point prompts: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"Failed to save point prompts: {e}")

            def _load_point_prompts():
                try:
                    # Select directory containing point prompts
                    directory = QFileDialog.getExistingDirectory(
                        None,
                        "Select Directory Containing Point Prompts",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    directory_path = Path(directory)
                    
                    # Initialize storage if needed
                    if not hasattr(self, 'point_prompts_4d') or self.point_prompts_4d is None:
                        self.point_prompts_4d = {}
                    if not hasattr(self, 'point_id_map') or self.point_id_map is None:
                        self.point_id_map = {}
                    
                    # Clear existing point prompts
                    self.point_prompts_4d.clear()
                    self.point_id_map.clear()
                    
                    # Load point prompts for each timestep
                    loaded_count = 0
                    for t in range(self.n_timesteps):
                        points_file = directory_path / f"point_prompts_{t}.npy"
                        ids_file = directory_path / f"point_ids_{t}.npy"
                        
                        if points_file.exists():
                            # Load points
                            points = np.load(points_file)
                            self.point_prompts_4d[t] = points
                            
                            # Load IDs if available
                            if ids_file.exists():
                                ids = np.load(ids_file)
                                # Restore point_id_map
                                for point, point_id in zip(points, ids):
                                    z, y, x = int(point[0]), int(point[1]), int(point[2])
                                    key = (t, z, y, x)
                                    self.point_id_map[key] = int(point_id)
                            else:
                                # Default to ID 1 if no IDs file
                                for point in points:
                                    z, y, x = int(point[0]), int(point[1]), int(point[2])
                                    key = (t, z, y, x)
                                    self.point_id_map[key] = 1
                            
                            loaded_count += 1
                    
                    # Update current visible layer
                    current_t = getattr(self, 'current_timestep', 0)
                    if "point_prompts" in self._viewer.layers:
                        layer = self._viewer.layers["point_prompts"]
                        points = self.point_prompts_4d.get(current_t, np.empty((0, 3)))
                        layer.data = points
                    
                    # Refresh point manager widget if it exists
                    if hasattr(self, '_point_manager_widget'):
                        try:
                            self._point_manager_widget.refresh_point_list()
                        except Exception:
                            pass
                    
                    if loaded_count > 0:
                        show_info(f"Loaded point prompts for {loaded_count} timestep(s) from {directory}")
                    else:
                        show_info("No point prompts found in directory")
                    
                except Exception as e:
                    print(f"Failed to load point prompts: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"Failed to load point prompts: {e}")

            btn_save_points.clicked.connect(_save_point_prompts)
            btn_load_points.clicked.connect(_load_point_prompts)

            # Auto-place prompts button (separate row)
            auto_row = QtWidgets.QWidget()
            auto_layout = QtWidgets.QHBoxLayout()
            auto_row.setLayout(auto_layout)
            btn_auto_prompts = QtWidgets.QPushButton("Auto-Place Prompts")
            btn_auto_prompts.setToolTip(
                "Automatically place prompts at bright spots (intensity-based)\n"
                "Works on raw image, no GPU needed, runs in <1 second"
            )
            auto_layout.addWidget(btn_auto_prompts)
            point_layout.addWidget(auto_row)

            def _auto_place_prompts():
                """Automatically place point prompts at bright spots based on intensity."""
                try:
                    from scipy.ndimage import center_of_mass, label
                    
                    # Get current timestep image
                    t = self.current_timestep
                    if self.image_4d is None:
                        show_info("⚠️ No image loaded. Load data first.")
                        return
                    
                    image_3d = self.image_4d[t]  # (Z, Y, X)
                    
                    print(f"\n{'='*60}")
                    print(f"🔍 Auto-detecting neurons at timestep {t}")
                    print(f"{'='*60}")
                    print(f"  Image shape: {image_3d.shape}")
                    print(f"  Intensity range: {image_3d.min():.0f} to {image_3d.max():.0f}")
                    
                    # Fixed percentile threshold
                    percentile = 98
                    
                    # Find threshold
                    threshold = np.percentile(image_3d, percentile)
                    print(f"  Threshold ({percentile}th percentile): {threshold:.0f}")
                    
                    # Create binary mask
                    bright_mask = image_3d > threshold
                    
                    # Watershed separation to split touching neurons
                    print("  Applying watershed separation for touching neurons...")
                    
                    distance = distance_transform_edt(bright_mask)
                    
              # Find local maxima as seeds
                    local_max = maximum_filter(distance, size=5) == distance
                    local_max[distance == 0] = False# 5% above threshold
                    
                    # Label seed regions
                    markers, _ = label(local_max)
                    
                    # Watershed to separate touching regions
                    labeled = watershed(-distance, markers, mask=bright_mask)
                    num_features = labeled.max()
                    
                    print(f"  Found {num_features} regions after watershed separation")
                    
                    # Filter and get centroids
                    prompts = []
                    skipped_small = 0
                    skipped_large = 0
                    
                    for region_id in range(1, num_features + 1):
                        region_mask = labeled == region_id
                        size = np.sum(region_mask)
                        
                        # Skip tiny regions (noise) - reduced from 10 to 5
                        if size < 5:  # pixels
                            skipped_small += 1
                            continue
                        
                        # Skip huge regions (background artifacts)
                        if size > 10000:
                            print(f"    Skipping region {region_id}: too large ({size} px)")
                            skipped_large += 1
                            continue
                        
                        # Get centroid
                        z, y, x = center_of_mass(region_mask)
                        prompts.append([z, y, x])
                        print(f"    ✓ Region {region_id}: size={size} px, center=({z:.1f}, {y:.1f}, {x:.1f})")
                    
                    if skipped_small > 0:
                        print(f"  (Skipped {skipped_small} tiny regions < 5 pixels)")
                    if skipped_large > 0:
                        print(f"  (Skipped {skipped_large} large regions > 10000 pixels)")
                    
                    prompts = np.array(prompts)
                    
                    if len(prompts) == 0:
                        show_info(
                            f"⚠️ No bright spots found with threshold={threshold:.0f}\n\n"
                            f"Try:\n"
                            f"• Click button again and use LOWER percentile (e.g., 98.0)\n"
                            f"• Check if neurons are visible in viewer\n"
                            f"• Adjust image contrast\n\n"
                            f"Current settings:\n"
                            f"  Percentile: {percentile}\n"
                            f"  Threshold: {threshold:.0f}\n"
                            f"  Regions found: {num_features}\n"
                            f"  After filtering: 0"
                        )
                        print("  ⚠️ No valid regions found after filtering")
                        return
                    
                    print(f"\n  📍 Placing {len(prompts)} prompts...")
                    
                    # Add to point_prompts layer
                    if "point_prompts" in self._viewer.layers:
                        layer = self._viewer.layers["point_prompts"]
                        existing = np.array(layer.data) if len(layer.data) > 0 else np.empty((0, 3))
                        # Append to existing prompts
                        if len(existing) > 0:
                            prompts = np.vstack([existing, prompts])
                            print(f"  Added to {len(existing)} existing prompts")
                        layer.data = prompts
                        layer.visible = True
                    else:
                        layer = self._viewer.add_points(
                            prompts,
                            name="point_prompts",
                            size=10,
                            face_color='orange',
                            edge_color='white',
                            edge_width=2,
                            ndim=3
                        )
                    
                    # Store in 4D map
                    self.point_prompts_4d[t] = prompts
                    
                    # Assign IDs (sequential starting from current max ID)
                    existing_ids = list(self.point_id_map.values()) if hasattr(self, 'point_id_map') and self.point_id_map else [0]
                    start_id = max(existing_ids) + 1 if existing_ids else 1
                    
                    for i, point in enumerate(prompts):
                        z, y, x = int(point[0]), int(point[1]), int(point[2])
                        point_id = start_id + i
                        self.point_id_map[(t, z, y, x)] = point_id
                    
                    # Update colors
                    point_ids = self._get_point_ids_for_timestep(t, prompts)
                    self._update_point_colors(layer, point_ids)
                    
                    # Refresh point manager widget if it exists
                    if hasattr(self, '_point_manager_widget'):
                        try:
                            self._point_manager_widget.refresh_point_list()
                        except Exception:
                            pass
                    
                    print(f"{'='*60}")
                    print(f"✅ Auto-placed {len(prompts)} prompts successfully!")
                    print(f"{'='*60}\n")
                    
                    show_info(
                        f"✓ Auto-placed {len(prompts)} prompts\n\n"
                        f"Settings used:\n"
                        f"  Percentile: {percentile}\n"
                        f"  Threshold: {threshold:.0f}\n"
                        f"  Regions found: {num_features}\n"
                        f"  Valid prompts: {len(prompts)}\n\n"
                        f"Too few? Click again and use LOWER percentile (e.g., 98.0)\n"
                        f"Too many? Click again and use HIGHER percentile (e.g., 99.5)\n\n"
                        f"Next: Click 'Segment Volume' to segment all neurons"
                    )
                    
                except ImportError as e:
                    show_info(f"⚠️ Missing dependency: {e}\n\nInstall scipy: pip install scipy")
                    print(f"Failed to auto-place prompts: {e}")
                except Exception as e:
                    print(f"Failed to auto-place prompts: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"❌ Error: {str(e)}")

            btn_auto_prompts.clicked.connect(_auto_place_prompts)

            def _copy_prompts_to_all_timesteps():
                """Copy current timestep's point prompts to all timesteps, preserving IDs."""
                try:
                    t_current = self.current_timestep
                    
                    # Get current prompts
                    if "point_prompts" not in self._viewer.layers:
                        show_info("⚠️ No point prompts in current timestep")
                        return
                    
                    layer = self._viewer.layers["point_prompts"]
                    current_prompts = np.array(layer.data)
                    
                    if len(current_prompts) == 0:
                        show_info("⚠️ No point prompts to copy")
                        return
                    
                    # Initialize storage if needed
                    if not hasattr(self, 'point_prompts_4d') or self.point_prompts_4d is None:
                        self.point_prompts_4d = {}
                    if not hasattr(self, 'point_id_map') or self.point_id_map is None:
                        self.point_id_map = {}
                    
                    # Get IDs from current timestep
                    current_ids = []
                    for point in current_prompts:
                        z, y, x = int(point[0]), int(point[1]), int(point[2])
                        key = (t_current, z, y, x)
                        point_id = self.point_id_map.get(key, 1)
                        current_ids.append(point_id)
                    
                    # Copy prompts and IDs to all timesteps
                    copied_count = 0
                    for t in range(self.n_timesteps):
                        if t == t_current:
                            continue  # Skip current timestep
                        
                        # Store prompts
                        self.point_prompts_4d[t] = current_prompts.copy()
                        
                        # Store IDs
                        for point, point_id in zip(current_prompts, current_ids):
                            z, y, x = int(point[0]), int(point[1]), int(point[2])
                            key = (t, z, y, x)
                            self.point_id_map[key] = point_id
                        
                        copied_count += 1
                    
                    # Refresh point manager widget if it exists
                    if hasattr(self, '_point_manager_widget'):
                        try:
                            self._point_manager_widget.refresh_point_list()
                        except Exception:
                            pass
                    
                    print(f"✅ Copied {len(current_prompts)} prompts to {copied_count} timesteps")
                    show_info(
                        f"✓ Copied {len(current_prompts)} prompts from timestep {t_current}\n"
                        f"  to {copied_count} other timesteps\n\n"
                        f"IDs preserved for temporal tracking.\n\n"
                        f"Next: Click 'Segment Volume' to segment all timesteps"
                    )
                    
                except Exception as e:
                    print(f"Failed to copy prompts: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"❌ Error: {str(e)}")
            
            btn_copy_all_t.clicked.connect(_copy_prompts_to_all_timesteps)

            # ========== CLEAN SEGMENTATIONS ==========
            clean_row = QtWidgets.QWidget()
            clean_layout = QtWidgets.QVBoxLayout()
            clean_row.setLayout(clean_layout)

            clean_label = QtWidgets.QLabel("<b>Clean Segmentations (Otsu)</b>")
            clean_layout.addWidget(clean_label)

            # Strictness slider
            strictness_row = QtWidgets.QWidget()
            strictness_layout = QtWidgets.QHBoxLayout()
            strictness_row.setLayout(strictness_layout)
            
            strictness_layout.addWidget(QtWidgets.QLabel("Strictness:"))
            self._clean_strictness_slider = QtWidgets.QSlider(Qt.Horizontal)
            self._clean_strictness_slider.setMinimum(50)
            self._clean_strictness_slider.setMaximum(100)
            self._clean_strictness_slider.setValue(60)  # Default 0.6 (60%)
            self._clean_strictness_slider.setToolTip(
                "Lower = more lenient (keeps more pixels)\n"
                "Higher = stricter (removes more pixels)\n"
                "Default: 60% (recommended for neurons)"
            )
            strictness_layout.addWidget(self._clean_strictness_slider)
            
            self._strictness_value_label = QtWidgets.QLabel("60%")
            strictness_layout.addWidget(self._strictness_value_label)
            
            def update_strictness_label(value):
                self._strictness_value_label.setText(f"{value}%")
            
            self._clean_strictness_slider.valueChanged.connect(update_strictness_label)
            clean_layout.addWidget(strictness_row)

            btn_clean = QtWidgets.QPushButton("Clean Segmentations")
            btn_clean.setToolTip("Remove dim pixels within each object")
            clean_layout.addWidget(btn_clean)

            def _clean_segmentations():
                """Remove dim voxels within each object using selected method."""
                try:
                    if self.current_object_4d is None or self.image_4d is None:
                        show_info("⚠️ No segmentations to clean")
                        return
                    
                    strictness = self._clean_strictness_slider.value() / 100.0  # Convert to 0.5-1.0
                    
                    print(f"\n{'='*60}")
                    print(f"🧹 Cleaning segmentations using Otsu (strictness: {strictness:.2f})")
                    print(f"{'='*60}")
                    
                    total_voxels_removed = 0
                    objects_processed = 0
                    
                    # Process each timestep
                    for t in range(self.n_timesteps):
                        seg = self.current_object_4d[t]
                        img = self.image_4d[t]
                        unique_ids = np.unique(seg)
                        unique_ids = unique_ids[unique_ids > 0]  # Exclude background
                        
                        if len(unique_ids) == 0:
                            continue
                        
                        # Process each object
                        for obj_id in unique_ids:
                            mask = seg == obj_id
                            intensities = img[mask]
                            
                            if len(intensities) < 10:  # Skip tiny objects
                                continue
                            
                            # Calculate threshold using Otsu
                            try:
                                from skimage.filters import threshold_otsu
                                threshold = threshold_otsu(intensities)
                                # Apply strictness factor (lower = more lenient)
                                threshold = threshold * strictness
                            except Exception:
                                # Fallback to mean if Otsu fails
                                threshold = np.mean(intensities) * strictness
                            
                            # Remove voxels below threshold
                            dim_voxels = mask & (img <= threshold)
                            num_removed = np.sum(dim_voxels)
                            
                            if num_removed > 0:
                                seg[dim_voxels] = 0  # Set dim voxels to background
                                total_voxels_removed += num_removed
                                print(f"  t={t}, ID={obj_id}: removed {num_removed} dim voxels (threshold={threshold:.2f})")
                            
                            objects_processed += 1
                    
                    if total_voxels_removed == 0:
                        show_info("✓ No dim voxels found to remove")
                        return
                    
                    # Refresh layer
                    if "current_object_4d" in self._viewer.layers:
                        self._viewer.layers["current_object_4d"].refresh()
                    
                    print(f"{'='*60}")
                    print(f"✅ Cleaned {objects_processed} objects")
                    print(f"   Removed {total_voxels_removed} total dim voxels")
                    print(f"{'='*60}\n")
                    
                    show_info(
                        f"✓ Cleaned segmentations\n\n"
                        f"Method: Otsu (strictness: {int(strictness*100)}%)\n"
                        f"Objects processed: {objects_processed}\n"
                        f"Voxels removed: {total_voxels_removed}\n\n"
                        f"Dim pixels (likely background/edges) removed from each object"
                    )
                    
                except Exception as e:
                    print(f"Failed to clean segmentations: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"❌ Error: {str(e)}")

            btn_clean.clicked.connect(_clean_segmentations)
            point_layout.addWidget(clean_row)

            # ========== PROPAGATE SEGMENTATION ==========
            propagate_row = QtWidgets.QWidget()
            propagate_layout = QtWidgets.QVBoxLayout()
            propagate_row.setLayout(propagate_layout)

            propagate_label = QtWidgets.QLabel("<b>Propagate Segmentation</b>")
            propagate_layout.addWidget(propagate_label)

            btn_propagate = QtWidgets.QPushButton("Propagate Current Segmentation (Shift+P)")
            btn_propagate.setToolTip("Track objects across timesteps using centroid matching")
            propagate_layout.addWidget(btn_propagate)

            def _propagate_segmentation():
                """Propagate committed segmentation sequentially to adjacent timesteps.
                
                Uses temporal continuity: copies segmentation from t to t±1, then adjusts
                each object's position to match local intensity maxima in the new timestep.
                This mirrors the successful point propagation strategy.
                """
                try:
                    if self.segmentation_4d is None or self.image_4d is None:
                        show_info("⚠️ No committed segmentation to propagate")
                        return
                    
                    from scipy.ndimage import center_of_mass, shift, maximum_filter
                    from scipy.ndimage import label, binary_dilation
                    
                    current_t = int(self.current_timestep)
                    seg_current = self.segmentation_4d[current_t]
                    
                    # Get object IDs from current timestep
                    current_ids = np.unique(seg_current)
                    current_ids = current_ids[current_ids > 0]
                    
                    if len(current_ids) == 0:
                        show_info("⚠️ No objects in current segmentation")
                        return
                    
                    print(f"\n{'='*60}")
                    print(f"🔄 Propagating {len(current_ids)} objects from timestep {current_t}")
                    print(f"{'='*60}")
                    
                    # Propagation parameters - relaxed for better tracking
                    MIN_IOU_THRESHOLD = 0.25  # Minimum IoU to consider a match (lowered for dim neurons)
                    MAX_CENTROID_DISTANCE = 50  # Max pixels a neuron can move per timestep (increased)
                    MAX_SIZE_RATIO = 4.0  # Max size change ratio (increased for brightness changes)
                    BLOB_PERCENTILE = 20  # Lower threshold for blob detection (catches dim neurons)
                    
                    total_propagated = 0
                    
                    # Helper function: Watershed split merged blobs
                    def watershed_split_blob(blob_mask, img, seed_centroids):
                        """Split a merged blob using watershed with predicted centroids as seeds."""
                        if len(seed_centroids) < 2:
                            return blob_mask.astype(np.int32)
                        
                        # Create markers from seed centroids
                        markers = np.zeros_like(blob_mask, dtype=np.int32)
                        marker_id = 1
                        
                        for z, y, x in seed_centroids:
                            z, y, x = int(z), int(y), int(x)
                            if (0 <= z < blob_mask.shape[0] and 
                                0 <= y < blob_mask.shape[1] and 
                                0 <= x < blob_mask.shape[2] and
                                blob_mask[z, y, x]):
                                markers[z, y, x] = marker_id
                                marker_id += 1
                        
                        # Check if we have enough valid markers
                        if marker_id < 3:  # Need at least 2 markers
                            return blob_mask.astype(np.int32)
                        
                        # Distance transform for watershed
                        distance = distance_transform_edt(blob_mask)
                        
                        # Watershed segmentation
                        separated = watershed(-distance, markers, mask=blob_mask)
                        
                        return separated
                    
                    # Forward propagation: current_t → t+1 → t+2 → ... → end
                    
                    for t in range(current_t + 1, self.n_timesteps):
                        seg_prev = self.segmentation_4d[t - 1]
                        img_curr = self.image_4d[t]
                        new_seg = np.zeros_like(seg_prev, dtype=np.uint32)
                        
                        # Get object IDs from previous timestep
                        prev_ids = np.unique(seg_prev)
                        prev_ids = prev_ids[prev_ids > 0]
                        
                        if len(prev_ids) == 0:
                            continue
                        
                        # STEP 1: Detect blobs with LOWER threshold (better sensitivity)
                        threshold = np.percentile(img_curr[img_curr > 0], BLOB_PERCENTILE) if np.any(img_curr > 0) else 0
                        foreground_mask = img_curr > threshold
                        labeled_blobs, num_blobs = label(foreground_mask)
                        
                        if num_blobs == 0:
                            # No blobs detected - copy all masks from previous timestep
                            for obj_id in prev_ids:
                                mask_prev = (seg_prev == obj_id)
                                new_seg[mask_prev] = obj_id
                            self.segmentation_4d[t] = new_seg
                            continue
                        
                        # STEP 2: Detect conflicts (multiple neurons claiming same blob)
                        blob_claims = {}  # blob_id -> [list of obj_ids]
                        neuron_centroids = {}  # obj_id -> (z, y, x)
                        
                        for obj_id in prev_ids:
                            mask_prev = (seg_prev == obj_id)
                            if not mask_prev.any():
                                continue
                            
                            centroid = center_of_mass(mask_prev)
                            neuron_centroids[obj_id] = centroid
                            
                            # Check which blobs overlap with this neuron
                            for blob_id in range(1, num_blobs + 1):
                                blob_mask = (labeled_blobs == blob_id)
                                overlap = np.sum(mask_prev & blob_mask)
                                
                                if overlap > 0:
                                    if blob_id not in blob_claims:
                                        blob_claims[blob_id] = []
                                    blob_claims[blob_id].append(obj_id)
                        
                        # STEP 3: Split conflicted blobs using watershed
                        split_blobs = labeled_blobs.copy()
                        next_blob_id = num_blobs + 1
                        
                        for blob_id, claiming_neurons in blob_claims.items():
                            if len(claiming_neurons) > 1:
                                # CONFLICT! Multiple neurons want this blob - split it
                                blob_mask = (labeled_blobs == blob_id)
                                seed_centroids = [neuron_centroids[oid] for oid in claiming_neurons]
                                
                                # Watershed split
                                separated = watershed_split_blob(blob_mask, img_curr, seed_centroids)
                                
                                # Replace merged blob with separated regions
                                unique_regions = np.unique(separated[separated > 0])
                                for region_id in unique_regions:
                                    region_mask = (separated == region_id)
                                    if region_mask.any():
                                        split_blobs[region_mask] = next_blob_id
                                        next_blob_id += 1
                                
                                # Remove original merged blob
                                split_blobs[blob_mask & (split_blobs == blob_id)] = 0
                        
                        # STEP 4: Hungarian algorithm on (potentially split) blobs
                        num_final_blobs = int(split_blobs.max())
                        cost_matrix = np.ones((len(prev_ids), num_final_blobs)) * 1e6
                        
                        for i, obj_id in enumerate(prev_ids):
                            mask_prev = (seg_prev == obj_id)
                            if not mask_prev.any():
                                continue
                            
                            centroid_prev = neuron_centroids.get(obj_id, center_of_mass(mask_prev))
                            size_prev = np.count_nonzero(mask_prev)
                            
                            for blob_id in range(1, num_final_blobs + 1):
                                blob_mask = (split_blobs == blob_id)
                                if not blob_mask.any():
                                    continue
                                
                                centroid_blob = center_of_mass(blob_mask)
                                size_blob = np.count_nonzero(blob_mask)
                                
                                # Spatial constraint
                                distance = np.sqrt(np.sum((np.array(centroid_prev) - np.array(centroid_blob))**2))
                                if distance > MAX_CENTROID_DISTANCE:
                                    continue
                                
                                # Size constraint
                                size_ratio = size_blob / size_prev if size_prev > 0 else 1.0
                                if size_ratio > MAX_SIZE_RATIO or size_ratio < (1.0 / MAX_SIZE_RATIO):
                                    continue
                                
                                # Calculate IoU
                                intersection = np.sum(mask_prev & blob_mask)
                                union = np.sum(mask_prev | blob_mask)
                                
                                if union > 0:
                                    iou = intersection / union
                                    cost_matrix[i, blob_id - 1] = -iou
                        
                        # STEP 5: Hungarian assignment
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        
                        # Apply assignments
                        objects_propagated = 0
                        assigned_neurons = set()
                        
                        for i, j in zip(row_ind, col_ind):
                            obj_id = prev_ids[i]
                            iou = -cost_matrix[i, j]
                            
                            if iou > 0:  # Any positive IoU is valid
                                blob_id = j + 1
                                new_mask = (split_blobs == blob_id)
                                new_seg[new_mask & (new_seg == 0)] = obj_id
                                objects_propagated += 1
                                total_propagated += 1
                                assigned_neurons.add(obj_id)
                        
                        # Handle neurons that weren't assigned (copy from previous timestep)
                        for obj_id in prev_ids:
                            if obj_id not in assigned_neurons:
                                mask_prev = (seg_prev == obj_id)
                                new_seg[mask_prev] = obj_id
                                objects_propagated += 1
                                total_propagated += 1
                        
                        # Always write the new segmentation back
                        self.segmentation_4d[t] = new_seg
                    
                    # Backward propagation: current_t → t-1 → t-2 → ... → 0
                    
                    for t in range(current_t - 1, -1, -1):
                        seg_prev = self.segmentation_4d[t + 1]
                        img_curr = self.image_4d[t]
                        new_seg = np.zeros_like(seg_prev, dtype=np.uint32)
                        
                        # Get object IDs from previous timestep (t+1 when going backward)
                        prev_ids = np.unique(seg_prev)
                        prev_ids = prev_ids[prev_ids > 0]
                        
                        if len(prev_ids) == 0:
                            continue
                        
                        # STEP 1: Detect blobs with LOWER threshold
                        threshold = np.percentile(img_curr[img_curr > 0], BLOB_PERCENTILE) if np.any(img_curr > 0) else 0
                        foreground_mask = img_curr > threshold
                        labeled_blobs, num_blobs = label(foreground_mask)
                        
                        if num_blobs == 0:
                            # No blobs detected - copy all masks from next timestep
                            for obj_id in prev_ids:
                                mask_prev = (seg_prev == obj_id)
                                new_seg[mask_prev] = obj_id
                            self.segmentation_4d[t] = new_seg
                            continue
                        
                        # STEP 2: Detect conflicts
                        blob_claims = {}
                        neuron_centroids = {}
                        
                        for obj_id in prev_ids:
                            mask_prev = (seg_prev == obj_id)
                            if not mask_prev.any():
                                continue
                            
                            centroid = center_of_mass(mask_prev)
                            neuron_centroids[obj_id] = centroid
                            
                            for blob_id in range(1, num_blobs + 1):
                                blob_mask = (labeled_blobs == blob_id)
                                overlap = np.sum(mask_prev & blob_mask)
                                
                                if overlap > 0:
                                    if blob_id not in blob_claims:
                                        blob_claims[blob_id] = []
                                    blob_claims[blob_id].append(obj_id)
                        
                        # STEP 3: Split conflicted blobs using watershed
                        split_blobs = labeled_blobs.copy()
                        next_blob_id = num_blobs + 1
                        
                        for blob_id, claiming_neurons in blob_claims.items():
                            if len(claiming_neurons) > 1:
                                blob_mask = (labeled_blobs == blob_id)
                                seed_centroids = [neuron_centroids[oid] for oid in claiming_neurons]
                                
                                separated = watershed_split_blob(blob_mask, img_curr, seed_centroids)
                                
                                unique_regions = np.unique(separated[separated > 0])
                                for region_id in unique_regions:
                                    region_mask = (separated == region_id)
                                    if region_mask.any():
                                        split_blobs[region_mask] = next_blob_id
                                        next_blob_id += 1
                                
                                split_blobs[blob_mask & (split_blobs == blob_id)] = 0
                        
                        # STEP 4: Hungarian algorithm on split blobs
                        num_final_blobs = int(split_blobs.max())
                        cost_matrix = np.ones((len(prev_ids), num_final_blobs)) * 1e6
                        
                        for i, obj_id in enumerate(prev_ids):
                            mask_prev = (seg_prev == obj_id)
                            if not mask_prev.any():
                                continue
                            
                            centroid_prev = neuron_centroids.get(obj_id, center_of_mass(mask_prev))
                            size_prev = np.count_nonzero(mask_prev)
                            
                            for blob_id in range(1, num_final_blobs + 1):
                                blob_mask = (split_blobs == blob_id)
                                if not blob_mask.any():
                                    continue
                                
                                centroid_blob = center_of_mass(blob_mask)
                                size_blob = np.count_nonzero(blob_mask)
                                
                                # Spatial constraint
                                distance = np.sqrt(np.sum((np.array(centroid_prev) - np.array(centroid_blob))**2))
                                if distance > MAX_CENTROID_DISTANCE:
                                    continue
                                
                                # Size constraint
                                size_ratio = size_blob / size_prev if size_prev > 0 else 1.0
                                if size_ratio > MAX_SIZE_RATIO or size_ratio < (1.0 / MAX_SIZE_RATIO):
                                    continue
                                
                                # Calculate IoU
                                intersection = np.sum(mask_prev & blob_mask)
                                union = np.sum(mask_prev | blob_mask)
                                
                                if union > 0:
                                    iou = intersection / union
                                    cost_matrix[i, blob_id - 1] = -iou
                        
                        # STEP 5: Hungarian assignment
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        
                        objects_propagated = 0
                        assigned_neurons = set()
                        
                        for i, j in zip(row_ind, col_ind):
                            obj_id = prev_ids[i]
                            iou = -cost_matrix[i, j]
                            
                            if iou > 0:
                                blob_id = j + 1
                                new_mask = (split_blobs == blob_id)
                                new_seg[new_mask & (new_seg == 0)] = obj_id
                                objects_propagated += 1
                                total_propagated += 1
                                assigned_neurons.add(obj_id)
                        
                        # Handle neurons that weren't assigned
                        for obj_id in prev_ids:
                            if obj_id not in assigned_neurons:
                                mask_prev = (seg_prev == obj_id)
                                new_seg[mask_prev] = obj_id
                                objects_propagated += 1
                                total_propagated += 1
                        
                        # Always write the new segmentation back
                        self.segmentation_4d[t] = new_seg
                    
                    # Refresh all relevant viewer layers
                    if "committed_objects_4d" in self._viewer.layers:
                        self._viewer.layers["committed_objects_4d"].refresh()
                    if "current_object_4d" in self._viewer.layers:
                        self._viewer.layers["current_object_4d"].refresh()
                    
                    print(f"\n{'='*60}")
                    print(f"✅ Propagated segmentation across {self.n_timesteps} timesteps")
                    print(f"   Total propagations: {total_propagated}")
                    print(f"{'='*60}\n")
                    
                    show_info(
                        f"✓ Propagated segmentation\n\n"
                        f"Source timestep: {current_t}\n"
                        f"Objects tracked: {len(current_ids)}\n"
                        f"Method: Hungarian + Watershed splitting\n"
                        f"IoU threshold: {MIN_IOU_THRESHOLD}, Distance: {MAX_CENTROID_DISTANCE}px"
                    )
                    
                except Exception as e:
                    print(f"Failed to propagate segmentation: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"❌ Error: {str(e)}")

            btn_propagate.clicked.connect(_propagate_segmentation)
            
            # Store reference for keybind access
            self._propagate_segmentation_func = _propagate_segmentation
            
            point_layout.addWidget(propagate_row)

            # ========== LOAD NEUROPAL PROMPTS ==========
            def _load_neuropal_prompts():
                """Load NeuroPAL-derived point prompts with neuron names."""
                try:
                    from pathlib import Path
                    import sys
                    
                    # Add NeuroPAL matching folder to path
                    neuropal_path = Path(__file__).parent.parent / "Neuropal Coordinate Matching"
                    if str(neuropal_path) not in sys.path:
                        sys.path.insert(0, str(neuropal_path))
                    
                    from load_neuropal_prompts import load_neuropal_prompts, create_neuron_name_layer_data  # type: ignore
                    
                    # Select directory containing NeuroPAL prompts
                    directory = QFileDialog.getExistingDirectory(
                        None,
                        "Select Directory Containing NeuroPAL Prompts",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    # Load prompts for timestep 0
                    result = load_neuropal_prompts(directory, timestep=0)
                    
                    if not result['success']:
                        show_info(f"Failed to load NeuroPAL prompts:\n{result['error']}")
                        return
                    
                    # Initialize storage if needed
                    if not hasattr(self, 'point_prompts_4d') or self.point_prompts_4d is None:
                        self.point_prompts_4d = {}
                    if not hasattr(self, 'point_id_map') or self.point_id_map is None:
                        self.point_id_map = {}
                    if not hasattr(self, 'neuron_names_map') or self.neuron_names_map is None:
                        self.neuron_names_map = {}
                    
                    # Store prompts and IDs for timestep 0
                    t = result['timestep']
                    prompts = result['prompts']
                    ids = result['ids']
                    names = result['names']
                    
                    self.point_prompts_4d[t] = prompts
                    self.neuron_names_map = names  # Store neuron names globally
                    
                    # Update point_id_map with neuron IDs
                    for point, point_id in zip(prompts, ids):
                        z, y, x = int(point[0]), int(point[1]), int(point[2])
                        key = (t, z, y, x)
                        self.point_id_map[key] = int(point_id)
                    
                    # Switch to timestep 0 to show the loaded prompts
                    try:
                        self._viewer.dims.current_step = (0,) + self._viewer.dims.current_step[1:]
                        self.current_timestep = 0
                    except Exception:
                        pass
                    
                    # Update or create point prompts layer
                    if "point_prompts" in self._viewer.layers:
                        layer = self._viewer.layers["point_prompts"]
                        layer.data = prompts
                        layer.visible = True
                        # Update colors based on IDs
                        point_ids_list = [int(pid) for pid in ids]
                        self._update_point_colors(layer, point_ids_list)
                    else:
                        # Create point prompts layer
                        layer = self._viewer.add_points(
                            prompts,
                            name="point_prompts",
                            size=10,
                            face_color='red',
                            edge_color='white',
                            edge_width=2,
                            ndim=3
                        )
                        # Update colors based on IDs
                        point_ids_list = [int(pid) for pid in ids]
                        self._update_point_colors(layer, point_ids_list)
                    
                    # Create or update neuron names layer
                    layer_data = create_neuron_name_layer_data(prompts, ids, names)
                    
                    if "Neuron_Names" in self._viewer.layers:
                        # Update existing layer
                        name_layer = self._viewer.layers["Neuron_Names"]
                        name_layer.data = layer_data['coordinates']
                        name_layer.text = layer_data['text']
                        name_layer.properties = layer_data['properties']
                        name_layer.visible = True
                    else:
                        # Create new layer with proper visibility
                        name_layer = self._viewer.add_points(
                            layer_data['coordinates'],
                            name="Neuron_Names",
                            size=8,  # Small but visible points
                            face_color='yellow',
                            edge_color='yellow',
                            edge_width=1,
                            text=layer_data['text'],
                            properties=layer_data['properties'],
                            ndim=3
                        )
                        name_layer.visible = True
                    
                    # Refresh point manager widget if it exists
                    if hasattr(self, '_point_manager_widget'):
                        try:
                            self._point_manager_widget.refresh_point_list()
                        except Exception:
                            pass
                    
                    # Force viewer refresh
                    try:
                        self._viewer.layers.selection.active = self._viewer.layers["point_prompts"]
                        self._viewer.reset_view()
                    except Exception:
                        pass
                    
                    show_info(
                        f"✓ Loaded {result['num_prompts']} NeuroPAL prompts\n"
                        f"Neurons: {', '.join([names[int(pid)] for pid in ids])}\n"
                        f"Switched to timestep 0"
                    )
                    
                except ImportError as e:
                    show_info(
                        f"Failed to import NeuroPAL loader:\n{str(e)}\n\n"
                        "Make sure load_neuropal_prompts.py exists in:\n"
                        "micro_sam/Neuropal Coordinate Matching/"
                    )
                    import traceback
                    traceback.print_exc()
                except Exception as e:
                    print(f"Failed to load NeuroPAL prompts: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"Failed to load NeuroPAL prompts:\n{str(e)}")

            btn_load_neuropal.clicked.connect(_load_neuropal_prompts)

            # Insert the point prompt widget after embeddings widget
            try:
                self._annotator_widget.layout().insertWidget(1, point_widget)
            except Exception as ex:
                # fallback: add at end
                print(f"Warning: insertWidget failed ({ex}), adding at end")
                self._annotator_widget.layout().addWidget(point_widget)

        except Exception as e:
            print(f"Failed to create point prompt tools: {e}")
            import traceback
            traceback.print_exc()

        # ================== SAVE SEGMENTATIONS ==================
        try:
            save_seg_widget = QtWidgets.QGroupBox("SAVE SEGMENTATIONS")
            save_seg_widget.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #00ff88; 
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            save_seg_layout = QtWidgets.QVBoxLayout()
            save_seg_widget.setLayout(save_seg_layout)

            # Save segmentation button
            btn_save_seg = QtWidgets.QPushButton("💾 Save Segmentation")
            btn_save_seg.setToolTip("Save raw ima ge and committed segmentation masks to NPZ file")
            
            def _save_segmentation():
                try:
                    from qtpy.QtWidgets import QFileDialog
                    
                    # Ask user for save location
                    filepath, _ = QFileDialog.getSaveFileName(
                        self._viewer.window._qt_window,
                        "Save Segmentation",
                        str(Path.home() / "my_segmentation.npz"),
                        "NumPy Archive (*.npz)"
                    )
                    
                    if not filepath:
                        return
                    
                    filepath = Path(filepath)
                    if not filepath.suffix:
                        filepath = filepath.with_suffix('.npz')
                    
                    # Check if we have data to save
                    if self.image_4d is None:
                        show_info("No image data to save")
                        return
                    
                    if self.segmentation_4d is None:
                        show_info("No segmentation to save. Commit objects first.")
                        return
                    
                    # Save data
                    print(f"Saving segmentation to {filepath}...")
                    np.savez(
                        filepath,
                        image_4d=self.image_4d,                    # Raw calcium data (T,Z,Y,X)
                        segmentation_4d=self.segmentation_4d,      # Committed segmentation (T,Z,Y,X)
                        n_timesteps=self.n_timesteps,
                        shape=self.image_4d.shape,
                    )
                    
                    # Save neuron names if available (for fluorescence extraction)
                    if hasattr(self, 'neuron_names_map') and self.neuron_names_map:
                        import json
                        json_path = filepath.with_name(filepath.stem + '_neuron_names.json')
                        try:
                            with open(json_path, 'w') as f:
                                json.dump(self.neuron_names_map, f, indent=2)
                            print(f"   - Also saved neuron names to {json_path.name}")
                        except Exception as e:
                            print(f"   ⚠️ Could not save neuron names: {e}")
                    
                    show_info(f"✅ Saved to {filepath.name}")
                    print(f"✅ Saved segmentation successfully!")
                    print(f"   - File: {filepath}")
                    print(f"   - Image shape: {self.image_4d.shape}")
                    print(f"   - Segmentation shape: {self.segmentation_4d.shape}")
                    
                    # Show unique neuron IDs
                    unique_ids = np.unique(self.segmentation_4d)
                    unique_ids = unique_ids[unique_ids > 0]  # Exclude background
                    print(f"   - Neurons saved: {len(unique_ids)} (IDs: {unique_ids.tolist()})")
                    
                    # Show neuron names if available
                    if hasattr(self, 'neuron_names_map') and self.neuron_names_map:
                        named_neurons = [self.neuron_names_map.get(int(nid), f"ID_{nid}") for nid in unique_ids]
                        print(f"   - Neuron names: {', '.join(named_neurons)}")
                    
                except Exception as e:
                    print(f"❌ Failed to save segmentation: {e}")
                    import traceback
                    traceback.print_exc()
                    show_info(f"Failed to save: {e}")
            
            btn_save_seg.clicked.connect(_save_segmentation)
            save_seg_layout.addWidget(btn_save_seg)

            # Insert the save segmentation widget after point prompt widget
            try:
                self._annotator_widget.layout().insertWidget(2, save_seg_widget)
            except Exception:
                # fallback: add at end
                self._annotator_widget.layout().addWidget(save_seg_widget)

        except Exception as e:
            print(f"Failed to create save segmentation tools: {e}")

            # --- Remap points UI and Napari points layer ---
            try:
                # create / reuse a Napari Points layer named 'remap_points' (3D coords)
                try:
                    if "remap_points" in self._viewer.layers:
                        self._remap_points_layer = self._viewer.layers["remap_points"]
                    else:
                        self._remap_points_layer = self._viewer.add_points(np.empty((0, 3)), name="remap_points", ndim=3, face_color='red', size=5)
                except Exception:
                    self._remap_points_layer = None

                # Create a styled container for remap points
                remap_widget = QtWidgets.QGroupBox("REMAP POINTS")
                remap_widget.setStyleSheet("""
                    QGroupBox {
                        font-weight: bold;
                        border: 2px solid #555;
                        border-radius: 5px;
                        margin-top: 10px;
                        padding-top: 10px;
                        color: #00e5ff;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px 0 5px;
                    }
                """)
                remap_widget.setLayout(QtWidgets.QVBoxLayout())
                remap_widget.layout().setContentsMargins(4, 4, 4, 4)
                remap_widget.layout().setSpacing(6)

                # Add ID remapper widget at the top
                try:
                    from ._widgets import IdRemapperWidget
                    remapper = IdRemapperWidget(self)
                    remap_widget.layout().addWidget(remapper)
                except Exception:
                    pass
                
                # Scroll area with a vertical layout for entries
                try:
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    inner = QtWidgets.QWidget()
                    inner.setLayout(QtWidgets.QVBoxLayout())
                    inner.layout().setSpacing(4)
                    inner.layout().setContentsMargins(0, 0, 0, 0)
                    scroll.setWidget(inner)
                    remap_widget.layout().addWidget(scroll)
                    self._remap_entries_container = inner.layout()
                except Exception:
                    # fallback: direct vertical layout
                    self._remap_entries_container = QtWidgets.QVBoxLayout()
                    remap_widget.layout().addLayout(self._remap_entries_container)

                # Apply button and shortcut hint
                apply_btn = QtWidgets.QPushButton("Apply remaps (Shift+R)")
                remap_widget.layout().addWidget(apply_btn)
                # Clear button to remove all remap points and entries
                clear_btn = QtWidgets.QPushButton("Clear remap points")
                remap_widget.layout().addWidget(clear_btn)

                # Insert the remap widget into the annotator panel (after embeddings)
                try:
                    self._annotator_widget.layout().addWidget(remap_widget)
                except Exception:
                    try:
                        self._annotator_widget.layout().insertWidget(1, remap_widget)
                    except Exception:
                        pass

                # storage for original IDs (aligned with points order) and widget refs
                self._remap_point_original_ids = []
                self._remap_target_widgets = []

                # connect points layer -> handler so new points create UI entries
                try:
                    if self._remap_points_layer is not None:
                        # keep a small wrapper that updates originals whenever points change
                        self._remap_points_layer.events.data.connect(self._on_remap_points_changed)
                except Exception:
                    pass

                # connect apply button
                try:
                    apply_btn.clicked.connect(self.apply_remaps)
                except Exception:
                    pass

                # connect clear button
                try:
                    clear_btn.clicked.connect(self.clear_remap_points)
                except Exception:
                    pass

                # keyboard shortcut Shift+R to apply remaps
                try:
                    shortcut = QtWidgets.QShortcut(QKeySequence("Shift+R"), remap_widget)
                    shortcut.activated.connect(self.apply_remaps)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            # don't fail initialization if Qt isn't available
            pass
        
        # Set maximum width for the dock widget to make it less wide
        try:
            if hasattr(self, '_annotator_widget'):
                self._annotator_widget.setMaximumWidth(550)
        except Exception:
            pass
    
    def _create_keybindings(self):
        """Override base class keybindings with 4D-specific shortcuts.
        
        4D annotator keybindings:
        - s: Segment current object (single slice, like original micro-sam)
        - Shift+S: Segment all objects across timesteps
        - Shift+P: Segment + Propagate current segmentation across time
        - c: Commit current segmentation
        - C: Commit all timesteps
        """
        # DON'T call parent _create_keybindings since we deleted the widgets it references
        # Instead, implement keybindings directly here
        
        # 's' key: segment at current timestep
        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            try:
                self._segment_current_object()
            except Exception as e:
                print(f"Error in segment: {e}")
                import traceback
                traceback.print_exc()
        
        # 'c' key: commit current timestep
        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            try:
                self.commit_current_timestep()
            except Exception as e:
                print(f"Error in commit: {e}")
                import traceback
                traceback.print_exc()
        
        # Override Shift+S for batch segmentation
        @self._viewer.bind_key("Shift-S", overwrite=True)
        def _segment_all_timesteps(viewer):
            try:
                self.segment_all_timesteps()
            except Exception as e:
                print(f"Error in segment_all_timesteps: {e}")
                import traceback
                traceback.print_exc()
        
        # Override Shift+P for propagation
        @self._viewer.bind_key("Shift-P", overwrite=True)
        def _segment_and_propagate(viewer):
            try:
                # Call the segmentation propagation function (IOU-based mask propagation)
                if hasattr(self, '_propagate_segmentation_func'):
                    self._propagate_segmentation_func()
                else:
                    # Fallback to old behavior if function not yet initialized
                    self.segment_and_propagate_all_timesteps()
            except Exception as e:
                print(f"Error in propagate segmentation: {e}")
                import traceback
                traceback.print_exc()
        
        # Override Shift+C for commit all (keeping lowercase 'c' for single commit from parent)
        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _commit_all_timesteps(viewer):
            try:
                self.commit_all_timesteps()
            except Exception as e:
                print(f"Error in commit_all_timesteps: {e}")
                import traceback
                traceback.print_exc()

    def _segment_current_object(self):
        """Segment object at current timestep only (used by 's' keybinding)."""
        try:
            t = self.current_timestep
            # Segment only the current timestep
            if not hasattr(self, "point_prompts_4d") or t not in self.point_prompts_4d:
                print(f"No point prompts at timestep {t}")
                return
            
            pts = self.point_prompts_4d[t]
            if len(pts) == 0:
                print(f"No point prompts at timestep {t}")
                return
            
            # Use the existing segmentation logic
            from . import util as sam_util
            from .util import segment_mask_in_volume
            
            state = AnnotatorState()
            if state.predictor is None or state.image_embeddings is None:
                print("No embeddings available for segmentation")
                return
            
            shape = self.image_4d[t].shape  # (Z, Y, X)
            
            # Get point layer for labels
            point_layer = self._viewer.layers.get("point_prompts")
            
            # Get point labels from features and convert to SAM format
            # Our point_type: 0=positive/green, 1=negative/red
            # SAM expects: 1=positive, 0=negative
            point_labels = np.ones(len(pts), dtype=int)  # Default to positive
            if point_layer is not None and hasattr(point_layer, 'features'):
                if 'point_type' in point_layer.features:
                    # Convert: 0->1 (positive), 1->0 (negative)
                    point_labels = 1 - point_layer.features['point_type'].values
            
            # Group points by Z-slice
            points_by_z = {}
            labels_by_z = {}
            for idx, point in enumerate(pts):
                z = int(point[0])
                if z not in points_by_z:
                    points_by_z[z] = []
                    labels_by_z[z] = []
                points_by_z[z].append(point[1:])
                labels_by_z[z].append(point_labels[idx] if idx < len(point_labels) else 1)
            
            # Segment each Z-slice
            seg_id = np.zeros(shape, dtype=np.uint32)
            segmented_slices = []
            
            for z, pos_points in points_by_z.items():
                slice_points = np.array(pos_points)
                slice_labels = np.array(labels_by_z[z])
                
                try:
                    mask_2d = sam_util.prompt_segmentation(
                        state.predictor,
                        slice_points,
                        slice_labels,
                        boxes=np.array([]),
                        masks=None,
                        shape=shape[1:],
                        multiple_box_prompts=False,
                        i=z,
                        image_embeddings=state.image_embeddings,
                    )
                    if mask_2d is not None and mask_2d.max() > 0:
                        seg_id[z] = mask_2d
                        segmented_slices.append(z)
                except Exception as e:
                    print(f"Error segmenting slice {z}: {e}")
            
            # Extend through volume
            if len(segmented_slices) > 0:
                try:
                    seg_id, _ = segment_mask_in_volume(
                        seg_id,
                        state.predictor,
                        state.image_embeddings,
                        np.array(segmented_slices),
                        stop_lower=False,
                        stop_upper=False,
                        iou_threshold=0.65,
                        projection="single_point",
                        box_extension=0.05,
                        update_progress=None,
                    )
                except Exception as e:
                    print(f"Error extending segmentation: {e}")
            
            # Update current_object_4d
            if seg_id.max() > 0:
                if self.current_object_4d is None:
                    self.current_object_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
                
                mask = seg_id > 0
                seg_result = np.zeros(shape, dtype=np.uint32)
                seg_result[mask] = 1
                self.current_object_4d[t] = seg_result
                
                # Update layer
                if "current_object_4d" in self._viewer.layers:
                    self._viewer.layers["current_object_4d"].refresh()
                    
                print(f"Segmented object at timestep {t}")
            else:
                print(f"No segmentation result at timestep {t}")
                
        except Exception as e:
            print(f"Error in _segment_current_object: {e}")
            import traceback
            traceback.print_exc()

    def commit_current_timestep(self):
        """Commit current_object_4d at current timestep to committed_objects_4d."""
        try:
            t = self.current_timestep
            
            if self.current_object_4d is None or self.current_object_4d[t].max() == 0:
                print(f"No object to commit at timestep {t}")
                return
            
            # Use committed_objects_4d which is synced with the layer
            if self.committed_objects_4d is None:
                self.committed_objects_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            
            # Get max ID in committed objects to offset new IDs
            max_id = self.committed_objects_4d.max()
            
            # Get current segmentation
            seg = self.current_object_4d[t]
            
            # Assign new IDs
            mask = seg > 0
            if mask.sum() > 0:
                new_id = int(max_id + 1)
                self.committed_objects_4d[t][mask] = new_id
                
                # Clear current object
                self.current_object_4d[t][:] = 0
                
                # Clear point prompts for this timestep
                if hasattr(self, 'point_prompts_4d') and t in self.point_prompts_4d:
                    self.point_prompts_4d[t] = np.empty((0, 3))
                
                # Clear point prompts layer
                if "point_prompts" in self._viewer.layers:
                    point_layer = self._viewer.layers["point_prompts"]
                    point_layer.data = np.empty((0, 3))
                    # Clear features too
                    if hasattr(point_layer, 'features') and len(point_layer.features) > 0:
                        point_layer.features['point_type'] = np.array([], dtype=int)
                        point_layer.features['label'] = np.array([], dtype=object)
                
                # Update layers
                if "committed_objects_4d" in self._viewer.layers:
                    self._viewer.layers["committed_objects_4d"].refresh()
                if "current_object_4d" in self._viewer.layers:
                    self._viewer.layers["current_object_4d"].refresh()
                
                print(f"Committed object {new_id} at timestep {t}")
            
        except Exception as e:
            print(f"Error in commit_current_timestep: {e}")
            import traceback
            traceback.print_exc()

    def _reorder_layers(self):
        """Reorder layers according to desired order and persist across timestep changes."""
        try:
            # Get all layer names currently in viewer
            current_layers = [layer.name for layer in self._viewer.layers]
            
            # Filter desired order to only include layers that exist
            existing_desired = [name for name in self._desired_layer_order if name in current_layers]
            
            # Move layers to desired positions (bottom to top)
            for idx, layer_name in enumerate(existing_desired):
                try:
                    if layer_name in self._viewer.layers:
                        current_idx = self._viewer.layers.index(layer_name)
                        if current_idx != idx:
                            self._viewer.layers.move(current_idx, idx)
                except Exception as e:
                    print(f"[WARN] Failed to move layer {layer_name}: {e}")
                    
        except Exception as e:
            print(f"[WARN] Layer reordering failed: {e}")

            # DROPDOWN REMAPPER WIDGET IF YOU WANT IT BACK
            # # Create and add the ID remapper widget if not already added
            # if not hasattr(self, '_remapper_widget'):
            #     try:
            #         from ._widgets import IdRemapperWidget
            #         from superqt import QCollapsible
            #         remapper = IdRemapperWidget(self)
            #         remapper_widget = QtWidgets.QWidget()
            #         remapper_widget.setLayout(QtWidgets.QVBoxLayout())
            #         remapper_collapsible = QCollapsible("ID Remapper", remapper_widget)
            #         remapper_collapsible.addWidget(remapper)
            #         remapper_widget.layout().addWidget(remapper_collapsible)
            #         if hasattr(self, '_annotator_widget') and hasattr(self._annotator_widget, 'layout'):
            #             self._annotator_widget.layout().insertWidget(1, remapper_widget)
            #         self._remapper_widget = remapper_widget
            #     except Exception as e:
            #         print(f"Failed to create ID remapper widget: {str(e)}")

    def update_image(self, image_4d):
        """Initialize annotator state with a 4D image.

        Adds/updates `raw_4d` and `committed_objects_4d` (4D labels). Also
        creates persistent 3D interactive layers for the current timestep
        that will be updated in-place when switching timesteps.
        """
        if image_4d.ndim != 4:
            raise ValueError(f"Expected 4D data (T,Z,Y,X), got {image_4d.shape}")

        self.image_4d = image_4d
        self.n_timesteps = image_4d.shape[0]
        self.current_timestep = 0

        # configure napari dims for time-series
        try:
            self._viewer.dims.ndim = 4
            self._viewer.dims.axis_labels = ["T", "Z", "Y", "X"]
        except Exception:
            pass
        # add or update persistent 4D raw image layer
        if "raw_4d" in self._viewer.layers:
            try:
                self._viewer.layers["raw_4d"].data = image_4d
                self._viewer.layers["raw_4d"].visible = True
            except Exception:
                pass
        else:
            try:
                self._viewer.add_image(image_4d, name="raw_4d")
            except Exception:
                pass

        # initialize persistent 4D containers and ensure Napari layers reference
        # the same underlying arrays (so edits in Napari mutate our arrays)
        self.segmentation_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.auto_segmentation_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.current_object_4d = np.zeros_like(image_4d, dtype=np.uint32)
        # Initialize 4D-aware point prompts dictionary
        self.point_prompts_4d = {}
        self._segmentation_cache = [self.segmentation_4d[t].copy() for t in range(self.n_timesteps)]

        # add or update persistent 4D labels layers so Napari edits directly
        # mutate the underlying 4D arrays (no per-timestep recreation)
        # Initialize committed_objects_4d if needed
        if self.committed_objects_4d is None:
            self.committed_objects_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
        
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self._viewer.layers["committed_objects_4d"].data = self.committed_objects_4d
            else:
                self._viewer.add_labels(data=self.committed_objects_4d, name="committed_objects_4d")
        except Exception:
            pass

        # Create a container for 4D tools with a styled frame
        try:
            tools_container = QtWidgets.QGroupBox("4D TOOLS")
            tools_container.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #ffb300;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            tools_layout = QtWidgets.QVBoxLayout()
            tools_container.setLayout(tools_layout)
            
            # Removed multi-ID system - using standard micro-sam single object workflow
            # All points belong to one object, segment with 's', commit with 'c'
            
            # Store current point mode (0=positive/green, 1=negative/red)
            # Values match color_cycle index: ["green", "red"]
            self._point_mode = 0
            
            # Add dropdown for positive/negative points
            point_type_row = QtWidgets.QWidget()
            point_type_layout = QtWidgets.QHBoxLayout()
            point_type_row.setLayout(point_type_layout)
            
            point_type_label = QtWidgets.QLabel("<b>Point Mode:</b>")
            point_type_layout.addWidget(point_type_label)
            
            self._point_mode_dropdown = QtWidgets.QComboBox()
            self._point_mode_dropdown.addItem("✓ Positive (Green)", 0)  # 0 = green (first in color_cycle)
            self._point_mode_dropdown.addItem("✗ Negative (Red)", 1)    # 1 = red (second in color_cycle)
            self._point_mode_dropdown.setCurrentIndex(0)  # Default to positive
            self._point_mode_dropdown.setToolTip("Select point type:\n• Positive (Green): Include regions\n• Negative (Red): Exclude regions")
            
            def on_mode_changed(index):
                # Get the mode value (0 or 1) from the selected item
                self._point_mode = self._point_mode_dropdown.itemData(index)
                # Update feature_defaults if layer has points
                if "point_prompts" in self._viewer.layers:
                    layer = self._viewer.layers["point_prompts"]
                    if hasattr(layer, 'data') and len(layer.data) > 0:
                        if hasattr(layer, 'features') and 'point_type' in layer.features:
                            layer.feature_defaults = {
                                'point_type': self._point_mode,
                                'label': str(self._point_mode)
                            }
            
            self._point_mode_dropdown.currentIndexChanged.connect(on_mode_changed)
            
            point_type_layout.addWidget(self._point_mode_dropdown)
            tools_layout.addWidget(point_type_row)
            
            # Add the small 4D timestep tools widget (segment/commit across T)
            tools_layout.addWidget(TimestepToolsWidget(self))
            
            # Removed Point Prompt Manager Widget - single object workflow
            # Removed Object Commit Widget - single object workflow
            
            # Add the container to the annotator widget
            self._annotator_widget.layout().addWidget(tools_container)
        except Exception as e:
            print(f"Failed to create 4D tools container: {e}")
            # Fallback to old method if styling fails
            try:
                self._annotator_widget.layout().addWidget(TimestepToolsWidget(self))
            except Exception:
                pass

        try:
            if "current_object_4d" in self._viewer.layers:
                self._viewer.layers["current_object_4d"].data = self.current_object_4d
            else:
                self._viewer.add_labels(data=self.current_object_4d, name="current_object_4d")
        except Exception:
            pass

        try:
            if "auto_segmentation_4d" in self._viewer.layers:
                self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
            else:
                self._viewer.add_labels(data=self.auto_segmentation_4d, name="auto_segmentation_4d")
        except Exception:
            pass
        
        # make point_prompts 4D-aware: each timestep has its own points
        try:
            t = getattr(self, "current_timestep", 0)
            pts_t = np.array(self.point_prompts_4d.get(t, np.empty((0, 3))))
            
            # if layer exists, update its data
            if "point_prompts" in self._viewer.layers:
                layer = self._viewer.layers["point_prompts"]
                layer.data = pts_t
            else:
                layer = self._viewer.add_points(pts_t, name="point_prompts", size=10,
                                                face_color="green", edge_color="green",
                                                edge_width=2, blending="translucent", ndim=3)
                # Set up positive/negative color cycle (green=positive, red=negative)
                # Use napari's built-in feature system for point types
                layer.face_color_cycle = ["green", "red"]
                layer.edge_color_cycle = ["green", "red"]
                
                # Set up features for point type (0=positive/green, 1=negative/red)
                # Also add 'label' feature for compatibility with base class widgets
                if len(pts_t) > 0:
                    layer.features = {
                        'point_type': np.zeros(len(pts_t), dtype=int),  # 0 = positive/green by default
                        'label': np.array(['0'] * len(pts_t))  # String labels for compatibility
                    }
                    layer.face_color = 'point_type'
                    layer.edge_color = 'point_type'
                    # Default new points based on current mode
                    layer.feature_defaults = {
                        'point_type': getattr(self, '_point_mode', 0),
                        'label': str(getattr(self, '_point_mode', 0))
                    }
                else:
                    # Don't set features or feature_defaults for empty layer
                    # They will be set when first point is added
                    pass
                
            # Setup cursor handling for point prompts layer
            try:
                canvas = self._viewer.window.qt_viewer.canvas.native
                
                def on_mouse_enter(event):
                    """Set crosshair cursor when entering viewer in add mode"""
                    try:
                        if layer.mode == 'add':
                            canvas.setCursor(Qt.CrossCursor)
                    except Exception:
                        pass
                
                def on_mouse_leave(event):
                    """Reset cursor to normal when leaving the viewer"""
                    try:
                        canvas.setCursor(Qt.ArrowCursor)
                    except Exception:
                        pass
                
                # Connect to canvas enter/leave events
                try:
                    original_enter = canvas.enterEvent
                    original_leave = canvas.leaveEvent
                    
                    def new_enter_event(event):
                        try:
                            on_mouse_enter(event)
                            if callable(original_enter):
                                return original_enter(event)
                        except Exception:
                            pass
                    
                    def new_leave_event(event):
                        try:
                            on_mouse_leave(event)
                            if callable(original_leave):
                                return original_leave(event)
                        except Exception:
                            pass
                    
                    canvas.enterEvent = new_enter_event
                    canvas.leaveEvent = new_leave_event
                except Exception:
                    pass
            except Exception:
                pass

            # listener: when user adds/deletes points, save back to the CURRENT timestep
            def _update_point_prompts(event=None):
                # Get current timestep dynamically, not from closure
                t_now = getattr(self, "current_timestep", 0)
                if "point_prompts" not in self._viewer.layers:
                    return
                layer = self._viewer.layers["point_prompts"]
                    
                new_data = np.array(layer.data)
                old_data = self.point_prompts_4d.get(t_now, np.empty((0, 3)))
                
                # Get current point mode
                point_mode = getattr(self, '_point_mode', 1)
                
                # If this is the first point added, set up features
                if len(old_data) == 0 and len(new_data) > 0:
                    # Initialize features and feature_defaults
                    if not hasattr(layer, 'features') or len(layer.features) == 0 or 'point_type' not in layer.features:
                        layer.features = {
                            'point_type': np.full(len(new_data), point_mode, dtype=int),
                            'label': np.array([str(point_mode)] * len(new_data))  # String labels for compatibility
                        }
                        layer.face_color = 'point_type'
                        layer.edge_color = 'point_type'
                        layer.face_color_cycle = ["green", "red"]
                        layer.edge_color_cycle = ["green", "red"]
                        layer.feature_defaults = {
                            'point_type': point_mode,
                            'label': str(point_mode)
                        }
                    else:
                        # Features exist but may have wrong values - update them
                        if 'point_type' in layer.features:
                            layer.features['point_type'] = np.full(len(new_data), point_mode, dtype=int)
                            layer.features['label'] = np.array([str(point_mode)] * len(new_data))
                            layer.feature_defaults = {
                                'point_type': point_mode,
                                'label': str(point_mode)
                            }
                
                # If new points were added, make sure they have the right point_type
                elif len(new_data) > len(old_data):
                    if hasattr(layer, 'features') and 'point_type' in layer.features:
                        # Update feature_defaults
                        layer.feature_defaults = {
                            'point_type': point_mode,
                            'label': str(point_mode)
                        }
                        # Update the newly added points to have the correct type
                        num_new_points = len(new_data) - len(old_data)
                        # Set the last N points (newly added) to the current mode
                        if len(layer.features['point_type']) >= len(new_data):
                            for i in range(len(new_data) - num_new_points, len(new_data)):
                                layer.features['point_type'][i] = point_mode
                                if 'label' in layer.features:
                                    layer.features['label'][i] = str(point_mode)
                    
                    # Refresh the layer to update colors
                    try:
                        layer.refresh_colors()
                    except AttributeError:
                        try:
                            layer.refresh()
                        except Exception:
                            pass
                
                # Detect new points and assign incrementing IDs
                for pt in new_data:
                    key = (int(t_now), int(pt[0]), int(pt[1]), int(pt[2]))
                    if key not in self.point_id_map:
                        # New point - assign next available ID
                        self.point_id_map[key] = self.next_point_id
                        self.next_point_id += 1
                
                # Update point data
                self.point_prompts_4d[t_now] = new_data
                
                # Points use face_color_cycle [green, red] for positive/negative
                # No need to manually update colors - napari handles it via features
                
                # Refresh point manager widget if it exists
                if hasattr(self, "_point_manager_widget"):
                    try:
                        self._point_manager_widget.refresh_point_list()
                    except Exception:
                        pass

            # Store reference and connect
            self._point_prompt_connection = _update_point_prompts
            layer.events.data.connect(_update_point_prompts)
            self._point_prompt_connection_setup = True

        except Exception as e:
            print(f"Error initializing 4D-aware point prompts: {e}")

        # ensure our local arrays are the same object as Napari layer data
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self.segmentation_4d = self._viewer.layers["committed_objects_4d"].data
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
            if "auto_segmentation_4d" in self._viewer.layers:
                self.auto_segmentation_4d = self._viewer.layers["auto_segmentation_4d"].data
        except Exception:
            pass

        # set initial visible timestep via Napari dims (no layer recreation)
        try:
            # set dims current step to (t, z, y, x) = (0, 0, 0, 0)
            self._viewer.dims.current_step = (0,) + (0,) * (self._viewer.dims.ndim - 1)
            # connect dims handler once
            if not getattr(self, "_dims_handler_connected", False):
                self._viewer.dims.events.current_step.connect(self._on_dims_current_step)
                self._dims_handler_connected = True
        except Exception:
            pass
        
        # Apply desired layer order after all layers are created
        try:
            self._reorder_layers()
        except Exception:
            pass

        # final UI hook (no-op by default)
        self._update_timestep_controls()
    

    def _load_timestep(self, t: int):
        """Switch visible timestep by updating Napari dims without recreating layers.

        Persist any UI-only state (points) before switching, update Napari's
        dims.current_step to show the requested T slice, and refresh the
        points layer to show prompts for the new timestep.
        """
        new_t = int(t)
        if self.image_4d is None:
            return
        if not (0 <= new_t < self.n_timesteps):
            return

        prev_t = getattr(self, "current_timestep", None)
        # persist current points
        try:
            if prev_t is not None and "point_prompts" in self._viewer.layers:
                try:
                    pts = np.array(self._viewer.layers["point_prompts"].data)
                    self.point_prompts_4d[prev_t] = pts.copy() if pts.size else np.empty((0, 3))
                except Exception:
                    pass
        except Exception:
            pass

        # set Napari's current_step to the new timestep while preserving
        # the non-time axes (Z/Y/X) if possible. This prevents resetting
        # the Z slider back to 0 when switching timesteps.
        try:
            # get existing step tuple and preserve its non-time entries
            current = list(self._viewer.dims.current_step)
            ndim = max(4, getattr(self._viewer.dims, "ndim", 4))
            if len(current) < ndim:
                current = current + [0] * (ndim - len(current))
            current[0] = new_t
            self._viewer.dims.current_step = tuple(current)
        except Exception:
            try:
                self._viewer.dims.current_step = (new_t,) + (0,) * (max(4, getattr(self._viewer.dims, "ndim", 4)) - 1)
            except Exception:
                pass

        # refresh points layer to the new timestep (in-place)
        try:
            pts_new = self.point_prompts_4d.get(new_t, np.empty((0, 3)))
            lay = self._viewer.layers["point_prompts"] if "point_prompts" in self._viewer.layers else None
            if lay is not None:
                lay.data = np.array(pts_new) if pts_new is not None else np.empty((0, 3))
        except Exception:
            pass

        self.current_timestep = new_t

        if hasattr(self, "_ensure_embeddings_active_for_t"):
            try:
                self._ensure_embeddings_active_for_t(t)
                # activation is silent to avoid spamming the console
                # (previously printed activation messages here)
            except Exception:
                # silently ignore activation errors; callers may inspect state
                pass

    def _preview_timestep(self, t: int, downscale=(4, 4, 4)):
        """Optional preview while scrubbing — currently a no-op to avoid
        replacing the persistent raw layers which would re-order layers.
        """
        return

    def commit_segmentation(self, seg_volume: np.ndarray):
        """Save a 3D segmentation slice into the 4D labels layer in-place.

        Write directly into the `committed_objects_4d` Napari labels layer so
        that no layer recreation is necessary. Refresh the layer after write.
        """
        t = self.current_timestep
        if t is None:
            return

        try:
            layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
            if layer is None:
                # ensure our 4D container exists
                if self.segmentation_4d is None and self.image_4d is not None:
                    self.segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
                    try:
                        self._viewer.add_labels(data=self.segmentation_4d, name="committed_objects_4d")
                        layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
                    except Exception:
                        layer = None

            if layer is not None and seg_volume is not None:
                # Ensure the segmentation slice matches the target shape. If not,
                # attempt a nearest-neighbour resize to avoid broadcasting/aliasing.
                try:
                    target_shape = layer.data.shape[1:]
                    if getattr(seg_volume, "shape", None) != target_shape:
                        seg_volume = _sk_resize(
                            seg_volume.astype("float32"), target_shape, order=0, preserve_range=True, anti_aliasing=False
                        ).astype(seg_volume.dtype)
                except Exception:
                    # If resizing fails, continue and let assignment raise if incompatible.
                    pass

                # Prevent event-driven handlers from seeing an intermediate state
                # and avoid possible aliasing by assigning a copy under the event blocker.
                try:
                    ev = getattr(layer, "events", None)
                    if ev is not None and hasattr(ev, "data"):
                        with layer.events.data.blocker():
                            layer.data[t] = seg_volume.copy()
                    else:
                        layer.data[t] = seg_volume.copy()
                except Exception:
                    # Fallback to direct assignment without blocker.
                    layer.data[t] = seg_volume.copy()

                # keep local ref in sync
                self.segmentation_4d = layer.data
                try:
                    layer.refresh()
                except Exception:
                    pass

                # update cache
                try:
                    if self._segmentation_cache is None:
                        self._segmentation_cache = [None] * self.n_timesteps
                    self._segmentation_cache[t] = self.segmentation_4d[t].copy()
                except Exception:
                    pass
        except Exception:
            pass

        print(f"✅ Committed segmentation for timestep {t}")

    def save_current_object_to_4d(self):
        """Ensure `current_object_4d` Napari layer and local array are in sync.

        Editing is expected to happen directly on the 4D `current_object_4d`
        labels layer. This method refreshes the local reference so the
        Python-side array remains the same object as Napari layer.data.
        """
        try:
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
        except Exception:
            pass

    def save_point_prompts(self):
        """Save the point_prompts 3D layer into per-timestep storage."""
        t = int(getattr(self, "current_timestep", 0) or 0)
        if "point_prompts" in self._viewer.layers:
            try:
                pts = np.array(self._viewer.layers["point_prompts"].data)
                self.point_prompts_4d[t] = pts.copy() if pts.size else np.empty((0, 3))
            except Exception:
                pass
    
    def _get_point_id_from_coords(self, t, z, y, x):
        """Get the assigned ID for a point at given coordinates.
        
        Args:
            t: timestep
            z, y, x: point coordinates (rounded to int)
            
        Returns:
            The assigned ID (default 1 if not found)
        """
        key = (int(t), int(z), int(y), int(x))
        return self.point_id_map.get(key, 1)
    
    def _set_point_id_from_coords(self, t, z, y, x, point_id):
        """Set the assigned ID for a point at given coordinates.
        
        Args:
            t: timestep
            z, y, x: point coordinates (rounded to int)
            point_id: the ID to assign
        """
        key = (int(t), int(z), int(y), int(x))
        self.point_id_map[key] = int(point_id)
    
    def _get_point_ids_for_timestep(self, t, points_array):
        """Get IDs for all points in a timestep based on their coordinates.
        
        Args:
            t: timestep
            points_array: numpy array of shape (N, 3) with columns [z, y, x]
            
        Returns:
            List of IDs for each point (default 1 if not found)
        """
        if len(points_array) == 0:
            return []
        return [self._get_point_id_from_coords(t, pt[0], pt[1], pt[2]) for pt in points_array]
    
    def _update_point_colors(self, layer, point_ids):
        """Update point colors based on their assigned IDs.
        
        Args:
            layer: The napari points layer
            point_ids: List of IDs corresponding to each point
        """
        try:
            if len(point_ids) == 0:
                return
            
            # Color palette for different IDs
            color_palette = [
                [0.0, 1.0, 0.0, 1.0],    # ID 1: Green
                [1.0, 1.0, 0.0, 1.0],    # ID 2: Yellow
                [0.0, 1.0, 1.0, 1.0],    # ID 3: Cyan
                [1.0, 0.0, 1.0, 1.0],    # ID 4: Magenta
                [1.0, 0.5, 0.0, 1.0],    # ID 5: Orange
                [1.0, 0.75, 0.8, 1.0],   # ID 6: Pink
                [0.5, 1.0, 0.0, 1.0],    # ID 7: Lime
                [0.5, 0.0, 1.0, 1.0],    # ID 8: Purple
                [0.0, 0.5, 0.5, 1.0],    # ID 9: Teal
                [1.0, 0.0, 0.0, 1.0],    # ID 10: Red
            ]
            
            # Assign colors based on IDs
            colors = []
            for point_id in point_ids:
                color_idx = (point_id - 1) % len(color_palette)
                colors.append(color_palette[color_idx])
            
            # Update both face and edge colors
            layer.face_color = np.array(colors)
            layer.edge_color = np.array(colors)
            
        except Exception as e:
            print(f"Failed to update point colors: {e}")

    # ----------------- Embedding helpers for 4D -----------------
    def compute_embeddings_for_timestep(self, t: int, model_type: str = None, device: str | None = None, save_path: str | None = None, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute image embeddings for a single timestep and store them in AnnotatorState.

        This wraps AnnotatorState.initialize_predictor for convenience when working with 4D (T,Z,Y,X)
        data. It extracts the 3D volume at timestep `t` and computes embeddings with ndim=3.
        """
        from ._state import AnnotatorState

        if self.image_4d is None:
            raise RuntimeError("No 4D image loaded")
        if not (0 <= t < self.n_timesteps):
            raise IndexError("t out of range")

        image3d = self.image_4d[int(t)]
        state = AnnotatorState()
        # default model_type if not provided
        model_type = model_type or getattr(state, "predictor", None) and getattr(state.predictor, "model_name", None) or None
        # initialize predictor and compute embeddings for this 3D volume
        state.initialize_predictor(
            image3d,
            model_type=model_type or "vit_b_lm",
            ndim=3,
            save_path=save_path,
            device=device,
            tile_shape=tile_shape,
            halo=halo,
            prefer_decoder=prefer_decoder,
        )
        # Capture the computed embeddings and store them per-timestep only
        try:
            embeds = state.image_embeddings
        except Exception:
            embeds = None
        try:
            if embeds is not None:
                # store only in per-timestep cache
                self.embeddings_4d[int(t)] = embeds
        except Exception:
            pass

        # Only bind the embeddings into the global AnnotatorState if this
        # timestep is currently active; otherwise detach to avoid leaking the
        # embedding globally across timesteps.
        try:
            if int(getattr(self, "current_timestep", 0) or 0) == int(t):
                try:
                    state.image_embeddings = embeds
                    # update active marker
                    self._active_embedding_t = int(t)
                except Exception:
                    pass
            else:
                try:
                    # detach any embeddings from the global state
                    state.image_embeddings = None
                    # do not clear predictor, only the embedding handle
                    if getattr(state, "embedding_path", None) is not None and state.embedding_path == save_path:
                        state.embedding_path = None
                except Exception:
                    pass
        except Exception:
            pass
        # Update the state's image_name so widgets reflect the selection
        try:
            state.image_name = (self._viewer.layers["raw_4d"].name if "raw_4d" in self._viewer.layers else state.image_name)
        except Exception:
            pass
        # Ensure AnnotatorState has image_shape / image_scale set so downstream
        # segmentation widgets and helpers (which read state.image_shape) work
        # when embeddings are computed via this helper (instead of the embedding widget).
        try:
            # image3d has shape (Z, Y, X)
            state.image_shape = tuple(image3d.shape)
        except Exception:
            pass
        try:
            layer = self._viewer.layers["raw_4d"] if "raw_4d" in self._viewer.layers else None
            if layer is not None:
                # Napari image layer scale for raw_4d is (T, Z, Y, X). Use the spatial part.
                scale = getattr(layer, "scale", None)
                if scale is not None and len(scale) >= 4:
                    state.image_scale = tuple(scale[1:])
                elif scale is not None and len(scale) == 3:
                    state.image_scale = tuple(scale)
        except Exception:
            pass

        # return the per-timestep stored embedding (may be None)
        return self.embeddings_4d.get(int(t))

    def compute_embeddings_for_all_timesteps(self, model_type: str = None, device: str | None = None, base_save_path: str | None = None, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute embeddings for every timestep. Saves to separate files if base_save_path is provided.

        WARNING: this can be slow and memory/disk intensive. Use with care.
        Returns a list of image_embeddings objects (one per timestep).
        """
        results = []
        # ensure we have a per-timestep embeddings dict
        if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
            self.embeddings_4d = {}

        for t in range(self.n_timesteps):
            sp = None if base_save_path is None else f"{base_save_path}_t{t}.zarr"
            embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=sp, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
            results.append(embeds)
            try:
                # store embedding for this timestep; keep whatever structure compute returned
                self.embeddings_4d[int(t)] = embeds
            except Exception:
                # best-effort: ignore failures to cache
                pass
        return results

    def compute_and_save_embeddings(self, output_dir: str | Path, model_type: str = None, device: str | None = None, per_timestep: bool = True, overwrite: bool = False, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute embeddings for all timesteps and save into `output_dir`.

        By default this creates one zarr store per timestep named
        `embeddings_t{t}.zarr` inside `output_dir`. If `per_timestep` is
        False the method will attempt to compute embeddings in-memory and
        store a single `embeddings.npz` file (may be large).

        Returns a list of embedding objects (as returned by precompute), and
        writes a `manifest.json` into `output_dir` describing the files.
        """
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)

        manifest = {
            "n_timesteps": int(self.n_timesteps),
            "per_timestep": bool(per_timestep),
            "files": {},
        }

        results = []
        if per_timestep:
            for t in range(self.n_timesteps):
                fname = outp / f"embeddings_t{t}.zarr"
                if fname.exists() and not overwrite:
                    # load existing (skip compute)
                    results.append({"path": str(fname)})
                    manifest["files"][str(t)] = str(fname.name)
                    continue

                # compute and save to zarr
                save_path = str(fname)
                embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=save_path, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
                results.append(embeds)
                manifest["files"][str(t)] = str(fname.name)

        else:
            # compute all embeddings in-memory and write a single npz
            arrs = {}
            for t in range(self.n_timesteps):
                embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=None, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
                # `embeds` is a dict containing 'features' (numpy or zarr)
                feats = embeds.get("features")
                # If features are zarr-like, read into memory (could be large)
                try:
                    if hasattr(feats, "[:]"):
                        feats_np = feats[:]  # zarr or numpy-like
                    else:
                        feats_np = np.asarray(feats)
                except Exception:
                    feats_np = np.asarray(feats)
                arrs[f"t{t}"] = feats_np
                results.append({"features": feats_np})

            npz_path = outp / "embeddings.npz"
            if npz_path.exists() and not overwrite:
                raise FileExistsError(f"{npz_path} already exists. Use overwrite=True to replace.")
            np.savez_compressed(str(npz_path), **arrs)
            manifest["files"] = {str(t): str(npz_path.name) for t in range(self.n_timesteps)}

        # write manifest
        manifest_path = outp / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        return results

    def load_saved_embeddings(self, path: str | Path, lazy: bool = True):
        """Load embeddings previously saved with `compute_and_save_embeddings`.

        Args:
            path: Directory containing embeddings (manifest.json) or a single .npz file.
            lazy: If True and the embeddings are stored as zarr stores, return zarr objects
                  without loading full arrays into memory. If False, load numpy arrays.

        Returns:
            A dict mapping timestep (int) -> embedding dict (with at least key 'features').
        """
        p = Path(path)
        results = {}
        if p.is_dir():
            # Detect a parent directory containing per-timestep subfolders
            # named like t0, t1, ... and map them to results[t] = {"path": str(t_folder)}
            try:
                subdirs = [d for d in sorted(p.iterdir()) if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()]
                if subdirs:
                    for d in subdirs:
                        try:
                            t = int(d.name[1:])
                            results[t] = {"path": str(d)}
                        except Exception:
                            # ignore non-numeric t* dirs
                            pass
                    return results
            except Exception:
                # fall through to legacy manifest/zarr discovery
                pass
            manifest = p / "manifest.json"
            if not manifest.exists():
                # try to discover files by pattern embeddings_t*.zarr
                zarrs = sorted(p.glob("embeddings_t*.zarr"))
                if not zarrs:
                    raise FileNotFoundError(f"No manifest.json or embeddings_* found in {p}")
                for z in zarrs:
                    tstr = z.stem.split("_t")[-1]
                    import zarr as _zarr
                    f = _zarr.open(str(z), mode="r")
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"Skipping {z}: no array-like 'features' dataset found.")
                        except Exception:
                            pass
                        continue
                    # Try to surface input/original size metadata expected by downstream code
                    attrs = getattr(feats, "attrs", {}) or {}
                    # If the zarr store doesn't contain tiling metadata, synthesize
                    # a conservative `input_size`/`original_size` so downstream
                    # prompt-based segmentation treats this as a non-tiled embedding
                    # (avoids accessing attrs['shape'] which may be missing).
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        # fallback: infer spatial size from the last two dimensions
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    results[int(tstr)] = {"features": feats, "input_size": input_size, "original_size": original_size}
                return results

            with open(manifest, "r") as fh:
                manifestd = json.load(fh)
            for tstr, fname in manifestd.get("files", {}).items():
                t = int(tstr)
                filep = p / fname
                if filep.suffix == ".npz":
                    data = np.load(str(filep))
                    # assume key 't{t}' exists
                    arr = data.get(f"t{t}")
                    # npz does not contain metadata about input/original size -> leave as None
                    results[t] = {"features": arr, "input_size": None, "original_size": None}
                else:
                    import zarr as _zarr
                    f = _zarr.open(str(filep), mode="r")
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"Skipping {filep}: no array-like 'features' dataset found.")
                        except Exception:
                            pass
                        continue
                    attrs = getattr(feats, "attrs", {}) or {}
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    if lazy:
                        results[t] = {"features": feats, "input_size": input_size, "original_size": original_size}
                    else:
                        results[t] = {"features": feats[:], "input_size": input_size, "original_size": original_size}
            return results

        elif p.is_file() and p.suffix == ".npz":
            data = np.load(str(p))
            for key in data.files:
                if key.startswith("t"):
                    t = int(key[1:])
                    results[t] = {"features": data[key], "input_size": None, "original_size": None}
            return results
        else:
            raise FileNotFoundError(f"No embeddings found at {path}")

    def set_embeddings_folder(self, path: str | Path, lazy: bool = True):
        """Set a directory containing per-timestep embeddings and load the mapping.

        Stores the path in `self._last_embeddings_dir` and populates `self.embeddings_4d`.
        Returns the loaded mapping (t -> embedding entry).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embeddings path does not exist: {p}")
        self._last_embeddings_dir = str(p)
        loaded = self.load_saved_embeddings(p, lazy=lazy)
        # store mapping in per-timestep cache
        try:
            self.embeddings_4d = {int(k): v for k, v in loaded.items()}
        except Exception:
            self.embeddings_4d = loaded
        return self.embeddings_4d

    def reload_embeddings_from_last(self, lazy: bool = True):
        """Reload embeddings mapping from the last set embeddings directory.

        Raises RuntimeError if no directory was previously set via `set_embeddings_folder` or
        by the compute/save helpers. Returns the loaded mapping.
        """
        if not getattr(self, "_last_embeddings_dir", None):
            raise RuntimeError("No last embeddings directory is set. Call set_embeddings_folder(path) first.")
        return self.set_embeddings_folder(self._last_embeddings_dir, lazy=lazy)

    def _load_embedding_for_timestep(self, t: int):
        """Load embeddings for a specific timestep from disk (lazy loading helper).
        
        This method is called when switching to a timestep that has a lazy embedding entry.
        It loads the embedding from disk and activates it in AnnotatorState.
        """
        try:
            if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                return
            
            entry = self.embeddings_4d.get(t)
            if entry is None:
                return
            
            # If entry only has a path, load it from disk
            if isinstance(entry, dict) and "path" in entry and "features" not in entry:
                import zarr as _zarr
                # _select_array_from_zarr_group is defined at the top of this file
                
                path = Path(entry["path"])
                if not path.exists():
                    print(f"Warning: Embedding file not found: {path}")
                    return
                
                # Load the zarr file
                f = _zarr.open(str(path), mode="r")
                feats = _select_array_from_zarr_group(f)
                
                if feats is None:
                    print(f"Warning: No features found in {path}")
                    return
                
                # Get metadata from array attrs and root group attrs; fall back to image shape
                arr_attrs = dict(getattr(feats, "attrs", {}) or {})
                root_attrs = dict(getattr(f, "attrs", {}) or {})
                input_size = arr_attrs.get("input_size") or root_attrs.get("input_size")
                original_size = arr_attrs.get("original_size") or root_attrs.get("original_size")

                # Final fallback: use the current image shape (Y, X) rather than feature shape
                if original_size is None:
                    try:
                        img_shape_yx = tuple(self.image_4d[int(t)].shape[-2:])
                        original_size = img_shape_yx
                    except Exception:
                        # As a last resort do not guess from feats (would be 64x64 and wrong)
                        original_size = None
                if input_size is None:
                    input_size = original_size
                
                # Update the entry with loaded features
                self.embeddings_4d[t] = {
                    "features": feats,
                    "input_size": input_size,
                    "original_size": original_size,
                    "path": str(path)
                }
            
            # Now activate the embeddings for this timestep
            self._ensure_embeddings_active_for_t(t)
            
        except Exception as e:
            print(f"Failed to load embedding for timestep {t}: {e}")

    def _materialize_embedding_entry(self, entry):
        """Best-effort materialize a saved embedding entry.

        Accepts:
          - dict with 'features' -> returned as-is
          - dict with 'path' -> path to a .zarr store (or parent dir)
          - str path -> treated like dict with 'path'

        Returns embedding dict or None on failure.
        """
        if entry is None:
            return None
        # already materialized
        if isinstance(entry, dict) and "features" in entry:
            return entry

        p = None
        if isinstance(entry, dict) and "path" in entry and entry["path"]:
            p = Path(entry["path"])
        elif isinstance(entry, str):
            p = Path(entry)

        if p is None:
            return None

        # If p points to a per-timestep folder like 't3', treat it as a lazy
        # entry referencing that folder and return a simple {'path': str(p)}
        try:
            if p.is_dir() and p.name.startswith("t") and p.name[1:].isdigit():
                return {"path": str(p)}
        except Exception:
            pass

        # If given a parent directory, try to load via manifest (may return a mapping)
        try:
            if p.is_dir():
                loaded = self.load_saved_embeddings(p, lazy=True)
                # prefer the exact timestep if present; otherwise return the mapping
                return loaded
        except Exception:
            pass

            # If it's a zarr store, open it and return the 'features' object with synthesized metadata
        try:
            if p.suffix == ".zarr" or (p.exists() and p.is_dir() and any(p.glob("*.zarr"))):
                import zarr as _zarr

                # If p is a file-like zarr store, open it directly
                try:
                    f = _zarr.open(str(p), mode="r")
                except Exception:
                    # maybe the path points to a file-like directory; try opening parent
                    try:
                        f = _zarr.open(str(p), mode="r")
                    except Exception:
                        f = None

                if f is not None:
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"No suitable array-like dataset found inside {p}; cannot materialize embeddings.")
                        except Exception:
                            pass
                        return None
                    arr_attrs = dict(getattr(feats, "attrs", {}) or {})
                    root_attrs = dict(getattr(f, "attrs", {}) or {})
                    input_size = arr_attrs.get("input_size") or root_attrs.get("input_size")
                    original_size = arr_attrs.get("original_size") or root_attrs.get("original_size")
                    if original_size is None:
                        try:
                            # Use annotator image shape if available
                            # Note: we cannot know 't' here reliably; leave None and let activation fill in
                            original_size = None
                        except Exception:
                            original_size = None
                    if input_size is None:
                        input_size = original_size
                    return {"features": feats, "input_size": input_size, "original_size": original_size}
        except Exception:
            pass

        # Last resort: if it's an npz file, let load_saved_embeddings handle it
        try:
            if p.exists() and p.suffix == ".npz":
                loaded = self.load_saved_embeddings(p)
                # return the first timestep's entry if single
                if isinstance(loaded, dict):
                    # if only one entry, return that
                    if len(loaded) == 1:
                        return list(loaded.values())[0]
                    return loaded
        except Exception:
            pass

        return None

    def _ensure_embeddings_active_for_t(self, t: int):
        """If we have cached embeddings for timestep t, activate them on AnnotatorState.

        This will materialize lazy entries if necessary.
        """
        try:
            if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                return
            # Detach any previously active embeddings if switching timesteps
            try:
                state_detach = AnnotatorState()
                if getattr(self, "_active_embedding_t", None) is not None and self._active_embedding_t != int(t):
                    try:
                        state_detach.image_embeddings = None
                        state_detach.embedding_path = None
                    except Exception:
                        pass
                    try:
                        self._active_embedding_t = None
                    except Exception:
                        pass
            except Exception:
                pass

            entry = self.embeddings_4d.get(int(t))
            if entry is None:
                # No embeddings for this timestep: ensure global state is cleared
                try:
                    state_clear = AnnotatorState()
                    state_clear.image_embeddings = None
                    state_clear.embedding_path = None
                except Exception:
                    pass
                return

            # If entry is a mapping of multiple timesteps (returned from load_saved_embeddings on a dir),
            # prefer the exact t key.
            if isinstance(entry, dict) and any(isinstance(k, str) and k.isdigit() for k in entry.keys()):
                # already a mapping produced by load_saved_embeddings; pick the t entry if present
                maybe = entry.get(str(t)) or entry.get(int(t))
                if maybe is not None:
                    entry = maybe

            # If entry is a lazy path or string, materialize. Do this in background
            # to avoid blocking the UI for large zarr loads.
            if not (isinstance(entry, dict) and "features" in entry):
                # If already loading, return early
                if self._embedding_loading.get(int(t), False):
                    return

                # Mark as loading and spawn background thread
                self._embedding_loading[int(t)] = True

                def _bg():
                    try:
                        mat = self._materialize_embedding_entry(entry)
                        if mat is not None:
                            try:
                                self.embeddings_4d[int(t)] = mat
                            except Exception:
                                pass
                            # once materialized, call this method again to perform activation
                            try:
                                # clear loading flag before re-entering
                                self._embedding_loading[int(t)] = False
                            except Exception:
                                pass
                            try:
                                # call activation synchronously now that mat exists
                                self._ensure_embeddings_active_for_t(t)
                            except Exception:
                                pass
                        else:
                            try:
                                show_info(f"Failed to materialize embeddings for timestep {t}.")
                            except Exception:
                                pass
                    finally:
                        try:
                            self._embedding_loading[int(t)] = False
                        except Exception:
                            pass

                thread = threading.Thread(target=_bg, daemon=True)
                thread.start()
                return

            # Finally, set AnnotatorState's image_embeddings to this dict so downstream code uses it
            try:
                state = AnnotatorState()
                # if predictor is missing we'll (re)initialize it for this timestep
                image3d = None
                try:
                    image3d = self.image_4d[int(t)]
                except Exception:
                    image3d = None

                # If the entry is already materialized and contains features, use it as save_path
                save_path = None
                if isinstance(entry, dict) and "features" in entry:
                    # Ensure minimal metadata is present; do not reject on mismatch
                    if image3d is not None:
                        expected_shape = tuple(image3d.shape[-2:])
                        if entry.get("original_size") is None:
                            entry["original_size"] = expected_shape
                        if entry.get("input_size") is None:
                            entry["input_size"] = entry.get("original_size")
                    save_path = entry
                elif isinstance(entry, dict) and entry.get("path"):
                    save_path = str(entry.get("path"))
                elif isinstance(entry, str):
                    save_path = entry

                # Determine a model_type to use if predictor is not present
                try:
                    model_type = getattr(state.predictor, "model_type", None) or getattr(state.predictor, "model_name", None)
                except Exception:
                    model_type = None
                if model_type is None:
                    try:
                        model_type = _vutil._DEFAULT_MODEL
                    except Exception:
                        model_type = "vit_b_lm"

                # Initialize predictor / embeddings for this timestep. If predictor already exists,
                # initialize_predictor will reuse it.
                try:
                    if image3d is not None:
                        state.initialize_predictor(
                            image3d,
                            model_type=model_type,
                            ndim=3,
                            save_path=save_path,
                            predictor=getattr(state, "predictor", None),
                            prefer_decoder=getattr(state, "decoder", None) is not None,
                        )
                except Exception:
                    # Continue even if predictor init fails; downstream code may handle it.
                    pass

                # ensure state.image_embeddings references the entry
                try:
                    state.image_embeddings = entry
                    try:
                        self._active_embedding_t = int(t)
                    except Exception:
                        pass
                except Exception:
                    pass

                # also set embedding_path if available (do this early so loaders can access files)
                try:
                    if isinstance(entry, dict) and entry.get("path"):
                        state.embedding_path = str(entry.get("path"))
                    elif isinstance(entry, str):
                        state.embedding_path = entry
                except Exception:
                    pass

                # Ensure image_shape/scale/name are set for downstream widgets
                try:
                    if image3d is not None:
                        state.image_shape = tuple(image3d.shape)
                except Exception:
                    pass
                try:
                    layer = self._viewer.layers["raw_4d"] if "raw_4d" in self._viewer.layers else None
                    if layer is not None:
                        scale = getattr(layer, "scale", None)
                        if scale is not None and len(scale) >= 4:
                            state.image_scale = tuple(scale[1:])
                        elif scale is not None and len(scale) == 3:
                            state.image_scale = tuple(scale)
                except Exception:
                    pass

                # Ensure AMG state exists. If embeddings were saved on disk (embedding_path), try to load
                # the precomputed amg/is state. Otherwise compute AMG state in-memory.
                try:
                    if state.amg_state is None:
                        # If embedding_path exists on disk, load cached AMG/IS state
                        if getattr(state, "embedding_path", None):
                            try:
                                if state.decoder is not None:
                                    state.amg_state = _load_is_state(state.embedding_path)
                                else:
                                    state.amg_state = _load_amg_state(state.embedding_path)
                            except Exception:
                                state.amg_state = None

                        # If still missing and we have in-memory embeddings, compute AMG state now
                        if state.amg_state is None and isinstance(state.image_embeddings, dict) and "features" in state.image_embeddings and image3d is not None:
                            try:
                                is_tiled = state.image_embeddings.get("input_size") is None
                                amg = instance_segmentation.get_amg(state.predictor, is_tiled=is_tiled, decoder=state.decoder)
                                # initialize amg on the full 3D volume
                                amg.initialize(image3d, image_embeddings=state.image_embeddings, verbose=False)
                                state.amg = amg
                                state.amg_state = amg.get_state()
                            except Exception:
                                # best-effort; leave amg_state None if computation fails
                                pass
                except Exception:
                    pass

                # Show a concise one-time info about active embeddings for this timestep
                try:
                    if int(t) not in getattr(self, "_reported_embedding_info_t", set()):
                        orig = None
                        tiled = False
                        src_name = "memory"
                        try:
                            if isinstance(entry, dict):
                                orig = entry.get("original_size")
                                feats = entry.get("features")
                                if feats is not None and hasattr(feats, "attrs"):
                                    tiled = feats.attrs.get("tile_shape") is not None
                                pth = entry.get("path")
                                if pth:
                                    src_name = Path(pth).name
                            elif isinstance(entry, str):
                                src_name = Path(entry).name
                        except Exception:
                            pass
                        try:
                            self._reported_embedding_info_t.add(int(t))
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _on_dims_current_step(self, event):
        """Handle napari dims current_step changes (time slider).
        Persists current 3D edits and loads the new timestep into the 3D views.
        Each timestep now keeps independent point prompts (4D-aware).
        """
        import numpy as np

        # --- detect new timestep ---
        try:
            val = getattr(event, "value", None)
            if val is None:
                val = getattr(event, "current_step", None) or getattr(self._viewer.dims, "current_step", None)
            new_t = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        except Exception:
            return

        prev_t = getattr(self, "current_timestep", None)
        if new_t == prev_t:
            return

        # --- persist old points ---
        try:
            if prev_t is not None and "point_prompts" in self._viewer.layers:
                pts = np.array(self._viewer.layers["point_prompts"].data)
                if not hasattr(self, "point_prompts_4d"):
                    self.point_prompts_4d = {}
                self.point_prompts_4d[prev_t] = pts.copy() if pts.size else np.empty((0, 3))
                # Point IDs are already tracked in coordinate-based point_id_map
        except Exception as e:
            print(f"[WARN] Save point prompts failed: {e}")

        # --- update current timestep ---
        self.current_timestep = new_t

        # --- embeddings switching ---
        try:
            mgr = getattr(self, "timestep_embedding_manager", None)
            if mgr is not None:
                try:
                    mgr.on_timestep_changed(new_t)
                except Exception:
                    self._ensure_embeddings_active_for_t(new_t)
            else:
                self._ensure_embeddings_active_for_t(new_t)
        except Exception:
            pass

        # --- load the new timestep volume ---
        try:
            self._load_timestep(new_t)
        except Exception as e:
            print(f"[WARN] Load timestep failed: {e}")

        # --- load per-timestep point prompts ---
        try:
            if not hasattr(self, "point_prompts_4d"):
                self.point_prompts_4d = {}

            new_pts = self.point_prompts_4d.get(new_t, np.empty((0, 3)))
            
            # Get point IDs based on coordinates
            point_ids = self._get_point_ids_for_timestep(new_t, new_pts)

            # if layer doesn't exist yet, create once
            if "point_prompts" not in self._viewer.layers:
                layer = self._viewer.add_points(
                    new_pts,
                    name="point_prompts",
                    size=10,
                    face_color="green",  # SAM positive color
                    edge_color="green",
                    edge_width=0,
                    blending="translucent",
                    ndim=3,
                )
                
                # Update colors based on IDs
                self._update_point_colors(layer, point_ids)
                
                # Setup cursor handling for point prompts layer
                try:
                    canvas = self._viewer.window.qt_viewer.canvas.native
                    
                    def on_mouse_enter(event):
                        """Set crosshair cursor when entering viewer in add mode"""
                        try:
                            if layer.mode == 'add':
                                canvas.setCursor(Qt.CrossCursor)
                        except Exception:
                            pass
                    
                    def on_mouse_leave(event):
                        """Reset cursor to normal when leaving the viewer"""
                        try:
                            canvas.setCursor(Qt.ArrowCursor)
                        except Exception:
                            pass
                    
                    # Connect to canvas enter/leave events
                    try:
                        original_enter = canvas.enterEvent
                        original_leave = canvas.leaveEvent
                        
                        def new_enter_event(event):
                            try:
                                on_mouse_enter(event)
                                if callable(original_enter):
                                    return original_enter(event)
                            except Exception:
                                pass
                        
                        def new_leave_event(event):
                            try:
                                on_mouse_leave(event)
                                if callable(original_leave):
                                    return original_leave(event)
                            except Exception:
                                pass
                        
                        canvas.enterEvent = new_enter_event
                        canvas.leaveEvent = new_leave_event
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                layer = self._viewer.layers["point_prompts"]
                # Update stored point data BEFORE updating layer to prevent callback confusion
                self.point_prompts_4d[new_t] = new_pts
                
                # Block events when updating data to prevent ID reset
                if hasattr(layer, 'events') and hasattr(layer.events, 'data'):
                    with layer.events.data.blocker():
                        layer.data = new_pts
                else:
                    layer.data = new_pts
                # Update colors for the new timestep
                self._update_point_colors(layer, point_ids)

            # --- setup event listener ONCE if not already connected ---
            if not hasattr(self, "_point_prompt_connection_setup"):
                try:
                    def _update_points(event=None):
                        t_now = getattr(self, "current_timestep", 0)
                        if "point_prompts" not in self._viewer.layers:
                            return
                        layer = self._viewer.layers["point_prompts"]
                        
                        new_data = np.array(layer.data)
                        
                        # Update point data
                        self.point_prompts_4d[t_now] = new_data
                        
                        # Get IDs based on coordinates (automatically handles new points with default ID 1)
                        point_ids = self._get_point_ids_for_timestep(t_now, new_data)
                        
                        # Update colors based on IDs
                        self._update_point_colors(layer, point_ids)
                        
                        # Refresh point manager widget if it exists
                        if hasattr(self, "_point_manager_widget"):
                            try:
                                self._point_manager_widget.refresh_point_list()
                            except Exception:
                                pass

                    self._point_prompt_connection = _update_points
                    layer.events.data.connect(self._point_prompt_connection)
                    self._point_prompt_connection_setup = True
                except Exception as e:
                    print(f"[WARN] Failed to setup point prompt connection: {e}")

        except Exception as e:
            print(f"[WARN] Reload point prompts failed: {e}")
        
        # Refresh point manager widget after timestep change
        try:
            if hasattr(self, "_point_manager_widget"):
                self._point_manager_widget.refresh_point_list()
        except Exception:
            pass
        
        # Restore layer order after timestep change
        try:
            self._reorder_layers()
        except Exception:
            pass

    def next_timestep(self):
        """Move to the next timestep (if available)."""
        self.save_current_object_to_4d()
        self.save_point_prompts()
        if self.current_timestep + 1 < self.n_timesteps:
            self._load_timestep(self.current_timestep + 1)
        else:
            print("🚫 Already at last timestep.")

    def _update_timestep_controls(self):
        """Placeholder for UI controls update (no-op)."""
        pass

    # ----------------- Automatic segmentation helpers for 4D -----------------
    def auto_segment_timestep(self, t: int, mode: str = "auto", device: str | None = None, tile_shape=None, halo=None, gap_closing: int | None = None, min_z_extent: int | None = None, with_background: bool = True, min_object_size: int = 100, prefer_decoder: bool = True):
        """Run automatic 3D segmentation for a single timestep and store result in auto_segmentation_4d[t].

        This uses the project's `automatic_3d_segmentation` implementation which handles tiled vs untiled
        embeddings and decoder vs AMG-based segmentation.
        """
        if self.image_4d is None:
            raise RuntimeError("No 4D image loaded")
        if not (0 <= t < self.n_timesteps):
            raise IndexError("t out of range")

        # Ensure embeddings and predictor are initialized for this timestep.
        state = AnnotatorState()
        # If no predictor or embeddings are present for this timestep, compute them.
        if state.predictor is None or state.image_embeddings is None:
            # compute embeddings for this timestep (store per-timestep only)
            self.compute_embeddings_for_timestep(t=t, model_type=None, device=device, save_path=None, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
            # After computing, bind the per-timestep embeddings into the global state
            try:
                emb = self.embeddings_4d.get(int(t))
                if emb is None:
                    try:
                        show_info(f"❌ No embeddings available for timestep {t}; cannot run segmentation.")
                    except Exception:
                        print(f"❌ No embeddings available for timestep {t}; cannot run segmentation.")
                    return None
                state.image_embeddings = emb
                # set embedding path if available
                try:
                    if isinstance(emb, dict) and emb.get("path"):
                        state.embedding_path = str(emb.get("path"))
                except Exception:
                    pass
                # mark active
                try:
                    self._active_embedding_t = int(t)
                except Exception:
                    pass
            except Exception:
                pass

        predictor = state.predictor

        # Determine if tiled embeddings are used.
        is_tiled = False
        try:
            feats = state.image_embeddings.get("features") if isinstance(state.image_embeddings, dict) else None
            if feats is not None and hasattr(feats, "attrs") and feats.attrs.get("tile_shape") is not None:
                is_tiled = True
        except Exception:
            is_tiled = False

        # Create segmentor (AMG or decoder-based) using existing util.
        segmentor = instance_segmentation.get_amg(predictor, is_tiled=is_tiled, decoder=state.decoder)

        # Extract the 3D volume for this timestep
        vol3d = np.asarray(self.image_4d[int(t)])

        # Run automatic 3d segmentation
        seg = automatic_3d_segmentation(
            volume=vol3d,
            predictor=predictor,
            segmentor=segmentor,
            embedding_path=state.embedding_path,
            with_background=with_background,
            gap_closing=gap_closing,
            min_z_extent=min_z_extent,
            tile_shape=tile_shape,
            halo=halo,
            verbose=True,
            return_embeddings=False,
            min_object_size=min_object_size,
        )

        # Ensure segmentation output matches the volume shape (Z,Y,X). If not,
        # resize the segmentation volume (nearest-neighbour) to the target shape.
        try:
            target_shape = vol3d.shape
            if getattr(seg, "shape", None) != target_shape:
                seg = _sk_resize(seg.astype("float32"), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(seg.dtype)
        except Exception:
            # If resizing fails, continue and let assignment handle or raise.
            pass

        # Ensure our container exists
        if self.auto_segmentation_4d is None:
            self.auto_segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                if "auto_segmentation_4d" in self._viewer.layers:
                    self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
                else:
                    self._viewer.add_labels(data=self.auto_segmentation_4d, name="auto_segmentation_4d")
            except Exception:
                pass

        # Write back into the 4D auto segmentation container and refresh layer in-place
        try:
            self.auto_segmentation_4d[int(t)] = seg
            layer = self._viewer.layers["auto_segmentation_4d"] if "auto_segmentation_4d" in self._viewer.layers else None
            if layer is not None:
                layer.data[int(t)] = seg
                try:
                    layer.refresh()
                except Exception:
                    pass
        except Exception:
            # As a fallback, try replacing the whole layer data
            try:
                if "auto_segmentation_4d" in self._viewer.layers:
                    self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
            except Exception:
                pass

        print(f"✅ Auto-segmentation completed for timestep {t}")
        return seg

    def auto_segment_all_timesteps(self, mode: str = "auto", device: str | None = None, tile_shape=None, halo=None, gap_closing: int | None = None, min_z_extent: int | None = None, with_background: bool = True, min_object_size: int = 100, prefer_decoder: bool = True, overwrite: bool = False):
        """Run automatic 3D segmentation for every timestep and store results in auto_segmentation_4d.

        Returns a list of segmentation arrays per timestep.
        """
        results = []
        
        for t in range(self.n_timesteps):
            # skip existing unless overwrite
            if not overwrite and self.auto_segmentation_4d is not None:
                if np.any(self.auto_segmentation_4d[t]):
                    results.append(self.auto_segmentation_4d[t])
                    continue
            seg = self.auto_segment_timestep(t=t, mode=mode, device=device, tile_shape=tile_shape, halo=halo, gap_closing=gap_closing, min_z_extent=min_z_extent, with_background=with_background, min_object_size=min_object_size, prefer_decoder=prefer_decoder)
            results.append(seg)
        return results

    

    def debug_segmentation_4d(self, t: int) -> bool:
        """Debug segmentation setup for a specific timestep.
        
        Returns True if segmentation can proceed, False if fatal error found.
        """
        print(f"\n🔍 Debugging timestep {t}")
        fatal_error = False
        
        # 1. Check embeddings exist
        if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
            print(f"❌ embeddings_4d attribute missing")
            return False
        
        entry = self.embeddings_4d.get(int(t))
        if entry is None:
            print(f"❌ No embeddings for timestep {t}")
            return False
        else:
            # Check if embeddings are loaded (have features)
            if isinstance(entry, dict) and "features" in entry:
                print(f"✔ Embeddings found (materialized)")
            elif isinstance(entry, dict) and "path" in entry:
                print(f"✔ Embeddings found (lazy, path: {entry['path']})")
            else:
                print(f"⚠ Embeddings found but in unexpected format: {type(entry)}")
        
        # Check AnnotatorState has embeddings activated
        try:
            from ._state import AnnotatorState
            state = AnnotatorState()
            if state.image_embeddings is None:
                print(f"❌ Embeddings not activated in AnnotatorState")
                return False
            else:
                print(f"✔ Embeddings activated in AnnotatorState")
                
            # Check predictor exists
            if state.predictor is None:
                print(f"❌ No predictor in AnnotatorState")
                return False
            else:
                print(f"✔ Predictor exists")
        except Exception as e:
            print(f"❌ Failed to check AnnotatorState: {e}")
            return False
        
        # 2. Check point prompts exist
        if not hasattr(self, "point_prompts_4d"):
            print(f"❌ point_prompts_4d attribute missing")
            return False
        
        pts = self.point_prompts_4d.get(int(t))
        if pts is None or (hasattr(pts, 'size') and pts.size == 0):
            print(f"❌ No point prompts in timestep {t}")
            return False
        else:
            print(f"✔ Point prompts found: {len(pts)} points")
            # Show first few points
            if len(pts) > 0:
                print(f"  First point: {pts[0]}")
        
        # 4. Check image dimensions
        if self.image_4d is None:
            print(f"❌ image_4d is None")
            return False
        
        try:
            image3d = self.image_4d[int(t)]
            shape = image3d.shape
            
            if len(shape) == 3:
                print(f"✔ Image dimensions OK: {shape} (Z, Y, X)")
            elif len(shape) == 2:
                print(f"⚠ Image is 2D: {shape} (Y, X) - may need 2D segmentation")
            elif len(shape) == 4:
                print(f"❌ Image has wrong dimensions: {shape} (should be 3D, got 4D)")
                return False
            else:
                print(f"❌ Unexpected image dimensions: {shape}")
                return False
                
            # Check image is not empty
            if image3d.size == 0:
                print(f"❌ Image is empty")
                return False
                
            # Check image has reasonable values
            img_min, img_max = image3d.min(), image3d.max()
            print(f"  Image value range: [{img_min}, {img_max}]")
            if img_min == img_max:
                print(f"⚠ Image has constant value - may produce poor segmentation")
                
        except Exception as e:
            print(f"❌ Failed to check image: {e}")
            return False
        
        # 5. Check current_object_4d exists
        if self.current_object_4d is None:
            print(f"⚠ current_object_4d is None (will be created)")
        else:
            print(f"✔ current_object_4d exists with shape {self.current_object_4d.shape}")
        
        print(f"✅ All checks passed - ready for segmentation\n")
        return True

    def segment_and_propagate_all_timesteps(self):
        """Segment timesteps with point prompts, then propagate across time.
        
        This function:
        1. Detects timesteps with point prompts
        2. Segments each timestep with prompts (3D segmentation)
        3. Propagates the segmentation across all timesteps using mask-based tracking
        4. Stores results in current_object_4d
        """
        if self.image_4d is None or self.n_timesteps is None:
            show_info("No 4D image loaded")
            return

        # Ensure container exists
        if self.current_object_4d is None:
            self.current_object_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                if "current_object_4d" in self._viewer.layers:
                    self._viewer.layers["current_object_4d"].data = self.current_object_4d
                else:
                    self._viewer.add_labels(self.current_object_4d, name="current_object_4d")
            except Exception:
                pass

        # Step 1: Detect timesteps with point prompts
        prompt_map = getattr(self, "point_prompts_4d", None) or {}
        current_t = getattr(self, "current_timestep", 0)
        
        # Check if current timestep has points in the layer
        point_layer = self._viewer.layers["point_prompts"] if "point_prompts" in self._viewer.layers else None
        current_has_points = False
        if point_layer is not None:
            current_points = np.array(point_layer.data)
            current_has_points = len(current_points) > 0
            if current_has_points and current_t not in prompt_map:
                # Save current points to map so they're detected
                prompt_map[current_t] = current_points
        
        prompt_timesteps = sorted([
            t for t in range(self.n_timesteps)
            if t in prompt_map and prompt_map[t] is not None and len(prompt_map[t]) > 0
        ])
        
        print(f"📊 Debug: prompt_map keys: {list(prompt_map.keys())}")
        print(f"📊 Current timestep: {current_t}, has points: {current_has_points}")
        
        if len(prompt_timesteps) == 0:
            show_info("No timesteps with point prompts found. Please add point prompts first.")
            return
        
        print(f"🔄 Found {len(prompt_timesteps)} timesteps with prompts: {prompt_timesteps}")
        
        # CRITICAL: Compute embeddings for ALL timesteps before propagation
        if hasattr(self, 'timestep_embedding_manager') and self.timestep_embedding_manager is not None:
            for t in range(self.n_timesteps):
                try:
                    # Check if embeddings already exist
                    entry = self.embeddings_4d.get(int(t)) if hasattr(self, "embeddings_4d") else None
                    if entry is None:
                        self.timestep_embedding_manager.on_timestep_changed(int(t))
                except Exception:
                    pass
        
        print(f"\n")
        
        # Step 2: Segment each timestep with prompts (using existing segment_all_timesteps logic)
        # We'll call the existing segmentation, but only for prompt timesteps
        original_t = current_t
        # point_layer already retrieved above
        
        # Disconnect callbacks during batch processing
        dims_callback_disconnected = False
        try:
            self._viewer.dims.events.current_step.disconnect(self._on_dims_current_step)
            dims_callback_disconnected = True
        except Exception:
            pass
        
        if point_layer is not None and hasattr(point_layer, 'events'):
            try:
                if hasattr(self, '_point_prompt_connection'):
                    point_layer.events.data.disconnect(self._point_prompt_connection)
            except Exception:
                pass
        
        # Segment only timesteps with prompts
        segmented_timesteps = {}  # Maps timestep -> list of object_ids
        
        for t in prompt_timesteps:
            print(f"\n📍 Segmenting timestep {t}...")
            
            # Get points for this timestep
            if t == original_t and point_layer is not None:
                pts = np.array(point_layer.data)
                print(f"  Using {len(pts)} points from current layer")
            else:
                pts = prompt_map.get(t, np.empty((0, 3)))
                print(f"  Using {len(pts)} points from stored map")
            
            if len(pts) == 0:
                print(f"  ⏭️  Skipping - no points found")
                continue
            
            # Activate embeddings for this timestep
            try:
                self._ensure_embeddings_active_for_t(int(t))
                from ._state import AnnotatorState
                state = AnnotatorState()
                
                if state.image_embeddings is None:
                    print(f"⚠ Embeddings not available for timestep {t}, skipping")
                    continue
                
                # Set image metadata
                if state.image_shape is None:
                    image3d = self.image_4d[int(t)]
                    state.image_shape = tuple(image3d.shape)
            except Exception as e:
                print(f"❌ Failed to activate embeddings for timestep {t}: {e}")
                continue
            
            # Set point prompts
            try:
                if point_layer is not None:
                    if hasattr(point_layer, 'events') and hasattr(point_layer.events, 'data'):
                        with point_layer.events.data.blocker():
                            point_layer.data = pts
                    else:
                        point_layer.data = pts
            except Exception:
                pass
            
            # Perform segmentation (similar to segment_all_timesteps)
            try:
                print(f"  Starting segmentation logic...")
                from . import util as sam_util
                from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
                
                image3d = self.image_4d[int(t)]
                shape = image3d.shape
                seg_merged = np.zeros(shape, dtype=np.uint32)
                
                print(f"  Image shape: {shape}")
                
                # Get point IDs
                point_ids = self._get_point_ids_for_timestep(t, pts)
                print(f"  Point IDs: {point_ids}")
                
                # Group points by ID
                points_by_id = {}
                for point_idx, point in enumerate(pts):
                    point_id = point_ids[point_idx] if point_idx < len(point_ids) else 1
                    if point_id not in points_by_id:
                        points_by_id[point_id] = []
                    points_by_id[point_id].append((point_idx, point))
                
                # Segment each ID group
                object_ids = []
                for target_id, points_list in points_by_id.items():
                    # Collect negative prompts from other IDs
                    negative_points_by_z = {}
                    for other_id, other_points_list in points_by_id.items():
                        if other_id != target_id:
                            for _, other_point in other_points_list:
                                z = int(other_point[0])
                                if z not in negative_points_by_z:
                                    negative_points_by_z[z] = []
                                negative_points_by_z[z].append(other_point[1:3])
                    
                    # Group points by Z
                    points_by_z = {}
                    for point_idx, point in points_list:
                        z = int(point[0])
                        if z not in points_by_z:
                            points_by_z[z] = []
                        points_by_z[z].append(point[1:3])
                    
                    # Segment and extend
                    id_seg = np.zeros(shape, dtype=np.uint32)
                    shape_2d = shape[1:]
                    
                    for z_slice, pts_2d_list in points_by_z.items():
                        pts_2d = np.array(pts_2d_list)
                        pos_labels = np.ones(len(pts_2d), dtype=int)
                        
                        neg_pts_at_z = negative_points_by_z.get(z_slice, [])
                        if len(neg_pts_at_z) > 0:
                            neg_pts = np.array(neg_pts_at_z)
                            all_points = np.vstack([pts_2d, neg_pts])
                            all_labels = np.concatenate([pos_labels, np.zeros(len(neg_pts), dtype=int)])
                        else:
                            all_points = pts_2d
                            all_labels = pos_labels
                        
                        seg_2d = sam_util.prompt_segmentation(
                            state.predictor, all_points, all_labels,
                            boxes=np.array([]), masks=None, shape=shape_2d,
                            multiple_box_prompts=False,
                            image_embeddings=state.image_embeddings, i=z_slice,
                        )
                        
                        if seg_2d is not None and seg_2d.max() > 0:
                            seg_3d = np.zeros(shape, dtype=np.uint32)
                            seg_3d[z_slice] = (seg_2d > 0).astype(np.uint32)
                            
                            try:
                                seg_3d, _ = segment_mask_in_volume(
                                    seg_3d, state.predictor, state.image_embeddings,
                                    np.array([z_slice]), stop_lower=False, stop_upper=False,
                                    iou_threshold=0.65, projection="single_point", verbose=False,
                                )
                            except Exception:
                                pass
                            
                            mask = seg_3d > 0
                            id_seg[mask] = 1
                    
                    if id_seg.max() > 0:
                        mask = id_seg > 0
                        seg_merged[mask] = target_id
                        object_ids.append(target_id)
                        print(f"  ✓ Segmented object with ID {target_id}")
                
                if seg_merged.max() > 0:
                    self.current_object_4d[int(t)] = seg_merged
                    segmented_timesteps[t] = object_ids
                    print(f"✅ Segmented timestep {t} with {len(object_ids)} objects")
                    
            except Exception as e:
                print(f"❌ Segmentation failed for timestep {t}: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Propagate across time for each object
        if len(segmented_timesteps) == 0:
            show_info("No objects were successfully segmented")
            if dims_callback_disconnected:
                try:
                    self._viewer.dims.events.current_step.connect(self._on_dims_current_step)
                except Exception:
                    pass
            return
        
        print(f"\n🔄 Starting temporal propagation...")
        
        # For each unique object ID across all segmented timesteps
        all_object_ids = set()
        for obj_ids in segmented_timesteps.values():
            all_object_ids.update(obj_ids)
        
        for obj_id in sorted(all_object_ids):
            print(f"\n📦 Propagating object ID {obj_id}...")
            
            # Find timesteps where this object exists
            obj_timesteps = sorted([t for t, ids in segmented_timesteps.items() if obj_id in ids])
            
            if len(obj_timesteps) == 0:
                continue
            
            print(f"  Object exists at timesteps: {obj_timesteps}")
            
            # Debug: show mask info for each anchor
            for t_anchor in obj_timesteps:
                mask_anchor = (self.current_object_4d[t_anchor] == obj_id)
                z_slices_with_mask = np.where(mask_anchor.any(axis=(1, 2)))[0]
                total_pixels = np.count_nonzero(mask_anchor)
                print(f"    Anchor t={t_anchor}: {len(z_slices_with_mask)} z-slices, {total_pixels} total pixels")
            
            # Propagate from each anchor point in both directions
            for i, t_anchor in enumerate(obj_timesteps):
                mask_anchor = (self.current_object_4d[t_anchor] == obj_id)
                
                # Propagate backward from this anchor
                if i == 0:
                    # First anchor: propagate backward to start
                    if t_anchor > 0:
                        print(f"    Propagating backward from anchor t={t_anchor} to t=0")
                        self._propagate_object_backward(obj_id, t_anchor, 0, mask_anchor)
                else:
                    # Not first anchor: propagate backward to previous anchor
                    t_prev_anchor = obj_timesteps[i - 1]
                    if t_anchor > t_prev_anchor + 1:
                        print(f"    Propagating backward from anchor t={t_anchor} to t={t_prev_anchor + 1}")
                        self._propagate_object_backward(obj_id, t_anchor, t_prev_anchor + 1, mask_anchor)
                
                # Propagate forward from this anchor
                if i == len(obj_timesteps) - 1:
                    # Last anchor: propagate forward to end
                    if t_anchor < self.n_timesteps - 1:
                        print(f"    Propagating forward from anchor t={t_anchor} to t={self.n_timesteps - 1}")
                        self._propagate_object_forward(obj_id, t_anchor, self.n_timesteps, mask_anchor)
                else:
                    # Not last anchor: propagate forward to next anchor
                    t_next_anchor = obj_timesteps[i + 1]
                    if t_anchor < t_next_anchor - 1:
                        print(f"    Propagating forward from anchor t={t_anchor} to t={t_next_anchor - 1}")
                        self._propagate_object_forward(obj_id, t_anchor, t_next_anchor, mask_anchor)
        
        # Reconnect callbacks
        if dims_callback_disconnected:
            try:
                self._viewer.dims.events.current_step.connect(self._on_dims_current_step)
            except Exception:
                pass
        
        if point_layer is not None and hasattr(point_layer, 'events'):
            try:
                if hasattr(self, '_point_prompt_connection'):
                    point_layer.events.data.connect(self._point_prompt_connection)
            except Exception:
                pass
        
        # Restore original timestep
        try:
            self._viewer.dims.set_current_step(0, int(original_t))
        except Exception:
            pass
        
        # Refresh layer
        try:
            if "current_object_4d" in self._viewer.layers:
                self._viewer.layers["current_object_4d"].refresh()
        except Exception:
            pass
        
        # Remove all point prompts after successful propagation
        try:
            if "point_prompts" in self._viewer.layers:
                point_layer = self._viewer.layers["point_prompts"]
                point_layer.data = np.empty((0, 3))
                print(f"✅ Cleared all point prompts after propagation")
            
            # Clear stored point prompts from all timesteps
            if hasattr(self, 'point_prompts_4d'):
                for t in range(self.n_timesteps):
                    if t in self.point_prompts_4d:
                        self.point_prompts_4d[t] = np.empty((0, 3))
            
            # Clear point ID map
            if hasattr(self, 'point_id_map'):
                self.point_id_map.clear()
            
            # Refresh point manager widget if it exists
            if hasattr(self, '_point_manager_widget'):
                try:
                    self._point_manager_widget.refresh_point_list()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠ Failed to clear point prompts: {e}")
        
        show_info("Segmentation and propagation complete!")
        print(f"✅ Completed propagation for {len(all_object_ids)} objects")

    def _extract_tyx_embeddings_for_z(self, z_slice):
        """Extract embeddings for a specific Z-slice across all timesteps, organized as TYX.
        
        This reorganizes the per-timestep ZYX embedding structure into a TYX structure
        where time becomes the first dimension, suitable for using segment_mask_in_volume
        across the temporal dimension.
        
        Args:
            z_slice: The Z-slice index to extract across all timesteps
            
        Returns:
            ImageEmbeddings dict with 'features' as (T, C, H, W) array where T is time,
            or None if extraction fails
        """
        print(f"      Creating TYX embeddings for Z={z_slice}...")
        
        # Collect embeddings for this z-slice across all timesteps
        tyx_features_list = []
        
        for t in range(self.n_timesteps):
            # Get embeddings for this timestep
            entry = self.embeddings_4d.get(int(t))
            if entry is None or not isinstance(entry, dict):
                print(f"        ⚠ No embeddings for t={t}")
                return None
            
            # Materialize if lazy
            if "path" in entry and "features" not in entry:
                try:
                    self._load_embedding_for_timestep(int(t))
                    entry = self.embeddings_4d.get(int(t))
                except Exception as e:
                    print(f"        ⚠ Failed to load embeddings for t={t}: {e}")
                    return None
            
            if "features" not in entry:
                print(f"        ⚠ No features in embeddings for t={t}")
                return None
            
            features = entry["features"]
            
            # Convert to numpy array if it's a zarr array or other lazy type
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Extract the z-slice from this timestep's embeddings
            # Features shape can be:
            # - (Z, C, H, W) for 3D embeddings
            # - (Z, 1, C, H, W) for 3D embeddings with batch dimension
            # - (C, H, W) for single slice
            try:
                if len(features.shape) == 5:
                    # (Z, 1, C, H, W) - has batch dimension, squeeze it
                    print(f"        t={t}: shape {features.shape}, squeezing batch dimension")
                    features = features.squeeze(1)  # Now (Z, C, H, W)
                    if z_slice < features.shape[0]:
                        z_features = np.array(features[z_slice])  # (C, H, W)
                    else:
                        print(f"        ⚠ Z={z_slice} out of bounds for t={t} (max: {features.shape[0]-1})")
                        return None
                elif len(features.shape) == 4:
                    # (Z, C, H, W) - extract specific z-slice
                    if z_slice < features.shape[0]:
                        z_features = np.array(features[z_slice])  # (C, H, W)
                    else:
                        print(f"        ⚠ Z={z_slice} out of bounds for t={t} (max: {features.shape[0]-1})")
                        return None
                elif len(features.shape) == 3:
                    # (C, H, W) - single slice, use if z_slice == 0
                    if z_slice == 0:
                        z_features = np.array(features)
                    else:
                        print(f"        ⚠ Single-slice embedding at t={t}, but z_slice={z_slice}")
                        return None
                else:
                    print(f"        ⚠ Unexpected embedding shape at t={t}: {features.shape}")
                    return None
                
                tyx_features_list.append(z_features)
            except Exception as e:
                print(f"        ⚠ Error extracting Z={z_slice} from t={t}: {e}")
                return None
        
        # Stack into (T, C, H, W)
        try:
            tyx_features = np.stack(tyx_features_list, axis=0)
        except Exception as e:
            print(f"        ⚠ Failed to stack features: {e}")
            return None
        
        # Add a batch dimension to make it 5D: (T, 1, C, H, W)
        # This is required because segment_mask_in_volume expects 5D embeddings
        # to use indexing (treating T as Z-slices)
        tyx_features = tyx_features[:, np.newaxis, :, :, :]
        
        # Create ImageEmbeddings dict
        # Use the input_size and original_size from the first timestep as reference
        first_entry = self.embeddings_4d.get(0)
        input_size = first_entry.get("input_size") if isinstance(first_entry, dict) else None
        original_size = first_entry.get("original_size") if isinstance(first_entry, dict) else None
        
        tyx_embeddings = {
            "features": tyx_features,
            "input_size": input_size,
            "original_size": original_size,
        }
        
        print(f"      ✓ Created TYX embeddings shape {tyx_features.shape}")
        return tyx_embeddings

    def _propagate_object_forward(self, obj_id, t_start, t_end, mask_start):
        """Propagate an object mask forward in time using volumetric segmentation.
        
        For each z-slice where the object exists, create a TYX volume (treating time as Z)
        and use segment_mask_in_volume to propagate across the temporal dimension.
        This is the approach recommended by your mentor.
        """
        from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
        from ._state import AnnotatorState
        
        print(f"  Propagating forward from t={t_start} to t={t_end} (exclusive)")
        print(f"  Total timesteps in dataset: {self.n_timesteps}")
        
        # Find Z slices with mask at anchor
        z_indices = np.where(mask_start.any(axis=(1, 2)))[0]
        if len(z_indices) == 0:
            print(f"    No z-slices with mask at anchor")
            return
        
        z_min, z_max = int(z_indices[0]), int(z_indices[-1])
        print(f"    Initial Z-range: {z_min}-{z_max} (will expand dynamically if object moves in Z)")
        
        # Get state for predictor
        state = AnnotatorState()
        
        # Track which Z-slices have been processed
        processed_z_slices = set()
        
        # Iteratively process Z-slices, expanding range as object moves in Z
        iteration = 0
        max_iterations = 20  # Safety limit to prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            print(f"    Iteration {iteration}: Processing Z-range {z_min}-{z_max}")
            
            # Collect Z-slices to process in this iteration
            z_slices_to_process = []
            for z in range(z_min, z_max + 1):
                if z not in processed_z_slices:
                    z_slices_to_process.append(z)
            
            if not z_slices_to_process:
                print(f"    All Z-slices processed, propagation complete")
                break
            
            # Process each new Z-slice
            for z in z_slices_to_process:
                # Check if this z-slice has mask at anchor or was populated in previous iteration
                mask_2d_anchor = mask_start[z] if z < mask_start.shape[0] else np.zeros(mask_start.shape[1:], dtype=bool)
                
                has_mask = mask_2d_anchor.any()
                if not has_mask:
                    # Check if previous iteration added this object to this z-slice at any timestep
                    for t in range(t_start, min(t_end + 1, self.n_timesteps)):
                        if z < self.current_object_4d[t].shape[0] and (self.current_object_4d[t][z] == obj_id).any():
                            has_mask = True
                            mask_2d_anchor = (self.current_object_4d[t][z] == obj_id)
                            break
                
                if not has_mask:
                    processed_z_slices.add(z)
                    continue
                
                print(f"      Z={z}: Propagating using segment_mask_in_volume...")
                
                try:
                    # Extract TYX embeddings for this z-slice
                    tyx_embeddings = self._extract_tyx_embeddings_for_z(z)
                    if tyx_embeddings is None:
                        print(f"      Z={z}: Failed to create TYX embeddings, skipping")
                        processed_z_slices.add(z)
                        continue
                    
                    # Create TYX segmentation volume (T, Y, X)
                    # Initialize with zeros for all timesteps
                    image_shape = self.image_4d[0].shape  # (Z, Y, X)
                    y_size, x_size = image_shape[1], image_shape[2]
                    tyx_seg = np.zeros((self.n_timesteps, y_size, x_size), dtype=np.uint32)
                    
                    # Set the anchor mask at t_start
                    tyx_seg[t_start] = mask_2d_anchor.astype(np.uint32)
                    
                    # Check for other anchor timesteps with this object in this z-slice
                    anchor_timesteps = [t_start]
                    for t_check in range(self.n_timesteps):
                        if t_check != t_start and (self.current_object_4d[t_check][z] == obj_id).any():
                            tyx_seg[t_check] = (self.current_object_4d[t_check][z] == obj_id).astype(np.uint32)
                            anchor_timesteps.append(t_check)
                    
                    # Use segment_mask_in_volume to propagate across time dimension
                    # The "segmented_slices" parameter is the anchor timestep(s)
                    # Use box projection + lower IOU threshold to handle spatial shifts (worm movement)
                    print(f"      Z={z}: Running segment_mask_in_volume with anchors at t={anchor_timesteps}...")
                    tyx_seg_result, (t_min, t_max) = segment_mask_in_volume(
                        segmentation=tyx_seg,
                        predictor=state.predictor,
                        image_embeddings=tyx_embeddings,
                        segmented_slices=np.array(anchor_timesteps),
                        stop_lower=False,  # Propagate backward in time
                        stop_upper=False,  # Propagate forward in time
                        iou_threshold=0.3,  # Lower threshold to handle spatial shifts
                        projection="box",  # Box projection allows spatial movement
                        verbose=True,  # Show where propagation stops
                    )
                    
                    print(f"      Z={z}: Propagated from t={t_min} to t={t_max}")
                    
                    # Check coverage and retry with even lower threshold if needed
                    expected_range = min(t_end, self.n_timesteps) - t_start
                    actual_range = t_max - t_min + 1
                    coverage = actual_range / expected_range if expected_range > 0 else 0
                    
                    if coverage < 0.8:  # Less than 80% coverage
                        print(f"      Z={z}: Low coverage ({coverage:.1%}). Retrying with IOU=0.2...")
                        tyx_seg_retry = tyx_seg.copy()
                        try:
                            tyx_seg_result, (t_min, t_max) = segment_mask_in_volume(
                                segmentation=tyx_seg_retry,
                                predictor=state.predictor,
                                image_embeddings=tyx_embeddings,
                                segmented_slices=np.array(anchor_timesteps),
                                stop_lower=False,
                                stop_upper=False,
                                iou_threshold=0.2,  # Very forgiving for large movements
                                projection="box",
                                verbose=True,
                            )
                            print(f"      Z={z}: Retry succeeded - propagated from t={t_min} to t={t_max}")
                        except Exception as e:
                            print(f"      Z={z}: Retry failed - {e}")
                    
                    # Write back to 4D volume: current_object_4d[:, z, :, :]
                    # Only write for timesteps between t_start and t_end
                    print(f"      Z={z}: Writing back results for timesteps {max(t_start, t_min)} to {min(t_end, t_max + 1) - 1}")
                    
                    # Track last valid mask for fallback
                    last_valid_mask = None
                    
                    for t in range(max(t_start, t_min), min(t_end, t_max + 1)):
                        if tyx_seg_result[t].any():
                            # Only write where we have new segmentation
                            mask_t = tyx_seg_result[t] > 0
                            pixels_count = np.count_nonzero(mask_t)
                            # Preserve this object's segmentation
                            self.current_object_4d[t][z][mask_t] = obj_id
                            print(f"      Z={z}: Wrote {pixels_count} pixels for t={t}")
                            # Store this as last valid mask
                            last_valid_mask = mask_t.copy()
                        else:
                            # No segmentation at this timestep - use fallback from previous timestep
                            if last_valid_mask is not None:
                                # Copy the previous timestep's mask
                                self.current_object_4d[t][z][last_valid_mask] = obj_id
                                pixels_count = np.count_nonzero(last_valid_mask)
                                print(f"      Z={z}: No mask at t={t}, copied {pixels_count} pixels from t={t-1} (fallback)")
                            else:
                                print(f"      Z={z}: No pixels for t={t} (no fallback available)")
                    
                    print(f"      Z={z}: Complete")
                    processed_z_slices.add(z)
                    
                except Exception as e:
                    print(f"      Z={z}: Error during propagation - {e}")
                    import traceback
                    traceback.print_exc()
                    processed_z_slices.add(z)
                    continue
            
            # After processing all Z-slices in this iteration, check if object expanded to new Z-slices
            new_z_min, new_z_max = z_min, z_max
            for t in range(t_start, min(t_end + 1, self.n_timesteps)):
                z_indices_t = np.where((self.current_object_4d[t] == obj_id).any(axis=(1, 2)))[0]
                if len(z_indices_t) > 0:
                    new_z_min = min(new_z_min, int(z_indices_t[0]))
                    new_z_max = max(new_z_max, int(z_indices_t[-1]))
            
            if new_z_min < z_min or new_z_max > z_max:
                print(f"    Object expanded in Z: {z_min}-{z_max} → {new_z_min}-{new_z_max}")
                z_min, z_max = new_z_min, new_z_max
            else:
                # No expansion, we're done
                print(f"    No Z expansion detected, propagation complete")
                break
        
        if iteration >= max_iterations:
            print(f"    Warning: Reached maximum iterations ({max_iterations}), stopping propagation")
        
        print(f"    Forward propagation complete")

    def _propagate_object_backward(self, obj_id, t_start, t_end, mask_start):
        """Propagate an object mask backward in time using volumetric segmentation.
        
        Since segment_mask_in_volume propagates in both directions from anchors,
        backward propagation uses the same volumetric approach as forward.
        """
        from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
        from ._state import AnnotatorState
        
        print(f"  Propagating backward from t={t_start} to t={t_end}")
        
        # Find Z slices with mask at anchor
        z_indices = np.where(mask_start.any(axis=(1, 2)))[0]
        if len(z_indices) == 0:
            print(f"    No z-slices with mask at anchor")
            return
        
        z_min, z_max = int(z_indices[0]), int(z_indices[-1])
        print(f"    Initial Z-range: {z_min}-{z_max} (will expand dynamically if object moves in Z)")
        
        # Get state for predictor
        state = AnnotatorState()
        
        # Track which Z-slices have been processed
        processed_z_slices = set()
        
        # Iteratively process Z-slices, expanding range as object moves in Z
        iteration = 0
        max_iterations = 20  # Safety limit
        
        while iteration < max_iterations:
            iteration += 1
            print(f"    Iteration {iteration}: Processing Z-range {z_min}-{z_max}")
            
            # Collect Z-slices to process in this iteration
            z_slices_to_process = []
            for z in range(z_min, z_max + 1):
                if z not in processed_z_slices:
                    z_slices_to_process.append(z)
            
            if not z_slices_to_process:
                print(f"    All Z-slices processed, propagation complete")
                break
            
            # For each z-slice, propagate across time using volumetric segmentation
            for z in z_slices_to_process:
                mask_2d_anchor = mask_start[z] if z < mask_start.shape[0] else np.zeros(mask_start.shape[1:], dtype=bool)
                
                has_mask = mask_2d_anchor.any()
                if not has_mask:
                    # Check if previous iteration added this object to this z-slice
                    for t in range(t_end, min(t_start + 1, self.n_timesteps)):
                        if z < self.current_object_4d[t].shape[0] and (self.current_object_4d[t][z] == obj_id).any():
                            has_mask = True
                            mask_2d_anchor = (self.current_object_4d[t][z] == obj_id)
                            break
                
                if not has_mask:
                    processed_z_slices.add(z)
                    continue
                
                print(f"      Z={z}: Propagating using segment_mask_in_volume...")
                
                try:
                    # Extract TYX embeddings for this z-slice
                    tyx_embeddings = self._extract_tyx_embeddings_for_z(z)
                    if tyx_embeddings is None:
                        print(f"      Z={z}: Failed to create TYX embeddings, skipping")
                        processed_z_slices.add(z)
                        continue
                    
                    # Create TYX segmentation volume (T, Y, X)
                    image_shape = self.image_4d[0].shape  # (Z, Y, X)
                    y_size, x_size = image_shape[1], image_shape[2]
                    tyx_seg = np.zeros((self.n_timesteps, y_size, x_size), dtype=np.uint32)
                    
                    # Set the anchor mask at t_start
                    tyx_seg[t_start] = mask_2d_anchor.astype(np.uint32)
                    
                    # Check for other anchor timesteps
                    anchor_timesteps = [t_start]
                    for t_check in range(self.n_timesteps):
                        if t_check != t_start and (self.current_object_4d[t_check][z] == obj_id).any():
                            tyx_seg[t_check] = (self.current_object_4d[t_check][z] == obj_id).astype(np.uint32)
                            anchor_timesteps.append(t_check)
                    
                    # Use segment_mask_in_volume
                    # Use box projection + lower IOU threshold to handle spatial shifts (worm movement)
                    print(f"      Z={z}: Running segment_mask_in_volume with anchors at t={anchor_timesteps}...")
                    tyx_seg_result, (t_min, t_max) = segment_mask_in_volume(
                        segmentation=tyx_seg,
                        predictor=state.predictor,
                        image_embeddings=tyx_embeddings,
                        segmented_slices=np.array(anchor_timesteps),
                        stop_lower=False,
                        stop_upper=True,  # Don't propagate forward (already done)
                        iou_threshold=0.3,  # Lower threshold to handle spatial shifts
                        projection="box",  # Box projection allows spatial movement
                        verbose=True,  # Show where propagation stops
                    )
                    
                    print(f"      Z={z}: Propagated from t={t_min} to t={t_max}")
                    
                    # Check coverage and retry with even lower threshold if needed
                    expected_range = t_start - t_end
                    actual_range = t_max - t_min + 1
                    coverage = actual_range / expected_range if expected_range > 0 else 0
                    
                    if coverage < 0.8:  # Less than 80% coverage
                        print(f"      Z={z}: Low coverage ({coverage:.1%}). Retrying with IOU=0.2...")
                        tyx_seg_retry = tyx_seg.copy()
                        try:
                            tyx_seg_result, (t_min, t_max) = segment_mask_in_volume(
                                segmentation=tyx_seg_retry,
                                predictor=state.predictor,
                                image_embeddings=tyx_embeddings,
                                segmented_slices=np.array(anchor_timesteps),
                                stop_lower=False,
                                stop_upper=True,
                                iou_threshold=0.2,  # Very forgiving for large movements
                                projection="box",
                                verbose=True,
                            )
                            print(f"      Z={z}: Retry succeeded - propagated from t={t_min} to t={t_max}")
                        except Exception as e:
                            print(f"      Z={z}: Retry failed - {e}")
                    
                    # Write back to 4D volume for timesteps between t_end and t_start
                    # Track last valid mask for fallback (going backward)
                    last_valid_mask = None
                    
                    # Iterate backward from t_start to t_end
                    for t in range(min(t_start, t_max), max(t_end - 1, t_min - 1), -1):
                        if tyx_seg_result[t].any():
                            mask_t = tyx_seg_result[t] > 0
                            self.current_object_4d[t][z][mask_t] = obj_id
                            pixels_count = np.count_nonzero(mask_t)
                            print(f"      Z={z}: Wrote {pixels_count} pixels for t={t}")
                            # Store this as last valid mask
                            last_valid_mask = mask_t.copy()
                        else:
                            # No segmentation at this timestep - use fallback from next timestep (t+1 when going backward)
                            if last_valid_mask is not None:
                                # Copy the next timestep's mask (we're going backward)
                                self.current_object_4d[t][z][last_valid_mask] = obj_id
                                pixels_count = np.count_nonzero(last_valid_mask)
                                print(f"      Z={z}: No mask at t={t}, copied {pixels_count} pixels from t={t+1} (fallback)")
                            else:
                                print(f"      Z={z}: No pixels for t={t} (no fallback available)")
                    
                    print(f"      Z={z}: Complete")
                    processed_z_slices.add(z)
                    
                except Exception as e:
                    print(f"      Z={z}: Error during propagation - {e}")
                    import traceback
                    traceback.print_exc()
                    processed_z_slices.add(z)
                    continue
            
            # After processing all Z-slices, check if object expanded to new Z-slices
            new_z_min, new_z_max = z_min, z_max
            for t in range(t_end, min(t_start + 1, self.n_timesteps)):
                z_indices_t = np.where((self.current_object_4d[t] == obj_id).any(axis=(1, 2)))[0]
                if len(z_indices_t) > 0:
                    new_z_min = min(new_z_min, int(z_indices_t[0]))
                    new_z_max = max(new_z_max, int(z_indices_t[-1]))
            
            if new_z_min < z_min or new_z_max > z_max:
                print(f"    Object expanded in Z: {z_min}-{z_max} → {new_z_min}-{new_z_max}")
                z_min, z_max = new_z_min, new_z_max
            else:
                # No expansion, we're done
                print(f"    No Z expansion detected, propagation complete")
                break
        
        if iteration >= max_iterations:
            print(f"    Warning: Reached maximum iterations ({max_iterations}), stopping propagation")
        
        print(f"    Backward propagation complete")

    def segment_all_timesteps(self):
        """Run manual segmentation for all timesteps that have point prompts.

        Reuses the same routine as the manual single-timestep/volume segmentation and writes
        each result into `self.current_object_4d[t]` (T, Z, Y, X).
        """
        if self.image_4d is None or self.n_timesteps is None:
            return

        # Ensure container and 4D layer exist
        if self.current_object_4d is None:
            self.current_object_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                if "current_object_4d" in self._viewer.layers:
                    self._viewer.layers["current_object_4d"].data = self.current_object_4d
                else:
                    self._viewer.add_labels(self.current_object_4d, name="current_object_4d")
            except Exception:
                pass

        # Mapping of per-timestep prompts
        prompt_map = getattr(self, "point_prompts_4d", None) or {}

        # Get point prompts layer reference
        point_layer = self._viewer.layers["point_prompts"] if "point_prompts" in self._viewer.layers else None

        # Remember original timestep to restore at the end
        original_t = getattr(self, "current_timestep", 0)
        original_pts = prompt_map.get(original_t, np.empty((0, 3)))
        
        print(f"🔄 Starting batch segmentation. Current timestep: {original_t}")

        # Disconnect point prompts event listener during batch segmentation to prevent interference
        if point_layer is not None and hasattr(point_layer, 'events') and hasattr(point_layer.events, 'data'):
            try:
                if hasattr(self, '_point_prompt_connection'):
                    point_layer.events.data.disconnect(self._point_prompt_connection)
            except Exception:
                pass

        # Access volumetric segmentation widget if available
        try:
            state = AnnotatorState()
            seg_widget = state.widgets.get("segment_nd") if getattr(state, "widgets", None) else None
        except Exception:
            seg_widget = None

        # Disconnect dimension slider callback to prevent timestep change interference
        dims_callback_disconnected = False
        try:
            self._viewer.dims.events.current_step.disconnect(self._on_dims_current_step)
            dims_callback_disconnected = True
        except Exception:
            pass

        for t in range(int(self.n_timesteps)):
            # For current timestep, get points from the layer (may have unsaved points)
            # For other timesteps, get from stored map
            if t == original_t and point_layer is not None:
                pts = np.array(point_layer.data)
            else:
                pts = prompt_map.get(t, np.empty((0, 3)))
            
            pts_arr = np.asarray(pts) if pts is not None else np.empty((0, 3))
            if pts_arr.size == 0:
                continue

            # CRITICAL: Save current points to prompt_map so debug_segmentation_4d can see them
            # This ensures newly placed points are recognized before debug checks
            self.point_prompts_4d[t] = pts_arr
            
            # DON'T update self.current_timestep to avoid triggering callbacks
            # Just work with 't' directly for embedding activation

            # Run debug checks before attempting segmentation
            if not self.debug_segmentation_4d(t):
                continue

            # Activate embeddings/predictor for this timestep
            # For batch segmentation, we need to ensure embeddings are fully loaded (not async)
            try:
                # Check if we have a lazy entry that needs loading
                entry = self.embeddings_4d.get(int(t)) if hasattr(self, "embeddings_4d") else None
                if entry is not None and isinstance(entry, dict) and "path" in entry and "features" not in entry:
                    # Load synchronously during batch processing
                    try:
                        self._load_embedding_for_timestep(int(t))
                        # Give a moment for the embedding to be activated
                        import time
                        time.sleep(0.1)
                    except Exception:
                        continue
                
                # Now ensure embeddings are active
                # Skip embedding manager for current timestep as it's already active
                if getattr(self, "timestep_embedding_manager", None) is not None and t != original_t:
                    try:
                        self.timestep_embedding_manager.on_timestep_changed(int(t))
                    except Exception:
                        pass
                
                # Ensure embeddings are active for this timestep
                self._ensure_embeddings_active_for_t(int(t))
                
                # Verify embeddings are actually loaded in AnnotatorState
                from ._state import AnnotatorState
                state = AnnotatorState()
                if state.image_embeddings is None:
                    continue
                    
                # Ensure image metadata is set for segmentation
                if state.image_shape is None:
                    try:
                        image3d = self.image_4d[int(t)]
                        state.image_shape = tuple(image3d.shape)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"Failed to activate embeddings for timestep {t}: {e}")
                continue

            # Manually set point prompts for this timestep without triggering callbacks
            try:
                if point_layer is not None:
                    # Block events when updating data to prevent callback interference
                    if hasattr(point_layer, 'events') and hasattr(point_layer.events, 'data'):
                        with point_layer.events.data.blocker():
                            point_layer.data = pts_arr
                    else:
                        point_layer.data = pts_arr
                elif "point_prompts" not in self._viewer.layers:
                    point_layer = self._viewer.add_points(pts_arr, name="point_prompts", size=10,
                                                          face_color="green", edge_color="green",
                                                          edge_width=2, blending="translucent")
                    # Set up positive/negative color cycle (green=positive, red=negative)
                    point_layer.face_color_cycle = ["green", "red"]
                    point_layer.edge_color_cycle = ["green", "red"]
                    
                    # Set up features for point type
                    if len(pts_arr) > 0:
                        point_layer.features = {
                            'point_type': np.zeros(len(pts_arr), dtype=int),  # 0 = positive/green by default
                            'label': np.array(['0'] * len(pts_arr))  # String labels for compatibility
                        }
                        point_layer.face_color = 'point_type'
                        point_layer.edge_color = 'point_type'
                        # Default new points to positive (only set when features exist)
                        point_layer.feature_defaults = {
                            'point_type': 0,
                            'label': '0'
                        }
                    else:
                        # Don't set features or feature_defaults for empty layer
                        pass
                    
                    # Setup cursor handling for point prompts layer
                    try:
                        canvas = self._viewer.window.qt_viewer.canvas.native
                        
                        def on_mouse_enter(event):
                            """Set crosshair cursor when entering viewer in add mode"""
                            try:
                                if point_layer.mode == 'add':
                                    canvas.setCursor(Qt.CrossCursor)
                            except Exception:
                                pass
                        
                        def on_mouse_leave(event):
                            """Reset cursor to normal when leaving the viewer"""
                            try:
                                canvas.setCursor(Qt.ArrowCursor)
                            except Exception:
                                pass
                        
                        # Connect to canvas enter/leave events
                        try:
                            original_enter = canvas.enterEvent
                            original_leave = canvas.leaveEvent
                            
                            def new_enter_event(event):
                                try:
                                    on_mouse_enter(event)
                                    if callable(original_enter):
                                        return original_enter(event)
                                except Exception:
                                    pass
                            
                            def new_leave_event(event):
                                try:
                                    on_mouse_leave(event)
                                    if callable(original_leave):
                                        return original_leave(event)
                                except Exception:
                                    pass
                            
                            canvas.enterEvent = new_enter_event
                            canvas.leaveEvent = new_leave_event
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

            # Run manual volumetric segmentation using Shift+S logic
            try:
                # Use segment_slices_with_prompts + segment_mask_in_volume like Shift+S
                from . import util as sam_util
                from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
                from ._state import AnnotatorState
                
                # Get the 3D image for this timestep
                image3d = self.image_4d[int(t)]
                state = AnnotatorState()
                shape = image3d.shape
                
                # Set state.image_shape if needed
                if state.image_shape is None or state.image_shape != shape:
                    state.image_shape = shape
                
                # Initialize merged segmentation
                seg_merged = np.zeros(shape, dtype=np.uint32)
                
                # SIMPLIFIED: All points are for ONE object (standard micro-sam workflow)
                # No multiple IDs - just segment all points together
                if len(pts_arr) > 0:
                    # Get box prompts layer (usually empty for point-based workflow)
                    box_layer = self._viewer.layers["prompts"] if "prompts" in self._viewer.layers else None
                    
                    # Get point labels from features and convert to SAM format
                    # Our point_type: 0=positive/green, 1=negative/red
                    # SAM expects: 1=positive, 0=negative
                    point_labels = np.ones(len(pts_arr), dtype=int)  # Default to positive
                    if point_layer is not None and hasattr(point_layer, 'features'):
                        if 'point_type' in point_layer.features:
                            # Convert: 0->1 (positive), 1->0 (negative)
                            point_labels = 1 - point_layer.features['point_type'].values
                    
                    print(f"Segmenting object at timestep {t} ({np.sum(point_labels==1)} positive, {np.sum(point_labels==0)} negative)", end="", flush=True)
                    
                    # All points go into one segmentation
                    # Group points by Z-slice with their labels
                    points_by_z = {}
                    labels_by_z = {}
                    for idx, point in enumerate(pts_arr):
                        z = int(point[0])
                        if z not in points_by_z:
                            points_by_z[z] = []
                            labels_by_z[z] = []
                        points_by_z[z].append(point[1:])  # Store (Y, X) only
                        labels_by_z[z].append(point_labels[idx] if idx < len(point_labels) else 1)
                    
                    # Segment each Z-slice that has points
                    seg_id = np.zeros(shape, dtype=np.uint32)
                    segmented_slices = []
                    
                    for z, pos_points in points_by_z.items():
                        # Get points and labels for this slice
                        slice_points = np.array(pos_points)
                        slice_labels = np.array(labels_by_z[z])
                        
                        # Segment this slice
                        try:
                            mask_2d = sam_util.prompt_segmentation(
                                state.predictor,
                                slice_points,
                                slice_labels,
                                boxes=np.array([]),
                                masks=None,
                                shape=shape[1:],  # (Y, X) for 2D slice
                                multiple_box_prompts=False,
                                i=z,
                                image_embeddings=state.image_embeddings,
                            )
                            if mask_2d is not None and mask_2d.max() > 0:
                                seg_id[z] = mask_2d
                                segmented_slices.append(z)
                        except Exception:
                            pass
                    
                    # Extend through the volume using all segmented slices as anchors
                    if len(segmented_slices) > 0:
                        try:
                            seg_id, (z_min, z_max) = segment_mask_in_volume(
                                seg_id,
                                state.predictor,
                                state.image_embeddings,
                                np.array(segmented_slices),  # Use ALL segmented slices
                                stop_lower=False,
                                stop_upper=False,
                                iou_threshold=0.65,  # Higher threshold = more conservative
                                projection="single_point",
                                box_extension=0.05,
                                update_progress=None,
                            )
                        except Exception:
                            pass
                    
                    # Assign to merged segmentation with ID=1 (standard single object)
                    if seg_id.max() > 0:
                        mask = seg_id > 0
                        seg_merged[mask] = 1  # Always use ID=1 for single object workflow
                        print(f"... Done")
                    else:
                        print(f"... Skipped")
                
                seg = seg_merged if seg_merged.max() > 0 else None
                  
                if seg is None:
                    continue
                    
                if seg is not None:
                    # Store directly into current_object_4d and update the layer
                    self.current_object_4d[int(t)] = seg.astype(np.uint32)
                    
                    # Update the napari layer
                    try:
                        lay = self._viewer.layers["current_object_4d"] if "current_object_4d" in self._viewer.layers else None
                        if lay is not None:
                            lay.refresh()
                    except Exception:
                        pass
                    
            except Exception:
                continue

        # Save the current timestep's segmentation before restoring (to preserve it)
        original_t_seg = None
        if self.current_object_4d is not None:
            try:
                original_t_seg = self.current_object_4d[int(original_t)].copy()
            except Exception:
                pass

        # Reconnect dimension slider callback
        if dims_callback_disconnected:
            try:
                self._viewer.dims.events.current_step.connect(self._on_dims_current_step)
            except Exception:
                pass

        # Restore original timestep and its point prompts
        try:
            # Use the proper API to restore timestep which will trigger _on_dims_current_step
            # This will properly reload points with their IDs
            self._viewer.dims.set_current_step(0, int(original_t))
        except Exception:
            pass

        # Restore the segmentation for the original timestep (in case callback overwrote it)
        if original_t_seg is not None and self.current_object_4d is not None:
            try:
                self.current_object_4d[int(original_t)] = original_t_seg
            except Exception:
                pass

        # Force refresh of current_object_4d layer to show the segmentation
        try:
            lay = self._viewer.layers["current_object_4d"] if "current_object_4d" in self._viewer.layers else None
            if lay is not None:
                lay.refresh()
        except Exception:
            pass

        # Reconnect point prompts event listener
        if point_layer is not None and hasattr(point_layer, 'events') and hasattr(point_layer.events, 'data'):
            try:
                if hasattr(self, '_point_prompt_connection'):
                    point_layer.events.data.connect(self._point_prompt_connection)
            except Exception:
                pass
        
        print("Done segmenting all prompts")

        try:
            show_info("Finished segmenting all timesteps with point prompts.")
        except Exception:
            pass

    def commit_all_timesteps(self):
        """Transfer all non-empty `current_object_4d[t]` into `committed_objects_4d` preserving their object IDs."""
        layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
        if layer is None:
            if self.segmentation_4d is None and self.image_4d is not None:
                self.segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                layer = self._viewer.add_labels(self.segmentation_4d, name="committed_objects_4d")
            except Exception:
                return

        # Sync local reference
        try:
            self.segmentation_4d = layer.data
        except Exception:
            pass

        # Get global max to offset new IDs if needed
        try:
            global_max = int(self.segmentation_4d.max()) if self.segmentation_4d is not None else 0
        except Exception:
            global_max = 0

        for t in range(int(self.n_timesteps)):
            if self.current_object_4d is None:
                break
            try:
                seg_t = np.asarray(self.current_object_4d[int(t)])
            except Exception:
                seg_t = None
            if seg_t is None or seg_t.size == 0 or not np.any(seg_t != 0):
                continue

            # Get unique object IDs in this timestep
            unique_ids = np.unique(seg_t[seg_t > 0])
            
            # Commit each object with its current ID (offset by global_max to avoid collisions)
            for obj_id in unique_ids:
                mask = seg_t == obj_id
                new_id = int(global_max + obj_id)
                
                try:
                    if hasattr(layer, "events") and hasattr(layer.events, "data"):
                        with layer.events.data.blocker():
                            layer.data[int(t)][mask] = new_id
                    else:
                        layer.data[int(t)][mask] = new_id
                except Exception:
                    try:
                        arr = np.asarray(layer.data[int(t)])
                        arr[mask] = new_id
                        layer.data[int(t)] = arr
                    except Exception:
                        pass

            # Update local cache
            try:
                self.segmentation_4d = layer.data
                if self._segmentation_cache is None:
                    self._segmentation_cache = [None] * self.n_timesteps
                self._segmentation_cache[int(t)] = np.asarray(layer.data[int(t)]).copy()
            except Exception:
                pass

        try:
            layer.refresh()
        except Exception:
            pass

        # Cleanup: clear all point prompts and current object masks after committing
        # Clear stored per-timestep prompts
        try:
            self.point_prompts_4d = {}
        except Exception:
            pass
        # Clear napari points layer
        try:
            if "point_prompts" in self._viewer.layers:
                self._viewer.layers["point_prompts"].data = np.empty((0, 3))
        except Exception:
            pass

        # Clear the 4D current object layer content
        try:
            if self.current_object_4d is not None:
                self.current_object_4d[...] = 0
            if "current_object_4d" in self._viewer.layers:
                lay_cur = self._viewer.layers["current_object_4d"]
                if hasattr(lay_cur, "events") and hasattr(lay_cur.events, "data"):
                    with lay_cur.events.data.blocker():
                        lay_cur.data = self.current_object_4d
                else:
                    lay_cur.data = self.current_object_4d
                try:
                    lay_cur.refresh()
                except Exception:
                    pass
        except Exception:
            pass

        # Optional: also clear 3D current_object layer if present
        try:
            if "current_object" in self._viewer.layers:
                co3d = self._viewer.layers["current_object"].data
                zeros3d = np.zeros_like(co3d, dtype=np.uint32)
                self._viewer.layers["current_object"].data = zeros3d
                try:
                    self._viewer.layers["current_object"].refresh()
                except Exception:
                    pass
        except Exception:
            pass

    def propagate_point_prompts_to_all_timesteps(self):
        """Intelligently propagate point prompts across all timesteps.
        
        For each timestep:
        1. Copy point locations from previous timestep
        2. Move each point to local intensity maximum within search radius
        3. If point lands in background (black), assign to nearest unprompted neuron blob
        4. Preserve point IDs across timesteps for consistent tracking
        """
        if self.image_4d is None or self.n_timesteps is None:
            show_info("No 4D image data loaded")
            return
        
        # Get current timestep to use as anchor
        current_t = int(getattr(self, "current_timestep", 0) or 0)
        
        # Check if current timestep has point prompts
        if not hasattr(self, "point_prompts_4d"):
            self.point_prompts_4d = {}
        
        # Get points from current layer (most up-to-date)
        if "point_prompts" in self._viewer.layers:
            current_points = np.array(self._viewer.layers["point_prompts"].data)
        else:
            current_points = self.point_prompts_4d.get(current_t, np.empty((0, 3)))
        
        if len(current_points) == 0:
            show_info(f"No point prompts in current timestep {current_t}. Please add points first.")
            return
        
        # Store current points with their IDs
        self.point_prompts_4d[current_t] = current_points.copy()
        
        # Get point IDs for current timestep
        current_ids = self._get_point_ids_for_timestep(current_t, current_points)
        
        # Base search radius for local maxima (in pixels)
        BASE_SEARCH_RADIUS = 7
        LARGE_SEARCH_RADIUS = 15  # Used when large displacement detected
        DISPLACEMENT_THRESHOLD = 10  # Pixels - above this triggers large search
        
        # Intensity threshold percentile (below this is considered background)
        BACKGROUND_THRESHOLD_PERCENTILE = 50
        
        print(f"🔄 Propagating {len(current_points)} points from timestep {current_t} to all {self.n_timesteps} timesteps")
        
        # Forward propagation (current_t -> end)
        for t in range(current_t + 1, int(self.n_timesteps)):
            # Calculate adaptive search radius based on previous displacement
            search_radius = self._calculate_adaptive_search_radius(
                t - 1, t - 2, BASE_SEARCH_RADIUS, LARGE_SEARCH_RADIUS, DISPLACEMENT_THRESHOLD
            )
            self._propagate_points_to_timestep(
                t, t - 1, search_radius, BACKGROUND_THRESHOLD_PERCENTILE
            )
        
        # Backward propagation (current_t -> start)
        for t in range(current_t - 1, -1, -1):
            # Calculate adaptive search radius based on previous displacement
            search_radius = self._calculate_adaptive_search_radius(
                t + 1, t + 2, BASE_SEARCH_RADIUS, LARGE_SEARCH_RADIUS, DISPLACEMENT_THRESHOLD
            )
            self._propagate_points_to_timestep(
                t, t + 1, search_radius, BACKGROUND_THRESHOLD_PERCENTILE
            )
        
        # Update viewer to show propagated points
        self._update_viewer_after_propagation(current_t)
        
        total_points = sum(len(pts) for pts in self.point_prompts_4d.values())
        show_info(f"✅ Propagated points to all timesteps! Total points: {total_points}")
    
    def _calculate_adaptive_search_radius(self, current_t, prev_t, base_radius, large_radius, displacement_threshold):
        """Calculate adaptive search radius based on recent displacement between timesteps.
        
        Args:
            current_t: Current timestep
            prev_t: Previous timestep to compare against
            base_radius: Normal search radius for small movements
            large_radius: Expanded search radius for large movements
            displacement_threshold: Displacement above which to use large_radius
            
        Returns:
            Adaptive search radius (int)
        """
        # If we don't have points for both timesteps, use base radius
        if not hasattr(self, 'point_prompts_4d'):
            return base_radius
            
        current_points = self.point_prompts_4d.get(current_t)
        prev_points = self.point_prompts_4d.get(prev_t)
        
        # Need both timesteps to calculate displacement
        if current_points is None or prev_points is None or len(current_points) == 0 or len(prev_points) == 0:
            return base_radius
        
        # Calculate mean displacement between matched points
        # Use point IDs to match corresponding neurons
        current_ids = self._get_point_ids_for_timestep(current_t, current_points)
        prev_ids = self._get_point_ids_for_timestep(prev_t, prev_points)
        
        # Build ID -> position mappings
        current_id_map = {current_ids[i]: current_points[i] for i in range(len(current_points))}
        prev_id_map = {prev_ids[i]: prev_points[i] for i in range(len(prev_points))}
        
        # Calculate displacements for matched IDs
        displacements = []
        for point_id in current_id_map:
            if point_id in prev_id_map:
                curr_pos = current_id_map[point_id]
                prev_pos = prev_id_map[point_id]
                displacement = np.sqrt(np.sum((curr_pos - prev_pos)**2))
                displacements.append(displacement)
        
        if len(displacements) == 0:
            return base_radius
        
        # Use max displacement to be conservative (handle worst case)
        max_displacement = np.max(displacements)
        mean_displacement = np.mean(displacements)
        
        # If recent movement was large, use expanded search radius
        if max_displacement > displacement_threshold:
            print(f"  ⚠ Large displacement detected ({max_displacement:.1f}px) - using expanded search radius {large_radius}px")
            return large_radius
        else:
            return base_radius
    
    def _propagate_points_to_timestep(self, target_t, source_t, search_radius, threshold_percentile):
        """Propagate points from source_t to target_t using Hungarian algorithm for optimal matching."""
        # Get source points
        source_points = self.point_prompts_4d.get(source_t, np.empty((0, 3)))
        if len(source_points) == 0:
            return
        
        # Get 3D image for target timestep
        image_3d = self.image_4d[int(target_t)]
        
        # Calculate threshold for background detection
        nonzero_values = image_3d[image_3d > 0]
        if len(nonzero_values) > 0:
            threshold = np.percentile(nonzero_values, threshold_percentile)
        else:
            threshold = 0
        
        # Create binary mask of foreground regions
        foreground_mask = image_3d > threshold
        
        # Label connected components in foreground
        labeled_blobs, num_blobs = label(foreground_mask)
        
        if num_blobs == 0:
            # No blobs found - keep points at original positions
            self.point_prompts_4d[target_t] = source_points.copy()
            point_ids = self._get_point_ids_for_timestep(source_t, source_points)
            for idx, point in enumerate(source_points):
                point_id = point_ids[idx] if idx < len(point_ids) else (idx + 1)
                z, y, x = int(point[0]), int(point[1]), int(point[2])
                self._set_point_id_from_coords(target_t, z, y, x, point_id)
            return
        
        # Get blob centroids and properties
        blob_centroids = {}
        blob_sizes = {}
        blob_intensities = {}
        
        for blob_id in range(1, num_blobs + 1):
            blob_mask = labeled_blobs == blob_id
            if np.any(blob_mask):
                # Centroid
                centroid = center_of_mass(blob_mask)
                blob_centroids[blob_id] = np.array(centroid)
                
                # Size (number of pixels)
                blob_sizes[blob_id] = np.sum(blob_mask)
                
                # Average intensity
                blob_intensities[blob_id] = np.mean(image_3d[blob_mask])
        
        # Get point IDs
        point_ids = self._get_point_ids_for_timestep(source_t, source_points)
        
        # Get source image to determine which blob each point was on
        image_3d_source = self.image_4d[int(source_t)]
        foreground_mask_source = image_3d_source > threshold
        labeled_blobs_source, _ = label(foreground_mask_source)
        
        # Step 1: Find candidate blobs for each point (local maximum search)
        # IMPORTANT: Constrain search to same blob to prevent jumping between neurons
        point_candidates = []  # List of dict with point_idx, blob_id, position, etc.
        
        for idx, point in enumerate(source_points):
            z, y, x = int(point[0]), int(point[1]), int(point[2])
            
            # Determine which blob this point was on in source frame
            source_blob_id = labeled_blobs_source[z, y, x] if z < labeled_blobs_source.shape[0] else 0
            
            # Find local maximum, preferring to stay within the same blob
            new_z, new_y, new_x, stayed_in_blob = self._find_local_maximum_in_blob(
                image_3d, labeled_blobs, z, y, x, search_radius, source_blob_id
            )
            
            # Get blob at new position
            blob_id = labeled_blobs[new_z, new_y, new_x]
            
            if blob_id > 0:
                # Point lands on a blob
                distance = np.sqrt((new_z - z)**2 + (new_y - y)**2 + (new_x - x)**2)
                intensity = image_3d[new_z, new_y, new_x]
                point_candidates.append({
                    'point_idx': idx,
                    'blob_id': blob_id,
                    'position': (new_z, new_y, new_x),
                    'distance': distance,
                    'intensity': intensity,
                    'original_pos': (z, y, x)
                })
            else:
                # Point lands in background - find nearest blob
                nearest_blob, nearest_dist = self._find_nearest_blob(
                    blob_centroids, z, y, x
                )
                if nearest_blob is not None:
                    centroid = blob_centroids[nearest_blob]
                    cz, cy, cx = int(centroid[0]), int(centroid[1]), int(centroid[2])
                    point_candidates.append({
                        'point_idx': idx,
                        'blob_id': nearest_blob,
                        'position': (cz, cy, cx),
                        'distance': nearest_dist,
                        'intensity': blob_intensities[nearest_blob],
                        'original_pos': (z, y, x)
                    })
                else:
                    # No blobs available - will handle later
                    point_candidates.append({
                        'point_idx': idx,
                        'blob_id': None,
                        'position': (z, y, x),
                        'distance': 0,
                        'intensity': 0,
                        'original_pos': (z, y, x)
                    })
        
        # Step 2: Use Hungarian algorithm for optimal point-to-blob assignment
        assignments = self._hungarian_match_points_to_blobs(
            point_candidates, blob_centroids, source_points
        )
        
        # Step 3: Apply assignments and update point prompts
        propagated_points = []
        for idx, assignment in enumerate(assignments):
            point_id = point_ids[idx] if idx < len(point_ids) else (idx + 1)
            z, y, x = assignment['position']
            propagated_points.append([z, y, x])
            self._set_point_id_from_coords(target_t, z, y, x, point_id)
        
        # Store propagated points
        self.point_prompts_4d[target_t] = np.array(propagated_points)
    
    def _hungarian_match_points_to_blobs(self, point_candidates, blob_centroids, source_points):
        """Use Hungarian algorithm to optimally assign points to blobs.
        
        This prevents:
        - Multiple points clustering on the same blob
        - Points disappearing when greedy matching fails
        - Incorrect neuron assignments
        """
        n_points = len(source_points)
        unique_blobs = set(c['blob_id'] for c in point_candidates if c['blob_id'] is not None)
        
        if len(unique_blobs) == 0:
            # No blobs available - keep original positions
            return point_candidates
        
        # Build list of all blobs (including duplicates from candidates)
        all_blobs = list(unique_blobs)
        n_blobs = len(all_blobs)
        
        # If more points than blobs, add dummy blobs (points will keep original positions)
        if n_points > n_blobs:
            n_blobs = n_points
            all_blobs.extend([None] * (n_points - len(all_blobs)))
        
        # Create cost matrix: cost[point_idx, blob_idx]
        # Lower cost = better match
        cost_matrix = np.zeros((n_points, n_blobs))
        LARGE_COST = 1e6
        
        for i, candidate in enumerate(point_candidates):
            for j, blob_id in enumerate(all_blobs):
                if blob_id is None:
                    # Dummy blob - high cost (only use if no real blob available)
                    cost_matrix[i, j] = LARGE_COST
                elif candidate['blob_id'] == blob_id:
                    # This is the candidate blob for this point - use distance as cost
                    cost_matrix[i, j] = candidate['distance']
                elif blob_id in blob_centroids:
                    # Alternative blob - compute distance from original position
                    orig_z, orig_y, orig_x = candidate['original_pos']
                    centroid = blob_centroids[blob_id]
                    distance = np.sqrt(
                        (centroid[0] - orig_z)**2 +
                        (centroid[1] - orig_y)**2 +
                        (centroid[2] - orig_x)**2
                    )
                    # Penalize alternative blobs slightly to prefer original candidates
                    cost_matrix[i, j] = distance * 1.2
                else:
                    cost_matrix[i, j] = LARGE_COST
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Build final assignments
        assignments = []
        for point_idx, blob_idx in zip(row_indices, col_indices):
            assigned_blob = all_blobs[blob_idx]
            
            if assigned_blob is None or cost_matrix[point_idx, blob_idx] >= LARGE_COST:
                # No good blob match - keep original position
                orig_pos = point_candidates[point_idx]['original_pos']
                assignments.append({
                    'position': orig_pos,
                    'blob_id': None
                })
            elif assigned_blob == point_candidates[point_idx]['blob_id']:
                # Matched to candidate blob - use candidate position
                assignments.append(point_candidates[point_idx])
            else:
                # Matched to alternative blob - use its centroid
                centroid = blob_centroids[assigned_blob]
                cz, cy, cx = int(centroid[0]), int(centroid[1]), int(centroid[2])
                assignments.append({
                    'position': (cz, cy, cx),
                    'blob_id': assigned_blob
                })
        
        return assignments
    
    def _find_nearest_blob(self, blob_centroids, z, y, x):
        """Find the nearest blob centroid to position (z, y, x).
        
        Returns:
            (blob_id, distance) or (None, inf) if no blobs
        """
        min_distance = float('inf')
        nearest_blob = None
        
        for blob_id, centroid in blob_centroids.items():
            distance = np.sqrt(
                (centroid[0] - z)**2 +
                (centroid[1] - y)**2 +
                (centroid[2] - x)**2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_blob = blob_id
        
        return nearest_blob, min_distance
    
    def _find_local_maximum(self, image_3d, z, y, x, radius):
        """Find local intensity maximum within radius of (z, y, x).
        
        Returns the brightest pixel within the search radius.
        Note: This does NOT constrain to a specific blob - blob filtering
        happens in the Hungarian matching step.
        """
        shape = image_3d.shape
        
        # Define search box
        z_min = max(0, z - radius)
        z_max = min(shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(shape[2], x + radius + 1)
        
        # Extract neighborhood
        neighborhood = image_3d[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Find maximum within neighborhood
        max_idx = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
        
        # Convert back to global coordinates
        new_z = z_min + max_idx[0]
        new_y = y_min + max_idx[1]
        new_x = x_min + max_idx[2]
        
        return new_z, new_y, new_x
    
    def _find_local_maximum_in_blob(self, image_3d, labeled_blobs, z, y, x, radius, current_blob_id):
        """Find local intensity maximum within radius, constrained to the same blob.
        
        Args:
            image_3d: 3D intensity image
            labeled_blobs: Labeled blob array from scipy.ndimage.label
            z, y, x: Current position
            radius: Search radius
            current_blob_id: Blob ID to constrain search to (0 means unconstrained)
            
        Returns:
            (new_z, new_y, new_x, found_in_same_blob)
        """
        shape = image_3d.shape
        
        # Define search box
        z_min = max(0, z - radius)
        z_max = min(shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(shape[2], x + radius + 1)
        
        # Extract neighborhood
        neighborhood = image_3d[z_min:z_max, y_min:y_max, x_min:x_max]
        blob_neighborhood = labeled_blobs[z_min:z_max, y_min:y_max, x_min:x_max]
        
        if current_blob_id > 0:
            # First try to find maximum within the same blob
            same_blob_mask = blob_neighborhood == current_blob_id
            if np.any(same_blob_mask):
                # Extract only pixels from the same blob
                blob_intensities = neighborhood[same_blob_mask]
                
                # Find maximum intensity within same blob
                max_intensity = np.max(blob_intensities)
                
                # Find position of this maximum in the neighborhood
                max_positions = np.where((neighborhood == max_intensity) & same_blob_mask)
                
                # Use first occurrence if multiple maxima
                max_idx = (max_positions[0][0], max_positions[1][0], max_positions[2][0])
                
                new_z = int(z_min + max_idx[0])
                new_y = int(y_min + max_idx[1])
                new_x = int(x_min + max_idx[2])
                
                return new_z, new_y, new_x, True
        
        # If no constraint or blob not found, search globally
        max_idx = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
        
        new_z = int(z_min + max_idx[0])
        new_y = int(y_min + max_idx[1])
        new_x = int(x_min + max_idx[2])
        
        return new_z, new_y, new_x, False
    
    def _find_nearest_unprompted_blob(self, labeled_blobs, blob_has_point, z, y, x, num_blobs):
        """Find centroid of nearest blob that doesn't have a point yet."""
        if num_blobs == 0:
            return None
        
        # Get centroids of all blobs
        from scipy.ndimage import center_of_mass
        
        # Find unprompted blobs
        unprompted_blobs = []
        for blob_id in range(1, num_blobs + 1):
            if blob_id not in blob_has_point:
                blob_mask = labeled_blobs == blob_id
                if np.any(blob_mask):
                    centroid = center_of_mass(blob_mask)
                    unprompted_blobs.append((blob_id, centroid))
        
        if len(unprompted_blobs) == 0:
            return None
        
        # Find nearest unprompted blob
        min_distance = float('inf')
        nearest_centroid = None
        
        for blob_id, (cz, cy, cx) in unprompted_blobs:
            distance = np.sqrt((cz - z)**2 + (cy - y)**2 + (cx - x)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_centroid = (int(cz), int(cy), int(cx))
        
        return nearest_centroid
    
    def _find_local_maximum_3d(self, image_3d, z, y, x, radius):
        """Find local intensity maximum within radius of (z, y, x) in 3D.
        
        This is an alias for _find_local_maximum for use in segmentation propagation.
        Returns tuple of (new_z, new_y, new_x).
        """
        return self._find_local_maximum(image_3d, z, y, x, radius)
    
    def _update_viewer_after_propagation(self, current_t):
        """Update viewer to show propagated points for current timestep."""
        if "point_prompts" in self._viewer.layers:
            # Get points for current timestep
            current_points = self.point_prompts_4d.get(current_t, np.empty((0, 3)))
            
            # Update layer
            layer = self._viewer.layers["point_prompts"]
            layer.data = current_points
            
            # Update colors based on IDs
            if len(current_points) > 0:
                point_ids = self._get_point_ids_for_timestep(current_t, current_points)
                self._update_point_colors(layer, point_ids)
        
        # Refresh point manager if available
        if hasattr(self, "_point_manager_widget"):
            self._point_manager_widget.refresh_point_list()
