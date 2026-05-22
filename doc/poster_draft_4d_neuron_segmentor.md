# Poster draft — 4D Neuron Segmentor (micro-sam-4d)

This file is **poster-ready text + a suggested box structure** for a 3-column science poster.
Replace placeholders in [BRACKETS] and drop in your own screenshots/figures.

## Suggested box layout (matches a common 3-column template)

**Left column**
- Box 1: Background / Problem
- Box 2: Objectives / Contributions

**Middle column**
- Box 3: Method / Algorithm (4D pipeline)
- Box 4: System / Implementation (napari + embeddings + tiling)

**Right column**
- Box 5: Results (qualitative panels)
- Box 6: Evaluation (optional quantitative table)

**Bottom strip**
- References • Acknowledgements • Contact / QR

---

## Top banner (full width)

**Title (pick one)**
1) 4D Neuron Segmentor: Interactive segmentation + tracking in volumetric time-lapse microscopy using SAM embeddings
2) Fast 4D neuron segmentation and tracking with prompt-based propagation and tiled SAM embeddings
3) Semi-automatic 4D neuron annotation in *C. elegans* using Segment Anything for Microscopy

**Authors / affiliation**
[YOUR NAME], [MENTOR / LAB] — [SCHOOL / INSTITUTE]

**One-sentence summary (small text under title)**
We built a 4D (3D+t) neuron segmentation workflow that uses SAM-based embeddings, prompt-based propagation, and tracking to reduce manual annotation time while preserving object consistency across slices and time.

---

## Left column

### Background / Problem
Volumetric time-lapse microscopy (3D+t) is a powerful way to study neural structure and dynamics, but manual neuron annotation is slow and difficult:
- **Scale:** hundreds of z-slices × many time points.
- **Dense packing:** neurons touch/overlap; boundaries can be ambiguous.
- **Drift + intensity changes:** morphology and appearance vary over time.
- **Consistency:** the same neuron must keep a stable identity across frames.

General-purpose segmentation models often struggle without domain adaptation, while classical methods can be brittle across imaging conditions.

### Project Goal
Create a practical **4D neuron segmentor** that:
- Supports **interactive** neuron segmentation from a few prompts.
- Propagates segmentations through **3D volumes** and across **time**.
- Works on large images via **tiling** and cached **image embeddings**.
- Produces outputs suitable for downstream analysis (label masks + tracks).

### Key Contributions (what’s new in your workflow)
- A unified workflow for **3D segmentation + time tracking** using SAM embeddings.
- **Prompt-to-volume propagation:** segment one slice and extend through z using projected prompts.
- **Efficient large-data handling:** precompute and reuse embeddings; optional tiled embeddings.
- A repeatable pipeline that turns a small number of interactions into consistent 4D labels.

---

## Middle column

### Method Overview (Algorithm)
**Input:** 3D+t microscopy volume (time series of 3D stacks)

**Core idea:** use **SAM image embeddings** as a reusable representation and drive segmentation via prompts (mask/box/points). Then propagate and track.

**Step 1 — Embedding precomputation (fast reuse)**
- Compute SAM image embeddings once per image/volume (optionally **tiled** for large data).
- Reuse embeddings for interactive updates without rerunning the full model each time.

**Step 2 — Interactive segmentation (one object at a time)**
- User provides prompts (points, boxes, or an initial mask).
- SAM predicts a mask; user refines if needed.

**Step 3 — 3D propagation across z (prompt projection)**
Starting from a segmented slice, propagate to neighboring slices by projecting prompts:
- Projection modes: **box**, **mask**, **points**, or combinations.
- Continue propagation while overlap stays above an **IoU threshold**.
- Optional gap handling to avoid holes and maintain continuity.

**Step 4 — Tracking across time (4D consistency)**
- Link object instances over time to maintain IDs.
- When available, integrate learned tracking (e.g., Trackastra) or use heuristic linking.

**Output:**
- 4D instance segmentation labels (T×Z×Y×X)
- Track table (object IDs over time; optional CTC/napari tracks)

### Implementation (Software)
- Built on `micro_sam` (Segment Anything for Microscopy): napari tools + Python API.
- Uses SAM predictor + embedding cache; supports large images via tiling.
- Designed to run interactively in napari and programmatically in Python.

### What to include as a figure in this column
- **Pipeline diagram:** prompts → embeddings → 3D propagation → tracking.
- **Screenshot:** napari layers (raw, embeddings optional, labels, tracks).

---

## Right column

### Results / Demonstrations
(Use your own screenshots; avoid invented numbers.)

**Qualitative demo panels (recommended):**
1) A 3D stack with a neuron segmented from minimal prompts.
2) Propagation through z showing consistent boundaries.
3) 4D tracking across time showing stable IDs and reduced flicker.

**Quantitative evaluation (optional; fill in if you measured it):**
- Segmentation: IoU / Dice vs manual labels on [N] frames.
- Tracking: ID switches, track fragmentation, or CTC metrics.
- Efficiency: average time per neuron or per volume before vs after.

### Key Takeaways
- Embedding reuse makes interactive refinement fast.
- Prompt-based propagation reduces manual slice-by-slice annotation.
- Tracking produces consistent neuron identities in 4D sequences.

### Limitations
- Very dense regions can still merge without additional prompts.
- Tracking quality depends on motion, imaging quality, and object appearance.
- Model behavior can vary across modalities; fine-tuning may help.

### Future Work
- Domain-specific fine-tuning for neuron data.
- Automated quality checks (merge/split detection).
- Better stimulus- or behavior-aligned summaries once 4D labels are stable.

---

## Bottom strip (full width)

### Usage (1–2 lines)
Interactive segmentation/tracking via napari plugin: `Plugins → Segment Anything for Microscopy`.

### References
- Segment Anything (SAM): Kirillov et al.
- micro-sam (Segment Anything for Microscopy): [Nature Methods paper]
- (If used) Trackastra for tracking.

### Acknowledgements
[MENTOR / LAB], [DATA PROVIDERS], [PROGRAM / FUNDING]

### Contact
[EMAIL] • [GITHUB / LINK]
