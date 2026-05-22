# Poster draft — C. elegans AVA/AVB calcium responses (NWB)

Use this as *ready-to-paste* text + a suggested box layout that matches a 3-column science-poster template.
Replace bracketed placeholders like [YOUR AFFILIATION].

---

## Top banner

**Title (1 line)**
Sex-specific AVA/AVB calcium dynamics aligned to sensory stimuli in *C. elegans* (NWB)

**Authors / affiliation (1–2 lines)**
[YOUR NAME], [MENTOR / LAB] — [SCHOOL / INSTITUTE]

**Optional subtitle (small)**
First-cycle stimulus alignment • ΔF/F0 normalization • Per-worm metrics • Sex-stratified comparisons • CSV export for Excel statistics

---

## Left column

### Background & Motivation
Neural activity in *C. elegans* can be measured with calcium imaging, but comparing responses across animals requires careful alignment to stimulus timing and consistent normalization. This project builds a reproducible pipeline to extract AVA/AVB activity from NWB files, align traces to stimulus onset, and quantify response strength and dynamics at the **worm level** (not frame level) to enable downstream statistics.

**Why AVA/AVB?**
- AVA: key interneuron associated with reversal/avoidance-related circuits.
- AVB: key interneuron associated with forward locomotion-related circuits.

### Project Goal
Create an end-to-end analysis workflow that:
- Detects stimuli from NWB annotations and uses **only the first occurrence (“cycle 1”)**.
- Computes **F/F0** (for export) and **ΔF/F0** (for metrics/plots) using a consistent baseline.
- Produces publication-ready plots and exports tables for Excel-based statistics.

### Data (NWB)
**Input:** NWB files containing:
- `processing/CalciumActivity/ActivityTraces` (neuron names + fluorescence/activity)
- `processing/CalciumActivity/StimulusInfo` (timestamps + stimulus labels)
- `acquisition/CalciumImageSeries` (sampling rate, starting time)

**Neurons extracted:** AVAL, AVAR, AVBL, AVBR

**Sex grouping (directory mode):** inferred from filename tokens (e.g. `-h2` = hermaphrodite, `-m3` = male).

---

## Middle column

### Methods: Analysis Pipeline
**1) Read & validate NWB**
- Load stimulus events and calcium traces.
- Verify required modules and columns exist.

**2) Stimulus detection (first-cycle only)**
- Compress stimulus annotations to label-change points.
- For each distinct stimulus label, select the **first timestamp** as cycle 1.

**3) Window extraction & alignment**
- Extract a window around cycle 1 (default: −15s to +15s).
- Interpolate each neuron trace onto a common relative-time grid.

**4) Bilateral combination (within worm)**
- AVA trace = mean(AVAL, AVAR)
- AVB trace = mean(AVBL, AVBR)

**5) Normalization**
- Baseline window for F0: **[−3.5, 0) seconds** relative to stimulus.
- Exported signal: **F/F0** over the full window.
- Response analysis: **ΔF/F0** over **0 to 15s**.

**6) Per-worm metrics (computed per worm first)**
Within the response window (0..15s):
- Mean ΔF/F0
- Slope (least-squares linear regression)
- AUC (trapezoidal integral)

**7) Group summaries**
- Combine bilateral metrics within each worm.
- Aggregate across worms by sex: mean ± SD.

### Implementation
- Python: `pynwb`, `numpy`, `matplotlib`
- Outputs:
  - One plot per stimulus label (first-cycle only)
  - CSV tables for Excel statistics

---

## Right column

### Results (what to show)
Include 2–4 example stimuli that illustrate different response types:
- **Food (OP50):** expected modulation of interneuron activity.
- **Aversive/chemosensory stimuli (e.g., CuSO4, IAA):** stimulus-locked changes.
- **Osmotic stimuli (e.g., sorbitol):** distinct dynamics.

For each selected stimulus, show:
- AVA ΔF/F0 (0..15s)
- AVB ΔF/F0 (0..15s)
- Raw AVA/AVB activity (−pre..+post) with stimulus at 0s
- Text box with sex mean ± SD of mean/slope/AUC (worm-level)

### Key Takeaways
- A standardized NWB-to-metrics pipeline enables fair comparisons across worms.
- First-cycle alignment reduces confounds from adaptation across repeated cycles.
- Exported F/F0 time-series and per-worm metrics allow statistical testing in Excel/R/Python.

### Limitations
- Missing neuron annotations (some NWBs may lack AVAL/AVAR or AVBL/AVBR).
- Stimulus labels vary across experiments; cleaning/standardization may be needed.
- Sex inference relies on filename conventions.

### Next Steps
- Add automated label normalization (e.g., merge equivalent names like “100mM NaCl” vs “100 mM NaCl”).
- Add confidence intervals or bootstrap summaries across worms.
- Extend to additional neurons / circuit-level comparisons.

---

## Bottom strip (small)

### CSV Exports (for Excel)
Two tables are exported when `--export-csv` is enabled:
- **Timeseries F/F0:** long-format table with columns: `stimulus_label, worm, sex, neuron(AVA/AVB), rel_time_s, ff0, f0`
- **Per-worm metrics (ΔF/F0):** `stimulus_label, worm, sex, neuron, mean_dff, slope_dff_per_s, auc_dff`

### Acknowledgements
[MENTOR / LAB], [DATA PROVIDERS], [FUNDING / PROGRAM]

### References (2–4)
- NWB: Neurodata Without Borders (data standard)
- Ca imaging / analysis references relevant to your dataset

### Contact / QR
[EMAIL] • [GITHUB / LINK TO REPO OR REPORT]
