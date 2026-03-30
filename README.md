# Healing Stones — Mayan Stele Fragment Reconstruction

Reconstructing a fragmented Mayan stele from 17 3D scanned fragments using machine learning and geometric registration.

## Pipelines

### 3D ML Pipeline (`reconstruct.py`)
1. Load .PLY fragments → boundary-aware sampling → PCA alignment
2. Extract geometric features per point (curvature, planarity, normal consistency etc.)
3. Train Random Forest with 5-fold cross validation to classify break surfaces vs original stone
4. Run initial RANSAC pass to generate relationship training data
5. Train Neural Network to predict which fragment pairs are neighbors
6. Use ML scores to prioritize registration order
7. Run multi-scale RANSAC + ICP with planarity penalty (6 scales)
8. Break surface exclusivity — matched edges removed from future candidates
9. Global pose graph optimization
10. Assemble fragments + detect missing material regions

### 2D Pipeline (`reconstruct_2d.py`)
1. Load PNG fragments
2. Extract SIFT keypoints + contour shape descriptors
3. Match all fragment pairs
4. Estimate layout via homography

### Fragment Ancestry Graph (`ancestry.py`)
1. Extract geometric properties per fragment (volume, aspect ratio, curvature, carving depth)
2. Detect front vs back facing fragments using normal direction analysis
3. Train neural network to predict vertical position (top → bottom)
4. Estimate horizontal positions (left → right)
5. Build directed ancestry graph of spatial relationships

## Results

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| Break classifier accuracy | 97.57% | 97.66% |
| Match coverage | 51.47% | 47.79% |
| Gap ratio | 94.4% | 92.65% |
| Max fitness | 1.0 | 0.9456 |

Note: Coverage reduction is expected as planarity penalty correctly rejects false flat surface matches, producing fewer but more accurate results.

## Setup
```bash
conda create -n healing python=3.10 -y
conda activate healing
pip install -r requirements.txt
```

## Run
```bash
# 3D ML reconstruction
python reconstruct.py

# 2D reconstruction
python reconstruct_2d.py

# Fragment ancestry graph
python ancestry.py
```

## Output
- `output_ml/reconstructed_stele.ply` — assembled 3D stele
- `output_ml/metrics_ml.json` — all metrics
- `output_ml/break_surfaces.png` — ML-detected break surfaces
- `output_ml/confusion_matrix.png` — classifier evaluation
- `output_ml/training_curves.png` — NN training curves
- `output_ml/feature_importance.png` — RF feature importance
- `output_ml/fitness_matrix.png` — pairwise registration heatmap
- `output_ml/connectivity_graph.png` — fragment connectivity
- `output_ml/before_after.png` — before vs after registration
- `output_ancestry/ancestry_graph.png` — directed spatial graph
- `output_ancestry/spatial_layout.png` — predicted stele layout
- `output_ancestry/vertical_ordering.png` — top to bottom ordering
- `output_2d/` — 2D pipeline outputs

## Pre-trained Models
https://drive.google.com/drive/folders/1Jy0osy4LKdsE-N0MY2ZOhBHG0SHG-j_V?usp=sharing

## Known Limitations & Overfitting Analysis

### Why overfitting occurred
1. **Flat surface bias** — FPFH descriptors produce strong features on flat surfaces, causing RANSAC to match flat faces even when not adjacent
2. **Small dataset** — 17 fragments meant ML models were trained and tested on same data with no unseen validation
3. **PCA alignment side effect** — aligning by principal axes caused fragments to rotate into similar orientations artificially

### Fixes applied
1. **Planarity penalty** — flat surface matches down-weighted using local neighborhood eigenvalues
2. **5-fold cross validation** — more reliable accuracy estimate across folds
3. **Break surface exclusivity** — matched edges removed from future candidates
4. **Next step** — using 2D image data to validate 3D matches using carved surface appearance
