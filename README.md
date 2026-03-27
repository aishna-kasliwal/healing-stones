# Healing Stones — Mayan Stele Fragment Reconstruction

## What this does
Takes 17 3D scanned fragments of a Mayan stele (.PLY files) and reconstructs them into a single object using machine learning and geometric registration.

## How to run

### Setup
```bash
conda create -n healing python=3.10 -y
conda activate healing
pip install -r requirements.txt
```

### 3D reconstruction (main)
```bash
python reconstruct.py
```

### 2D reconstruction
```bash
python reconstruct_2d.py
```

Both scripts run fully automated with no user intervention. Works on any .PLY or .OBJ files — just change `FRAGMENT_DIR` at the top of `reconstruct.py`.

## Pipeline (reconstruct.py)

1. Load .PLY fragments → sample points → PCA align → boundary-aware sampling
2. Extract geometric features per point (curvature, planarity, normal consistency etc.)
3. Train Random Forest to classify break surfaces vs original stone surface
4. Run initial RANSAC registration pass to generate training data
5. Train Neural Network to predict which fragment pairs are neighbors
6. Use ML scores to prioritize registration order
7. Run multi-scale RANSAC + ICP registration across all fragment pairs (6 scales)
8. Optimize global pose graph
9. Assemble fragments + detect missing material regions

## Pipeline (reconstruct_2d.py)

1. Load PNG fragments
2. Extract SIFT keypoints + contour shape descriptors
3. Match all fragment pairs
4. Estimate layout via homography
5. Output match scores + reconstructed layout

## Results

| Metric | Value |
|--------|-------|
| Break surface classifier accuracy | 97.57% |
| Match coverage | 51.47% |
| Max registration fitness | 1.0 |
| Gap ratio (missing material) | 94.4% |
| Total fragments | 17 |

## Outputs
All saved automatically to `output_ml/` and `output_2d/`:
- `reconstructed_stele.ply` — final assembled stele
- `metrics_ml.json` — all metrics
- `break_surfaces.png` — ML-detected break surfaces (red)
- `confusion_matrix.png` — classifier performance
- `training_curves.png` — neural network training
- `feature_importance.png` — which geometric features matter most
- `fitness_matrix.png` — pairwise registration heatmap
- `connectivity_graph.png` — fragment connectivity
- `before_after.png` — before vs after registration

## Pre-trained Models

Available on Google Drive: https://drive.google.com/drive/folders/1Jy0osy4LKdsE-N0MY2ZOhBHG0SHG-j_V?usp=sharing

- `break_classifier.pkl` — Random Forest break surface classifier
- `relationship_predictor.pth` — Neural Network fragment relationship predictor
- `vertical_order_net.pth` — Neural Network vertical ordering predictor
