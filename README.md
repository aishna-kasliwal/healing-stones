# Healing Stones — Mayan Stele Fragment Reconstruction

Reconstructing a fragmented Mayan stele from 17 3D scanned fragments using geometric feature matching and pose graph optimization.

## Results
| Run | Method | Coverage |
|-----|--------|----------|
| 1 | Basic RANSAC+ICP | 12.5% |
| 2 | Multi-scale RANSAC+ICP | 36.03% |
| 3 | Boundary sampling + PCA + 6-scale | 51.47% |

Gap ratio: 93.33% — reflects real missing material from the original stele

## Setup
```bash
conda create -n healing python=3.10 -y
conda activate healing
pip install -r requirements.txt
```

## Run
```bash
# 3D reconstruction
python reconstruct.py

# 2D reconstruction
python reconstruct_2d.py
```

## Output
- `output/reconstructed_stele.ply` — assembled 3D stele
- `output/metrics.json` — registration metrics
- `output/fitness_matrix.png` — pairwise fitness heatmap
- `output/connectivity_graph.png` — fragment connectivity graph
- `output/before_after.png` — before vs after registration
- `output_2d/fragments_grid.png` — all 12 fragments visualized
- `output_2d/match_matrix_2d.png` — 2D pairwise match scores
- `output_2d/reconstruction_layout.png` — estimated 2D layout
