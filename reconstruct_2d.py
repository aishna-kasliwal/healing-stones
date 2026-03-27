import os
import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations

FRAGMENT_DIR = "./fragments_2d/2D"
OUTPUT_DIR = "./output_2d"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_fragments(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".png") or f.endswith(".jpg")])
    print(f"found {len(files)} fragments")
    
    fragments = []
    for f in files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        fragments.append({"name": f, "img": img, "gray": gray})
        print(f"  {f} -> {img.shape}")
    return fragments


def get_features(fragment):
    gray = fragment["gray"]
    
    sift = cv2.SIFT_create(nfeatures=2000)
    kp, desc = sift.detectAndCompute(gray, None)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return {
        "keypoints": kp,
        "descriptors": desc,
        "contour": contour,
        "hu": hu,
        "bbox": cv2.boundingRect(contour) if contour is not None else None
    }


def match_pair(fa, fb):
    scores = {}

    if fa["descriptors"] is not None and fb["descriptors"] is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(fa["descriptors"], fb["descriptors"], k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        scores["sift_matches"] = len(good)
        scores["sift_score"] = len(good) / max(len(fa["keypoints"]), len(fb["keypoints"]), 1)
    else:
        scores["sift_matches"] = 0
        scores["sift_score"] = 0.0

    hu_dist = np.linalg.norm(fa["hu"] - fb["hu"])
    scores["hu_distance"] = round(float(hu_dist), 4)

    if fa["contour"] is not None and fb["contour"] is not None:
        shape_sim = cv2.matchShapes(fa["contour"], fb["contour"], cv2.CONTOURS_MATCH_I1, 0)
        scores["shape_similarity"] = round(float(shape_sim), 4)
    else:
        scores["shape_similarity"] = 999.0

    scores["combined_score"] = round(
        scores["sift_score"] * 0.7 + (1 / (1 + scores["shape_similarity"])) * 0.3, 4)

    return scores


def get_transform(frag_a, frag_b, fa, fb):
    if fa["descriptors"] is None or fb["descriptors"] is None:
        return None, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(fa["descriptors"], fb["descriptors"], k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return None, len(good)

    pts_a = np.float32([fa["keypoints"][m.queryIdx].pt for m in good])
    pts_b = np.float32([fb["keypoints"][m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def build_layout(fragments, features, match_matrix):
    n = len(fragments)
    positions = {0: (0, 0)}
    placed = {0}

    while len(placed) < n:
        best_score = -1
        best_i, best_j, best_H = None, None, None

        for i in placed:
            for j in range(n):
                if j in placed:
                    continue
                score = match_matrix[i][j]["combined_score"]
                if score > best_score:
                    H, inliers = get_transform(fragments[i], fragments[j], features[i], features[j])
                    if H is not None:
                        best_score = score
                        best_i, best_j, best_H = i, j, H

        if best_j is None:
            remaining = [j for j in range(n) if j not in placed]
            best_j = remaining[0]
            positions[best_j] = (len(placed) * 100, 0)
        else:
            h, w = fragments[best_i]["img"].shape[:2]
            cx, cy = w // 2, h // 2
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, best_H)
            ox, oy = positions[best_i]
            positions[best_j] = (
                ox + int(transformed[0][0][0]) - cx,
                oy + int(transformed[0][0][1]) - cy
            )

        placed.add(best_j)

    return positions


def plot_grid(fragments, save_path):
    n = len(fragments)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()
    
    for i, frag in enumerate(fragments):
        img = frag["img"]
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        axes[i].set_title(frag["name"].replace("NAR_ST_43B_FR_TEST_", "Fragment ").replace(".png", ""), fontsize=9)
        axes[i].axis("off")
    
    for i in range(n, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle("Mayan Stele Fragments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_match_matrix(match_matrix, names, save_path):
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = match_matrix[i][j]["combined_score"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    
    short = [n.replace("NAR_ST_43B_FR_TEST_", "F").replace(".png", "") for n in names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i,j] < 0.5 else "white")
    
    plt.colorbar(im, ax=ax, label="Match Score")
    ax.set_title("Pairwise Match Scores")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reconstruction(fragments, positions, save_path):
    fig, ax = plt.subplots(figsize=(20, 16))
    
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    min_x, min_y = min(all_x), min(all_y)
    colors = plt.cm.tab20(np.linspace(0, 1, len(fragments)))

    for idx, (i, (x, y)) in enumerate(positions.items()):
        frag = fragments[i]
        h, w = frag["img"].shape[:2]
        px, py = x - min_x, y - min_y

        rect = patches.Rectangle((px, py), w, h, linewidth=2,
                                   edgecolor=colors[idx], facecolor="none", alpha=0.8)
        ax.add_patch(rect)
        
        label = frag["name"].replace("NAR_ST_43B_FR_TEST_", "F").replace(".png", "")
        ax.text(px + w//2, py + h//2, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=colors[idx])

    max_x = max(p[0] - min_x + fragments[i]["img"].shape[1] for i, p in positions.items())
    max_y = max(p[1] - min_y + fragments[i]["img"].shape[0] for i, p in positions.items())

    ax.set_xlim(-50, max_x + 50)
    ax.set_ylim(-50, max_y + 50)
    ax.invert_yaxis()
    ax.set_title("Reconstructed Stele Layout", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    t0 = time.time()

    fragments = load_fragments(FRAGMENT_DIR)
    names = [f["name"] for f in fragments]
    n = len(fragments)

    print("\nextracting features...")
    features = []
    for frag in fragments:
        feat = get_features(frag)
        features.append(feat)
        print(f"  {frag['name']} -> {len(feat['keypoints'])} keypoints")

    print(f"\nmatching {n*(n-1)//2} pairs...")
    match_matrix = [[None]*n for _ in range(n)]
    all_scores = []

    for i in range(n):
        for j in range(n):
            if i == j:
                match_matrix[i][j] = {"combined_score": 1.0, "sift_score": 1.0,
                                       "sift_matches": 0, "hu_distance": 0.0,
                                       "shape_similarity": 0.0}
            elif j > i:
                scores = match_pair(features[i], features[j])
                match_matrix[i][j] = scores
                match_matrix[j][i] = scores
                all_scores.append((i, j, scores["combined_score"]))
                print(f"  {names[i][-6:]} <-> {names[j][-6:]} | score={scores['combined_score']:.3f} sift={scores['sift_matches']}")

    print("\nbuilding layout...")
    positions = build_layout(fragments, features, match_matrix)

    top_pairs = sorted(all_scores, key=lambda x: x[2], reverse=True)[:5]
    metrics = {
        "total_fragments": n,
        "pairs_evaluated": n*(n-1)//2,
        "avg_match_score": round(float(np.mean([s[2] for s in all_scores])), 4),
        "top_5_matches": [
            {"pair": f"{names[i][-6:]} <-> {names[j][-6:]}", "score": round(float(s), 4)}
            for i, j, s in top_pairs
        ],
        "runtime_seconds": round(time.time() - t0, 2)
    }

    print("\nmetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(os.path.join(OUTPUT_DIR, "metrics_2d.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_grid(fragments, os.path.join(OUTPUT_DIR, "fragments_grid.png"))
    plot_match_matrix(match_matrix, names, os.path.join(OUTPUT_DIR, "match_matrix_2d.png"))
    plot_reconstruction(fragments, positions, os.path.join(OUTPUT_DIR, "reconstruction_layout.png"))

    print(f"\ndone in {metrics['runtime_seconds']}s")
    print(f"outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
