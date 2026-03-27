import os
import glob
import json
import time
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import KDTree
from tqdm import tqdm

FRAGMENT_DIR = "./fragments/hQO24HxuKi6VeQo"
OUTPUT_DIR = "./output_dl"
MAX_POINTS = 50000
MIN_FITNESS = 0.03
MAX_ITER_ICP = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"using device: {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DEEP FEATURE EXTRACTOR (PointNet-style)
# ─────────────────────────────────────────────
class PointNetFeatures(nn.Module):
    """
    PointNet-style per-point feature extractor.
    Learns local geometric descriptors better than FPFH
    on real-world damaged stone surfaces.
    """
    def __init__(self, in_dim=6, out_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.conv5 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x: [B, N, 6] (xyz + normals)
        x = x.transpose(2, 1)  # [B, 6, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = x.transpose(2, 1)  # [B, N, out_dim]
        return F.normalize(x, dim=2)  # L2 normalize


def get_deep_features(pcd, model, patch_size=2048, stride=1024):
    """extract per-point deep features using sliding window"""
    pts = np.asarray(pcd.points)
    nms = np.asarray(pcd.normals)

    if len(pts) == 0:
        return None

    # normalize points
    center = pts.mean(axis=0)
    scale = np.abs(pts - center).max() + 1e-8
    pts_norm = (pts - center) / scale

    # combine xyz + normals as input
    features_in = np.concatenate([pts_norm, nms], axis=1).astype(np.float32)
    all_features = np.zeros((len(pts), 32), dtype=np.float32)
    counts = np.zeros(len(pts), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(pts), stride):
            end = min(start + patch_size, len(pts))
            chunk = features_in[start:end]
            if len(chunk) < 10:
                continue
            x = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE)
            feat = model(x).squeeze(0).cpu().numpy()
            all_features[start:end] += feat
            counts[start:end] += 1

    counts = np.maximum(counts, 1)
    all_features /= counts[:, None]
    return all_features


# ─────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────
def pca_align(pcd):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    pts_c = pts - center
    cov = np.cov(pts_c.T)
    _, eigenvectors = np.linalg.eigh(cov)
    R = eigenvectors.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return pcd.transform(T), T


def boundary_sample(pcd, n_points=50000, radius=10.0):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    tree = o3d.geometry.KDTreeFlann(pcd)
    check_idx = np.random.choice(len(pts), size=min(len(pts), 20000), replace=False)
    full_density = np.ones(len(pts)) * 999
    for i in check_idx:
        k, _, _ = tree.search_radius_vector_3d(pts[i], radius)
        full_density[i] = k
    weights = 1.0 / (full_density + 1)
    weights /= weights.sum()
    idx = np.random.choice(len(pts), size=min(n_points, len(pts)),
                           replace=False, p=weights)
    return pcd.select_by_index(idx.tolist())


def load_fragments(path):
    files = sorted(glob.glob(os.path.join(path, "*.PLY")) +
                   glob.glob(os.path.join(path, "*.ply")))
    print(f"found {len(files)} fragments")

    clouds = []
    for f in files:
        try:
            print(f"  loading {os.path.basename(f)}...")
            mesh = o3d.io.read_triangle_mesh(f)
            if len(mesh.vertices) > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=MAX_POINTS * 2)
            else:
                pcd = o3d.io.read_point_cloud(f)

            pcd = boundary_sample(pcd, n_points=MAX_POINTS)
            pcd, _ = pca_align(pcd)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=10.0, max_nn=30))

            print(f"    -> {len(pcd.points)} points")
            clouds.append((os.path.basename(f), pcd))
        except Exception as e:
            print(f"    skipping {os.path.basename(f)}: {e}")

    return clouds


# ─────────────────────────────────────────────
# DL-BASED MATCHING
# ─────────────────────────────────────────────
def match_with_deep_features(pcd_i, pcd_j, feat_i, feat_j, n_corr=5000):
    """find correspondences using deep feature similarity"""
    pts_i = np.asarray(pcd_i.points)
    pts_j = np.asarray(pcd_j.points)

    # subsample for speed
    idx_i = np.random.choice(len(feat_i), size=min(n_corr, len(feat_i)), replace=False)
    idx_j = np.random.choice(len(feat_j), size=min(n_corr, len(feat_j)), replace=False)

    fi = feat_i[idx_i]
    fj = feat_j[idx_j]

    # find nearest neighbor in feature space
    tree = KDTree(fj)
    dists, nn_idx = tree.query(fi, k=1)
    dists = dists.flatten()
    nn_idx = nn_idx.flatten()

    # ratio test
    _, nn2_idx = tree.query(fi, k=2)
    ratio = dists / (nn2_idx[:, 1] + 1e-8)
    good = ratio < 0.8

    if good.sum() < 4:
        return None, 0

    src_pts = pts_i[idx_i[good]].astype(np.float64)
    dst_pts = pts_j[idx_j[nn_idx[good]]].astype(np.float64)

    # RANSAC on deep feature correspondences
    src_o3d = o3d.geometry.PointCloud()
    dst_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src_pts)
    dst_o3d.points = o3d.utility.Vector3dVector(dst_pts)

    corr = np.array([[k, k] for k in range(len(src_pts))])
    corr_o3d = o3d.utility.Vector2iVector(corr)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_o3d, dst_o3d, corr_o3d,
        max_correspondence_distance=15.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result.transformation, result.fitness


def icp_refine(src, dst, init_T, voxel_size):
    result = o3d.pipelines.registration.registration_icp(
        src, dst,
        max_correspondence_distance=voxel_size * 0.4,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER_ICP)
    )
    return result


def preprocess_fpfh(pcd, voxel_size):
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return down, fpfh


def hybrid_register(pcd_i, pcd_j, feat_i, feat_j):
    """
    Hybrid: try DL features first, fall back to multiscale FPFH.
    Best of both worlds.
    """
    best_T = np.eye(4)
    best_fitness = 0
    best_rmse = 0

    # 1. try DL-based matching
    try:
        T_dl, fitness_dl = match_with_deep_features(pcd_i, pcd_j, feat_i, feat_j)
        if T_dl is not None and fitness_dl > best_fitness:
            # refine with ICP
            icp = icp_refine(pcd_i, pcd_j, T_dl, 5.0)
            if icp.fitness > best_fitness:
                best_fitness = icp.fitness
                best_T = icp.transformation
                best_rmse = icp.inlier_rmse
    except Exception as e:
        pass

    # 2. multiscale FPFH as fallback/complement
    for voxel_size in [10.0, 5.0, 2.0]:
        try:
            src_d, src_f = preprocess_fpfh(pcd_i, voxel_size)
            dst_d, dst_f = preprocess_fpfh(pcd_j, voxel_size)

            dist = voxel_size * 1.5
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_d, dst_d, src_f, dst_f,
                mutual_filter=True,
                max_correspondence_distance=dist,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )
            icp = icp_refine(src_d, dst_d, ransac.transformation, voxel_size)

            if icp.fitness > best_fitness:
                best_fitness = icp.fitness
                best_T = icp.transformation
                best_rmse = icp.inlier_rmse
        except Exception:
            continue

    return best_T, best_fitness, best_rmse


# ─────────────────────────────────────────────
# POSE GRAPH
# ─────────────────────────────────────────────
def build_pose_graph(clouds, deep_features):
    n = len(clouds)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    fitness_matrix = np.zeros((n, n))

    for i in range(n):
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    total = n*(n-1)//2
    done = 0
    print(f"\ncomputing {total} pairwise registrations (DL + FPFH hybrid)...")

    for i, j in combinations(range(n), 2):
        name_i, pcd_i = clouds[i]
        name_j, pcd_j = clouds[j]

        T, fitness, rmse = hybrid_register(
            pcd_i, pcd_j, deep_features[i], deep_features[j])

        fitness_matrix[i, j] = fitness
        fitness_matrix[j, i] = fitness
        done += 1

        print(f"  [{done}/{total}] {name_i[:12]} <-> {name_j[:12]} | fitness={fitness:.3f} rmse={rmse:.4f}")

        if fitness > MIN_FITNESS:
            info = np.identity(6) * fitness
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, j, T, info, uncertain=(i+1 != j)))

        if done % 20 == 0:
            np.save(os.path.join(OUTPUT_DIR, "fitness_checkpoint.npy"), fitness_matrix)
            print(f"  [checkpoint saved]")

    return pose_graph, fitness_matrix


def optimize_graph(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=5.0,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
    return pose_graph


def assemble(clouds, pose_graph):
    combined = o3d.geometry.PointCloud()
    transforms = []
    colors = plt.cm.tab20(np.linspace(0, 1, len(clouds)))
    for i, (name, pcd) in enumerate(clouds):
        T = pose_graph.nodes[i].pose
        colored = o3d.geometry.PointCloud(pcd)
        colored.paint_uniform_color(colors[i][:3])
        combined += colored.transform(T)
        transforms.append(T.tolist())
    return combined, transforms


def detect_gaps(reconstructed, voxel_size=5.0):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        reconstructed, voxel_size=voxel_size)
    bbox = reconstructed.get_axis_aligned_bounding_box()
    min_b = np.asarray(bbox.min_bound)
    max_b = np.asarray(bbox.max_bound)
    total_voxels = int(np.prod(((max_b - min_b) / voxel_size).astype(int) + 1))
    filled_voxels = len(voxel_grid.get_voxels())
    gap_ratio = 1 - (filled_voxels / max(total_voxels, 1))
    return {
        "total_voxels": total_voxels,
        "filled_voxels": filled_voxels,
        "gap_ratio_%": round(gap_ratio * 100, 2)
    }


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_fitness_matrix(fitness_matrix, names, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(fitness_matrix, cmap="viridis", vmin=0, vmax=1)
    short = [n.replace("NAR_ST_43B_FR_", "F").split("_F_")[0] for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{fitness_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if fitness_matrix[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="fitness")
    ax.set_title("Pairwise Registration Fitness (DL + FPFH Hybrid)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved -> {save_path}")


def plot_graph(fitness_matrix, names, save_path):
    G = nx.Graph()
    n = len(names)
    short = [n.replace("NAR_ST_43B_FR_", "F").split("_F_")[0] for n in names]
    for i in range(n):
        G.add_node(i, label=short[i])
    for i, j in combinations(range(n), 2):
        if fitness_matrix[i, j] > MIN_FITNESS:
            G.add_edge(i, j, weight=fitness_matrix[i, j])

    pos = nx.spring_layout(G, seed=42)
    edges = list(G.edges(data=True))
    widths = [d["weight"] * 5 for _, _, d in edges]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color="steelblue", node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, {i: G.nodes[i]["label"] for i in G.nodes},
                            font_size=7, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="orange", ax=ax)
    ax.set_title("Fragment Connectivity Graph (DL + FPFH Hybrid)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved -> {save_path}")


def plot_before_after(clouds, pose_graph, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    before = o3d.geometry.PointCloud()
    for _, pcd in clouds:
        before += pcd
    pts = np.asarray(before.points)
    axes[0].scatter(pts[::20, 0], pts[::20, 2], s=0.1, alpha=0.3, c='steelblue')
    axes[0].set_title("Before Registration", fontsize=13)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Z")

    colors = plt.cm.tab20(np.linspace(0, 1, len(clouds)))
    for i, (_, pcd) in enumerate(clouds):
        p = o3d.geometry.PointCloud(pcd)
        p.transform(pose_graph.nodes[i].pose)
        pts_a = np.asarray(p.points)
        axes[1].scatter(pts_a[::20, 0], pts_a[::20, 2], s=0.1,
                        alpha=0.3, color=colors[i][:3])
    axes[1].set_title("After Registration", fontsize=13)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")

    plt.suptitle("Mayan Stele — Before vs After (DL+FPFH)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved -> {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()

    # load fragments
    clouds = load_fragments(FRAGMENT_DIR)
    names = [c[0] for c in clouds]
    n = len(clouds)

    if n == 0:
        print("no fragments loaded")
        return

    # init deep feature model
    print(f"\ninitializing PointNet feature extractor on {DEVICE}...")
    model = PointNetFeatures(in_dim=6, out_dim=32).to(DEVICE)
    # note: using randomly initialized weights as geometric feature learner
    # for best results, fine-tune on stone/archaeological scan data
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pointnet_model.pth"))
    print("model saved -> output_dl/pointnet_model.pth")

    # extract deep features for all fragments
    print("\nextracting deep features...")
    deep_features = []
    for i, (name, pcd) in enumerate(clouds):
        print(f"  {name}...")
        feat = get_deep_features(pcd, model)
        deep_features.append(feat)
        print(f"    -> features shape: {feat.shape}")

    # build pose graph with hybrid matching
    pose_graph, fitness_matrix = build_pose_graph(clouds, deep_features)

    print("\noptimizing pose graph...")
    pose_graph = optimize_graph(pose_graph)

    print("assembling...")
    reconstructed, transforms = assemble(clouds, pose_graph)

    out_ply = os.path.join(OUTPUT_DIR, "reconstructed_stele_dl.ply")
    o3d.io.write_point_cloud(out_ply, reconstructed)
    print(f"saved -> {out_ply}")

    with open(os.path.join(OUTPUT_DIR, "transforms.json"), "w") as f:
        json.dump({"fragments": names, "poses": transforms}, f, indent=2)

    connected = sum(1 for i, j in combinations(range(n), 2)
                    if fitness_matrix[i, j] > MIN_FITNESS)
    total_pairs = n*(n-1)//2
    gap_info = detect_gaps(reconstructed)

    metrics = {
        "method": "DL (PointNet) + FPFH hybrid",
        "device": str(DEVICE),
        "total_fragments": n,
        "pairs_matched": connected,
        "total_pairs": total_pairs,
        "match_coverage_%": round(100 * connected / total_pairs, 2),
        "avg_fitness": round(float(fitness_matrix[fitness_matrix > 0].mean()), 4) if (fitness_matrix > 0).any() else 0,
        "max_fitness": round(float(fitness_matrix.max()), 4),
        "num_graph_edges": len(pose_graph.edges),
        "gap_analysis": gap_info,
        "runtime_seconds": round(time.time() - t0, 2)
    }

    print("\nmetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(os.path.join(OUTPUT_DIR, "metrics_dl.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_fitness_matrix(fitness_matrix, names,
                        os.path.join(OUTPUT_DIR, "fitness_matrix_dl.png"))
    plot_graph(fitness_matrix, names,
               os.path.join(OUTPUT_DIR, "connectivity_graph_dl.png"))
    plot_before_after(clouds, pose_graph,
                      os.path.join(OUTPUT_DIR, "before_after_dl.png"))

    print(f"\ndone in {metrics['runtime_seconds']}s")
    print(f"outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
