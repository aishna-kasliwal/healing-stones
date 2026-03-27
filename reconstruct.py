import os
import glob
import json
import time
import pickle
import warnings
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

warnings.filterwarnings('ignore')

FRAGMENT_DIR = "./fragments/hQO24HxuKi6VeQo"
OUTPUT_DIR   = "./output_ml"
MAX_POINTS   = 50000
MIN_FITNESS  = 0.03
MAX_ITER_ICP = 200
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
print(f"using: {DEVICE}")


# ── data loading ──────────────────────────────────────────────

def pca_align(pcd):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    _, vecs = np.linalg.eigh(cov)
    R = vecs.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return pcd.transform(T), T


def boundary_sample(pcd, n_points=50000, radius=10.0):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    tree = o3d.geometry.KDTreeFlann(pcd)
    check = np.random.choice(len(pts), size=min(len(pts), 20000), replace=False)
    density = np.ones(len(pts)) * 999
    for i in check:
        k, _, _ = tree.search_radius_vector_3d(pts[i], radius)
        density[i] = k
    w = 1.0 / (density + 1)
    w /= w.sum()
    idx = np.random.choice(len(pts), size=min(n_points, len(pts)), replace=False, p=w)
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
            pcd = mesh.sample_points_uniformly(number_of_points=MAX_POINTS * 2) \
                if len(mesh.vertices) > 0 else o3d.io.read_point_cloud(f)
            pcd = boundary_sample(pcd, n_points=MAX_POINTS)
            pcd=augment_pointcloud(pcd, n_aug=2)[0]
            pcd, _ = pca_align(pcd)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
            print(f"    -> {len(pcd.points)} points")
            clouds.append((os.path.basename(f), pcd))
        except Exception as e:
            print(f"    skipping: {e}")
    return clouds


# ── geometric features ────────────────────────────────────────

def extract_geometric_features(pcd, radius=15.0):
    pts = np.asarray(pcd.points)
    nms = np.asarray(pcd.normals)
    tree = o3d.geometry.KDTreeFlann(pcd)
    feats = np.zeros((len(pts), 8), dtype=np.float32)

    for i in range(len(pts)):
        k, idx, dists = tree.search_radius_vector_3d(pts[i], radius)
        if k < 4:
            continue
        idx = list(idx)
        nb = pts[idx]
        nb_nms = nms[idx]
        cov = np.cov((nb - nb.mean(axis=0)).T)
        ev = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1] + 1e-10
        e1, e2, e3 = ev
        mn = nb_nms.mean(axis=0)
        mn /= np.linalg.norm(mn) + 1e-10
        feats[i] = [
            (e1 - e2) / e1,                          # linearity
            (e2 - e3) / e1,                          # planarity
            e3 / e1,                                 # sphericity
            e3 / (e1 + e2 + e3),                     # curvature
            np.mean(np.abs(nb_nms @ nms[i])),        # normal consistency
            k / (np.pi * radius**2 + 1e-10),         # density
            1 - np.abs(nms[i] @ mn),                 # normal deviation
            np.sqrt(dists[1]) if k > 1 else 0        # nn distance
        ]
    return feats


def generate_break_labels(pcd, feats):
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-10)

    score = (norm(feats[:, 3]) * 0.4 +
             norm(1 - feats[:, 4]) * 0.3 +
             norm(feats[:, 6]) * 0.2 +
             norm(1 - feats[:, 1]) * 0.1)

    labels = (score > np.percentile(score, 70)).astype(int)
    return labels, score


# ── model 1: break surface classifier ────────────────────────

def train_break_classifier(all_feats, all_labels):
    print("\ntraining break surface classifier...")
    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)

    n = min((y == 1).sum(), (y == 0).sum())
    X_b = resample(X[y == 1], n_samples=n, random_state=42)
    X_o = resample(X[y == 0], n_samples=n, random_state=42)
    X_bal = np.vstack([X_b, X_o])
    y_bal = np.array([1]*n + [0]*n)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_bal)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_bal, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = (y_pred == y_te).mean()

    print(f"  accuracy: {acc*100:.2f}%")
    print(classification_report(y_te, y_pred, target_names=['original', 'break']))

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['original', 'break'],
                yticklabels=['original', 'break'], ax=ax)
    ax.set_title("break surface classifier — confusion matrix")
    ax.set_ylabel("true")
    ax.set_xlabel("predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    names = ['linearity', 'planarity', 'sphericity', 'curvature',
             'normal_consistency', 'density', 'normal_dev', 'nn_dist']
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(imp)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([names[i] for i in idx], rotation=45, ha='right')
    ax.set_title("feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    return clf, scaler, acc


# ── model 2: fragment relationship predictor ──────────────────

class RelationshipPredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fragment_descriptor(pcd, feats, mask):
    bf = feats[mask == 1] if mask.sum() > 0 else feats
    return np.concatenate([bf.mean(axis=0), bf.std(axis=0)])


def train_relationship_predictor(clouds, all_feats, masks, init_fitness):
    print("\ntraining fragment relationship predictor...")
    n = len(clouds)
    pairs, labels = [], []

    for i, j in combinations(range(n), 2):
        di = fragment_descriptor(clouds[i][1], all_feats[i], masks[i])
        dj = fragment_descriptor(clouds[j][1], all_feats[j], masks[j])
        pairs.append(np.concatenate([di, dj]))
        labels.append(1.0 if init_fitness[i, j] > 0.1 else 0.0)

    X = np.array(pairs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_s = min(len(pos), len(neg))

    if n_s == 0:
        print("  not enough pairs, skipping")
        return None, None

    idx = np.concatenate([
        np.random.choice(pos, n_s, replace=False),
        np.random.choice(neg, n_s, replace=False)
    ])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    sp = int(0.8 * len(X))

    X_tr = torch.FloatTensor(X[:sp]).to(DEVICE)
    y_tr = torch.FloatTensor(y[:sp]).to(DEVICE)
    X_te = torch.FloatTensor(X[sp:]).to(DEVICE)
    y_te = torch.FloatTensor(y[sp:]).to(DEVICE)

    model = RelationshipPredictor(in_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    crit = nn.BCELoss()

    losses, accs = [], []
    print("  epoch | loss   | val_acc")
    for ep in range(100):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            acc = ((model(X_te) > 0.5).float() == y_te).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
        if (ep + 1) % 20 == 0:
            print(f"  {ep+1:5d} | {loss.item():.4f} | {acc*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses)
    axes[0].set_title("training loss")
    axes[0].set_xlabel("epoch")
    axes[1].plot(accs)
    axes[1].set_title("val accuracy")
    axes[1].set_xlabel("epoch")
    plt.suptitle("relationship predictor training")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  final val accuracy: {accs[-1]*100:.2f}%")

    return model, scaler


# ── registration ──────────────────────────────────────────────

def preprocess_fpfh(pcd, voxel_size):
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return down, fpfh


def icp_refine(src, dst, init_T, voxel_size):
    return o3d.pipelines.registration.registration_icp(
        src, dst,
        max_correspondence_distance=voxel_size * 0.4,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER_ICP)
    )


def multiscale_register(src, dst):
    best_fitness, best_T, best_rmse = 0, np.eye(4), 0
    for vs in [15.0, 10.0, 7.0, 5.0, 3.0, 2.0]:
        try:
            sd, sf = preprocess_fpfh(src, vs)
            dd, df = preprocess_fpfh(dst, vs)
            dist = vs * 1.5
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                sd, dd, sf, df, mutual_filter=True,
                max_correspondence_distance=dist,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )
            icp = icp_refine(sd, dd, ransac.transformation, vs)
            if icp.fitness > 0.3:
                fine = icp_refine(src, dst, icp.transformation, vs * 0.5)
                if fine.fitness > icp.fitness:
                    icp = fine
            if icp.fitness > best_fitness:
                best_fitness, best_T, best_rmse = icp.fitness, icp.transformation, icp.inlier_rmse
        except Exception:
            continue
    return best_T, best_fitness, best_rmse


# ── pose graph ────────────────────────────────────────────────

def build_pose_graph(clouds, ml_priority):
    n = len(clouds)
    pg = o3d.pipelines.registration.PoseGraph()
    fm = np.zeros((n, n))

    for i in range(n):
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    pairs = sorted(combinations(range(n), 2),
                   key=lambda p: ml_priority.get(p, 0), reverse=True)
    total, done = len(pairs), 0

    print(f"\ncomputing {total} registrations (ml-prioritized)...")
    for i, j in pairs:
        ni, pi = clouds[i]
        nj, pj = clouds[j]
        T, fitness, rmse = multiscale_register(pi, pj)
        fm[i, j] = fm[j, i] = fitness
        done += 1
        print(f"  [{done}/{total}] {ni[:12]} <-> {nj[:12]} | fitness={fitness:.3f} ml={ml_priority.get((i,j),0):.3f}")

        if fitness > MIN_FITNESS:
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                i, j, T, np.identity(6) * fitness, uncertain=(i+1 != j)))

        if done % 20 == 0:
            np.save(os.path.join(OUTPUT_DIR, "checkpoint.npy"), fm)
            print("  [checkpoint saved]")

    return pg, fm


def optimize_graph(pg):
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=5.0,
            edge_prune_threshold=0.25,
            reference_node=0
        )
    )
    return pg


def assemble(clouds, pg):
    combined, transforms = o3d.geometry.PointCloud(), []
    colors = plt.cm.tab20(np.linspace(0, 1, len(clouds)))
    for i, (name, pcd) in enumerate(clouds):
        T = pg.nodes[i].pose
        c = o3d.geometry.PointCloud(pcd)
        c.paint_uniform_color(colors[i][:3])
        combined += c.transform(T)
        transforms.append(T.tolist())
    return combined, transforms


def detect_gaps(pcd, voxel_size=5.0):
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    bb = pcd.get_axis_aligned_bounding_box()
    total = int(np.prod(((np.asarray(bb.max_bound) - np.asarray(bb.min_bound)) / voxel_size).astype(int) + 1))
    filled = len(vg.get_voxels())
    return {"total_voxels": total, "filled_voxels": filled,
            "gap_ratio_%": round((1 - filled/max(total,1)) * 100, 2)}


# ── plots ─────────────────────────────────────────────────────

def plot_fitness_matrix(fm, names, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(fm, cmap="viridis", vmin=0, vmax=1)
    short = [n.replace("NAR_ST_43B_FR_", "F").split("_F_")[0] for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{fm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if fm[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="fitness")
    ax.set_title("pairwise registration fitness")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved -> {path}")


def plot_graph(fm, names, path):
    G = nx.Graph()
    n = len(names)
    short = [n.replace("NAR_ST_43B_FR_", "F").split("_F_")[0] for n in names]
    for i in range(n):
        G.add_node(i, label=short[i])
    for i, j in combinations(range(n), 2):
        if fm[i, j] > MIN_FITNESS:
            G.add_edge(i, j, weight=fm[i, j])

    pos = nx.spring_layout(G, seed=42)
    edges = list(G.edges(data=True))
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color="steelblue", node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, {i: G.nodes[i]["label"] for i in G.nodes},
                            font_size=7, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, width=[d["weight"]*5 for _,_,d in edges],
                           edge_color="orange", ax=ax)
    ax.set_title("fragment connectivity graph")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved -> {path}")


def plot_before_after(clouds, pg, path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    before = o3d.geometry.PointCloud()
    for _, pcd in clouds:
        before += pcd
    pts = np.asarray(before.points)
    axes[0].scatter(pts[::20, 0], pts[::20, 2], s=0.1, alpha=0.3, c='steelblue')
    axes[0].set_title("before registration")

    colors = plt.cm.tab20(np.linspace(0, 1, len(clouds)))
    for i, (_, pcd) in enumerate(clouds):
        p = o3d.geometry.PointCloud(pcd)
        p.transform(pg.nodes[i].pose)
        pa = np.asarray(p.points)
        axes[1].scatter(pa[::20, 0], pa[::20, 2], s=0.1, alpha=0.3, color=colors[i][:3])
    axes[1].set_title("after registration")

    plt.suptitle("mayan stele — before vs after (ml-guided)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved -> {path}")


def plot_break_surfaces(clouds, masks, path):
    n = len(clouds)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows*4),
                              subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    for i, (name, pcd) in enumerate(clouds):
        pts = np.asarray(pcd.points)
        m = masks[i]
        axes[i].scatter(pts[m==0,0], pts[m==0,1], pts[m==0,2],
                        c='steelblue', s=0.1, alpha=0.3)
        axes[i].scatter(pts[m==1,0], pts[m==1,1], pts[m==1,2],
                        c='red', s=0.1, alpha=0.5)
        axes[i].set_title(name.replace("NAR_ST_43B_FR_","F").split("_F_")[0], fontsize=8)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.suptitle("break surface detection (red = break)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"saved -> {path}")


# ── main ──────────────────────────────────────────────────────

def main():
    t0 = time.time()

    clouds = load_fragments(FRAGMENT_DIR)
    names = [c[0] for c in clouds]
    n = len(clouds)
    if n == 0:
        print("no fragments found")
        return

    # extract geometric features
    print("\nextracting geometric features...")
    all_feats, all_masks = [], []
    for name, pcd in clouds:
        print(f"  {name}...")
        feats = extract_geometric_features(pcd)
        labels, _ = generate_break_labels(pcd, feats)
        all_feats.append(feats)
        all_masks.append(labels)
        print(f"    break points: {labels.sum()}/{len(labels)} ({labels.mean()*100:.1f}%)")

    # train break classifier
    clf, scaler, clf_acc = train_break_classifier(all_feats, all_masks)

    # predict break surfaces
    print("\npredicting break surfaces...")
    refined_masks = []
    for i, feats in enumerate(all_feats):
        pred = clf.predict(scaler.transform(feats))
        refined_masks.append(pred)
        print(f"  {names[i]}: {pred.sum()} break points")

    with open(os.path.join(OUTPUT_DIR, "models", "break_classifier.pkl"), "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    print("saved -> models/break_classifier.pkl")

    # quick initial registration pass for relationship predictor
    print("\ninitial registration pass...")
    init_fm = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        try:
            sd, sf = preprocess_fpfh(clouds[i][1], 10.0)
            dd, df = preprocess_fpfh(clouds[j][1], 10.0)
            r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                sd, dd, sf, df, mutual_filter=False,
                max_correspondence_distance=15.0,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(15.0)],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 100)
            )
            init_fm[i, j] = init_fm[j, i] = r.fitness
            print(f"  [{i}<->{j}] fitness={r.fitness:.3f}")
        except Exception:
            pass

    # train relationship predictor
    rel_model, rel_scaler = train_relationship_predictor(
        clouds, all_feats, refined_masks, init_fm)

    if rel_model is not None:
        torch.save(rel_model.state_dict(),
                   os.path.join(OUTPUT_DIR, "models", "relationship_predictor.pth"))
        print("saved -> models/relationship_predictor.pth")

    # get ml priority scores
    print("\ngenerating ml priority scores...")
    ml_priority = {}
    pair_list = list(combinations(range(n), 2))

    if rel_model is not None:
        descs = [fragment_descriptor(clouds[i][1], all_feats[i], refined_masks[i])
                 for i in range(n)]
        pf = np.array([np.concatenate([descs[i], descs[j]]) for i, j in pair_list], dtype=np.float32)
        pf = rel_scaler.transform(pf)
        rel_model.eval()
        with torch.no_grad():
            scores = rel_model(torch.FloatTensor(pf).to(DEVICE)).cpu().numpy()
        for (i, j), s in zip(pair_list, scores):
            ml_priority[(i, j)] = ml_priority[(j, i)] = float(s)
    else:
        for i, j in pair_list:
            ml_priority[(i, j)] = 0.5

    # full ml-guided registration
    pg, fm = build_pose_graph(clouds, ml_priority)

    print("\noptimizing pose graph...")
    pg = optimize_graph(pg)

    print("assembling...")
    reconstructed, transforms = assemble(clouds, pg)

    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "reconstructed_stele.ply"), reconstructed)
    with open(os.path.join(OUTPUT_DIR, "transforms.json"), "w") as f:
        json.dump({"fragments": names, "poses": transforms}, f, indent=2)

    connected = sum(1 for i, j in combinations(range(n), 2) if fm[i, j] > MIN_FITNESS)
    total_pairs = n*(n-1)//2

    metrics = {
        "method": "random forest + neural network + ransac + icp",
        "device": str(DEVICE),
        "total_fragments": n,
        "break_classifier_accuracy_%": round(clf_acc * 100, 2),
        "pairs_matched": connected,
        "total_pairs": total_pairs,
        "match_coverage_%": round(100 * connected / total_pairs, 2),
        "avg_fitness": round(float(fm[fm > 0].mean()), 4) if (fm > 0).any() else 0,
        "max_fitness": round(float(fm.max()), 4),
        "num_graph_edges": len(pg.edges),
        "gap_analysis": detect_gaps(reconstructed),
        "runtime_seconds": round(time.time() - t0, 2)
    }

    print("\nmetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(os.path.join(OUTPUT_DIR, "metrics_ml.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_break_surfaces(clouds, refined_masks,
                        os.path.join(OUTPUT_DIR, "break_surfaces.png"))
    plot_fitness_matrix(fm, names,
                        os.path.join(OUTPUT_DIR, "fitness_matrix.png"))
    plot_graph(fm, names,
               os.path.join(OUTPUT_DIR, "connectivity_graph.png"))
    plot_before_after(clouds, pg,
                      os.path.join(OUTPUT_DIR, "before_after.png"))

    print(f"\ndone in {metrics['runtime_seconds']}s")
    print(f"outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()


def augment_pointcloud(pcd, n_aug=2):
    augmented = [pcd]
    for _ in range(n_aug):
        R = pcd.get_rotation_matrix_from_xyz(
            np.random.uniform(-np.pi/8, np.pi/8, 3))
        aug = o3d.geometry.PointCloud(pcd)
        aug.rotate(R, center=aug.get_center())
        pts = np.asarray(aug.points)
        pts += np.random.normal(0, 0.5, pts.shape)
        aug.points = o3d.utility.Vector3dVector(pts)
        augmented.append(aug)
    return augmented
