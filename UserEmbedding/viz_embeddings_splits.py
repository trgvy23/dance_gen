import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Optional: UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[Warn] umap-learn not installed; UMAP option will be disabled.")


def load_split(npz_path, split_name):
    """
    Load one split (train/test) from .npz.
    Expected keys: 'embeddings', 'dancer_labels', optional 'filenames'.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in data:
        raise KeyError(f"{npz_path} has no 'embeddings' key")
    if "dancer_labels" not in data and "all_labels" not in data:
        raise KeyError(f"{npz_path} has no 'dancer_labels' or 'all_labels' key")

    embs = data["embeddings"]
    if "dancer_labels" in data:
        labels = data["dancer_labels"]
    else:
        labels = data["all_labels"]

    filenames = data.get("filenames", None)
    split = np.array([split_name] * embs.shape[0])

    print(f"[Load {split_name}] {npz_path}")
    print(f"  embeddings: {embs.shape}")
    print(f"  labels:     {labels.shape}")
    if filenames is not None:
        print(f"  filenames:  {len(filenames)}")

    return embs, labels, filenames, split


def run_tsne(embs, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    z = tsne.fit_transform(embs)
    return z


def run_umap(embs, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    if not HAS_UMAP:
        raise RuntimeError("UMAP requested but umap-learn is not installed.")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    z = reducer.fit_transform(embs)
    return z


def plot_marker_by_split(
    z, dancer_labels, split_labels,
    title="Embeddings (color = dancer, marker = split)",
    save_path=None, max_points=None,
):
    """
    One panel:
      - color = dancer ID
      - marker = train/test
    """
    if max_points is not None and z.shape[0] > max_points:
        print(f"[Info] Subsampling {max_points}/{z.shape[0]} for plotting")
        idx = np.random.choice(z.shape[0], size=max_points, replace=False)
        z = z[idx]
        dancer_labels = dancer_labels[idx]
        split_labels = split_labels[idx]

    dancer_labels = np.array(dancer_labels)
    split_labels = np.array(split_labels)

    unique_dancers = np.unique(dancer_labels)
    unique_splits = np.unique(split_labels)

    # colormap per dancer
    cmap = plt.cm.get_cmap("tab20", len(unique_dancers))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_dancers)}

    # precompute color per point
    colors = np.array([label_to_color[lab] for lab in dancer_labels])

    markers = {
        "train": "o",
        "test": "^",
    }
    # fallback if your split labels are 0/1 etc.
    for s in unique_splits:
        if s not in markers:
            markers[s] = "o"

    plt.figure(figsize=(8, 8))
    for s in unique_splits:
        mask = split_labels == s
        if not np.any(mask):
            continue
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=6,
            marker=markers[s],
            c=colors[mask],
            alpha=0.8,
            edgecolors="none",
            label=str(s),
        )

    plt.title(title)
    plt.legend(title="Split", loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Save] {save_path}")
    else:
        plt.show()
    plt.close()


def plot_two_panels_by_split(
    z, dancer_labels, split_labels,
    title="Embeddings by split (same colors = same dancer)",
    save_path=None, max_points=None,
):
    """
    Two panels: left = train, right = test
      - color = dancer ID
    """
    if max_points is not None and z.shape[0] > max_points:
        print(f"[Info] Subsampling {max_points}/{z.shape[0]} for plotting")
        idx = np.random.choice(z.shape[0], size=max_points, replace=False)
        z = z[idx]
        dancer_labels = dancer_labels[idx]
        split_labels = split_labels[idx]

    dancer_labels = np.array(dancer_labels)
    split_labels = np.array(split_labels)
    unique_dancers = np.unique(dancer_labels)

    cmap = plt.cm.get_cmap("tab20", len(unique_dancers))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_dancers)}

    splits = ["train", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax, s in zip(axes, splits):
        mask_split = split_labels == s
        for lab in unique_dancers:
            mask = mask_split & (dancer_labels == lab)
            if not np.any(mask):
                continue
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=6,
                c=[label_to_color[lab]],
                alpha=0.8,
                edgecolors="none",
            )
        ax.set_title(f"{s} split")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Save] {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "umap"])
    parser.add_argument("--max_points", type=int, default=8000,
                        help="Max points for plotting (-1 = no limit)")
    parser.add_argument("--out_prefix", type=str, default="emb_viz",
                        help="Prefix for saved figures")
    args = parser.parse_args()

    train_embs, train_labels, _, train_split = load_split(args.train_npz, "train")
    test_embs, test_labels, _, test_split = load_split(args.test_npz, "test")

    # concat
    embs = np.concatenate([train_embs, test_embs], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)
    split_labels = np.concatenate([train_split, test_split], axis=0)

    print(f"[Combined] embs:   {embs.shape}")
    print(f"[Combined] labels: {labels.shape}")
    print(f"[Combined] splits: {split_labels.shape}")

    # dimensionality reduction
    if args.method == "tsne":
        z = run_tsne(embs)
        method_name = "t-SNE"
    else:
        z = run_umap(embs)
        method_name = "UMAP"

    max_points = None if args.max_points < 0 else args.max_points

    # 1) Single panel: color = dancer, marker = split
    plot_marker_by_split(
        z,
        labels,
        split_labels,
        title=f"{method_name}: color=dancer, marker=train/test",
        save_path=f"{args.out_prefix}_{args.method}_marker_split.png",
        max_points=max_points,
    )

    # 2) Two panels: left=train, right=test
    plot_two_panels_by_split(
        z,
        labels,
        split_labels,
        title=f"{method_name}: train vs test",
        save_path=f"{args.out_prefix}_{args.method}_two_panels.png",
        max_points=max_points,
    )


if __name__ == "__main__":
    main()

