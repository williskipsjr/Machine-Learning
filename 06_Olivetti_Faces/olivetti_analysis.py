"""Olivetti faces analysis

This script loads the Olivetti faces dataset, splits it stratified into
train/val/test sets, runs KMeans clustering and visualizations, trains a
classifier baseline, uses KMeans as a dimensionality reducer (cluster
centroid distances as features), and evaluates performance.

Usage:
    python olivetti_analysis.py --outdir output

"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def load_and_split(test_size=0.2, val_size=0.2, random_state=42):
    # Fetch dataset
    data = fetch_olivetti_faces()
    X = data.data  # shape (400, 4096)
    y = data.target  # shape (400,)

    # We'll first split off train+val vs test using stratified sampling
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss1.split(X, y))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]

    # Now split trainval into train and val
    # compute val proportion relative to trainval
    rel_val_size = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state)
    train_idx, val_idx = next(sss2.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    print("Split sizes: ", X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test, data


def show_cluster_examples(X, labels, n_clusters=10, outdir=None, imgs_per_cluster=5):
    # X expected in original 4096-vector space. Recreate 64x64 images
    h = w = 64
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        chosen = np.random.choice(idx, size=min(imgs_per_cluster, len(idx)), replace=False)
        fig, axes = plt.subplots(1, len(chosen), figsize=(len(chosen) * 2, 2))
        if len(chosen) == 1:
            axes = [axes]
        for ax, i in zip(axes, chosen):
            ax.imshow(X[i].reshape(h, w), cmap="gray")
            ax.axis("off")
        fig.suptitle(f"Cluster {c} ({len(idx)} images)")
        plt.tight_layout()
        plt.savefig(Path(outdir) / f"cluster_{c}.png")
        plt.close(fig)


def kmeans_and_visualize(X, n_clusters=20, outdir="output/kmeans", random_state=42):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(f"Fitting KMeans with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    # Save cluster centers as images
    centers = km.cluster_centers_
    h = w = 64
    fig, axes = plt.subplots(nrows=int(np.ceil(n_clusters / 5)), ncols=5, figsize=(10, 2 * int(np.ceil(n_clusters / 5))))
    axes = axes.flatten()
    for i in range(n_clusters):
        axes[i].imshow(centers[i].reshape(h, w), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(str(i))
    for j in range(n_clusters, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "kmeans_centers.png")
    plt.close(fig)

    # Save some example images per cluster
    show_cluster_examples(X, labels, n_clusters=n_clusters, outdir=outdir, imgs_per_cluster=5)
    return km, labels


def train_evaluate_classifier(X_train, y_train, X_val, y_val, C=1.0, max_iter=500):
    # Simple logistic regression baseline
    clf = LogisticRegression(max_iter=max_iter, C=C, multi_class="multinomial", solver="saga")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    return clf, acc, y_pred


def kmeans_features(X, km):
    # Return distances to cluster centers as features
    # sklearn KMeans has transform(X) returning distances to cluster centers
    return km.transform(X)


def run_experiments(outdir="output"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test, data = load_and_split()

    # Standardize features for classifier
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Baseline classifier on original features
    print("Training baseline classifier on original features...")
    clf_base, acc_base, _ = train_evaluate_classifier(X_train_s, y_train, X_val_s, y_val)

    results = {"baseline_acc": acc_base}

    # Try KMeans-based feature reductions
    best_acc = acc_base
    best_k = None
    for k in [5, 10, 20, 30, 40, 60, 80]:
        print(f"Experimenting with k={k} clusters for dimensionality reduction")
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(np.vstack([X_train, X_val, X_test]))
        # transform datasets into distance-to-centroid features
        X_train_k = kmeans_features(X_train, km)
        X_val_k = kmeans_features(X_val, km)

        # Option A: classifier on reduced features only
        scaler_k = StandardScaler()
        X_train_k_s = scaler_k.fit_transform(X_train_k)
        X_val_k_s = scaler_k.transform(X_val_k)
        _, acc_k_only, _ = train_evaluate_classifier(X_train_k_s, y_train, X_val_k_s, y_val)

        # Option B: append reduced features to original features
        X_train_aug = np.hstack([X_train_s, X_train_k_s])
        X_val_aug = np.hstack([X_val_s, X_val_k_s])
        _, acc_aug, _ = train_evaluate_classifier(X_train_aug, y_train, X_val_aug, y_val)

        results[f"k_{k}_reduced_only"] = acc_k_only
        results[f"k_{k}_augmented"] = acc_aug
        print(f"k={k}: reduced_only={acc_k_only:.4f}, augmented={acc_aug:.4f}")
        if acc_k_only > best_acc:
            best_acc = acc_k_only
            best_k = (k, "reduced_only")
        if acc_aug > best_acc:
            best_acc = acc_aug
            best_k = (k, "augmented")

    print("Best observed validation accuracy:", best_acc, best_k)
    results["best_val_acc"] = best_acc
    results["best_k_info"] = best_k

    # Fit final KMeans for visualization with a reasonable k (e.g., 40)
    km_vis, labels = kmeans_and_visualize(data.data, n_clusters=40, outdir=Path(outdir) / "kmeans_vis")

    # Evaluate final classifier on test set (using baseline classifier trained earlier)
    X_test_s = scaler.transform(X_test)
    y_test_pred = clf_base.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("Test accuracy of baseline classifier:", test_acc)
    results["test_acc_baseline"] = test_acc

    # Save results
    import json

    with open(Path(outdir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Experiments completed. Results saved to", outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="output", help="Output directory")
    args = parser.parse_args()
    run_experiments(outdir=args.outdir)


if __name__ == "__main__":
    main()
