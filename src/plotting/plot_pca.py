from matplotlib.figure import Figure
import numpy as np
from sklearn.decomposition import PCA
from adjustText import adjust_text
import textwrap

import logging
logger = logging.getLogger(__name__)

def update_pca_plot(concepts: list[tuple[str, np.ndarray]], indicate_last_concept: bool = False) -> Figure|None:
    '''Projects the concepts into a 2D PCA plot, if at least two concepts are available. It returns a scatter plot with the concept names as text labels. If indicate_last_concept is True, the last concept is highlighted in red.'''
    if len(concepts) < 2:
        logger.info(f"Need at least 2 concepts for PCA, got {len(concepts)} -> no PCA plot generated")
        return None

    concept_name = [concept[0] for concept in concepts]
    embeddings = np.array([concept[1] for concept in concepts])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_ * 100

    fig = Figure()
    ax = fig.subplots()
    if indicate_last_concept:
        ax.scatter(coords[:-1, 0], coords[:-1, 1], alpha=0.7, s=30, color='navy', zorder=2)
        ax.scatter(coords[-1, 0], coords[-1, 1], alpha=0.7, s=30, color='red', zorder=2)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30, color='navy', zorder=2)

    texts = []
    for i, label in enumerate(concept_name):
        label = '\n'.join(textwrap.wrap(label, width=15))
        texts.append(ax.text(coords[i, 0], coords[i, 1], label,
            fontsize=5, ha='center', va='bottom', zorder=1,
            bbox=dict(boxstyle='round,pad=0.2', fc='lightgrey', ec='none'),fontweight='bold'))

    adjust_text(texts, ax=ax)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title("Concepts Learned")
    fig.tight_layout()
    return fig

