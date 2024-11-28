import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from MatrixVectorizer import MatrixVectorizer
from constants import *
from set_seed import set_seed
from evaluation import evaluate_all

set_seed(42)


def evaluate(
        truths_vectors=None,
        truths_matrices=None,
        predictions_matrices=None,
        predictions_vectors=None,
        include_diagonal=False,
        verbose=False,
        include_fid=False,
):
    """
    Evaluate the performance of centrality prediction on graph data.

    Parameters:
    - truths_vectors (numpy array): Ground truth in vectorized form. If this parameter is not provided, truths_matrices
    is required.
    - truths_matrices (numpy array): Ground truth in matrix form. If this parameter is not provided, truths_vectors is
    required.
    - predictions_matrices (numpy array, optional): Predicted adjacency matrices, if this parameter is not provided,
    predictions_vectors is required.
    - predictions_vectors (numpy array, optional): Predicted centrality values for nodes, if this parameter is not
    provided, predictions_matrices is required.
    - include_diagonal (bool, optional): Include diagonal elements in computations.
    - verbose (bool, optional): Print intermediate results if True.
    - include_fid (bool, optional): Include Frechet Inception Distance (FID) computation if True.
    - device (str, optional): Device to use for computations.

    Returns:
    - List containing [MAE, PCC, Jensen-Shannon Distance, Avg MAE Betweenness Centrality, Avg MAE Eigenvector
    Centrality, Avg MAE PageRank Centrality, Avg MAE Degree Centrality, Avg MAE Clustering Coefficient].
    If include_fid is True, FID is also included in the list.
    """

    # Check on optional inputs
    assert predictions_matrices is not None or predictions_vectors is not None
    assert truths_matrices is not None or truths_vectors is not None

    if predictions_matrices is None:
        # Apply anti-vectorization
        predictions_matrices = np.empty(
            (predictions_vectors.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
        )
        for i, prediction in enumerate(predictions_vectors):
            predictions_matrices[i] = MatrixVectorizer.anti_vectorize(
                prediction, HR_MATRIX_SIZE, include_diagonal
            )
    else:
        # Apply vectorization
        predictions_vectors = np.empty((predictions_matrices.shape[0], HR_ARRAY_SIZE))
        for i, prediction in enumerate(predictions_matrices):
            predictions_vectors[i] = MatrixVectorizer.vectorize(
                prediction, include_diagonal
            )

    # Apply anti-vectorization on truth
    if truths_matrices is None:
        truths_matrices = np.empty(
            (truths_vectors.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
        )
        for i, truth in enumerate(truths_vectors):
            truths_matrices[i] = MatrixVectorizer.anti_vectorize(
                truth, HR_MATRIX_SIZE, include_diagonal
            )
    else:
        truths_vectors = np.empty((truths_matrices.shape[0], HR_ARRAY_SIZE))
        for i, truth in enumerate(truths_matrices):
            truths_vectors[i] = MatrixVectorizer.vectorize(truth, include_diagonal)


    return evaluate_all(truths_matrices, predictions_matrices)
