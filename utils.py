import matplotlib.pyplot as plt
import numpy as np
import os

"""
This file contains utility functions that are used in the main.py file.
"""

def euclidean_distance(v1: np.ndarray, v2: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes the squared Euclidean distance between two arrays along a specified axis.

    This function supports broadcasting and can handle:
      - A single vector vs. a batch/grid of vectors
      - Two arrays of matching shapes (e.g. pairwise comparisons)

    Args:
        v1 (np.ndarray): First input array (can be any shape).
        v2 (np.ndarray): Second input array (must be broadcastable with v1).
        axis (int): Axis along which to compute the distance (default: -1).

    Returns:
        np.ndarray: Array of squared distances.
    """
    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        raise TypeError("Both inputs must be NumPy ndarrays.")

    try:
        diff = v1 - v2
        return np.sum(diff ** 2, axis=axis)
    except ValueError as ve:
        raise ValueError(f"Shape mismatch: {ve}")
    except Exception as e:
        raise RuntimeError(f"Error computing Euclidean distance: {e}")


def output_image(data, path):
    """
    Saves a 2D or 3D array as an image to the specified file path.

    Args:
        data (np.ndarray): Image data to be saved.
        path (str): Destination file path for the image.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Display the image
    plt.imshow(data)
    plt.axis('off')

    # Save the image to a file
    plt.savefig(path)
    plt.close()
