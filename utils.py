import matplotlib.pyplot as plt
import numpy as np
import os

"""
This file contains utility functions that are used in the main.py file.
"""

def euclidean_distance(input_vec: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Computes the squared Euclidean distance between a 1D input vector and each node's weight vector in a 3D grid.

    Args:
        input_vec (np.ndarray): A 1D input vector of shape (k,)
        weights (np.ndarray): A 3D weight grid of shape (height, width, k)

    Returns:
        np.ndarray: A 2D array of distances with shape (height, width)
    """
    if input_vec.ndim != 1:
        raise ValueError("input_vec must be a 1D array.")
    if weights.ndim != 3 or weights.shape[2] != input_vec.shape[0]:
        raise ValueError("weights must be a 3D array with shape (height, width, input_dim).")

    return np.sum((weights - input_vec) ** 2, axis=2)


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
