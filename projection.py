# projection.py
import os
import time
import utils
import numpy as np

class Projection:

    """
    This is a parent class for all projection classes.
    It takes n-dimensional data and projects it into a 2D space.

    Currently only includes SOM, but can be extended to include
    more projection types, e.g. K-means
    """

    def __init__(self, input_data: np.ndarray, n_max_iterations: int, width: int, height: int):
        self.input_data = input_data
        self.height = height
        self.width = width
        self.n_max_iterations = n_max_iterations
        self.weights = self.generate_weights()

    def generate_weights(self) -> np.ndarray:
        """
        As described, there are i*j*k nodes, where:
            - i is the height of the SOM
            - j is the width of the SOM
            - k is the size of the input vector
        """
        if self.input_data.ndim != 2:
            raise ValueError("input_data must be a 2D array (samples x features).")

            # Validate SOM dimensions
        if not isinstance(self.height, int) or not isinstance(self.width, int):
            raise TypeError("height and width must be integers.")

        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive integers.")

        try:
            return np.random.random((self.height, self.width, self.input_data.shape[1]))
        except Exception as e:
            raise RuntimeError(f"Error generating weights: {e}")

    def file_name(self) -> None:
        """ Returns a string that describes the settings for the projection type """
        raise NotImplementedError

    def train(self) -> None:
        """ This is the main function that projects the data into the 2D space """
        raise NotImplementedError

    def run(self) -> None:
        """ Wrapper function to handle non-projection logic """

        # Start the timer
        start_time = time.time()

        # Run the SOM
        self.train()

        # End the timer
        end_time = time.time()
        run_time = round(end_time - start_time, 1)

        # Save the projection as a jpg file
        self.save_image(run_time)

    def save_image(self, run_time: float) -> None:
        """
        Saves the current projection (weights) as a JPG image.

        Args:
            run_time (float): The time taken to generate the projection,
                used in the filename.
        """

        # Generate a meaningful filename and save
        cwd = os.getcwd()
        file_name = self.file_name()
        path = cwd + f"/output/{file_name}, took {run_time} seconds.jpg"
        utils.output_image(self.weights, path)
