from projection import Projection
import numpy as np
import utils


class SelfOrganisingMap(Projection):

    """
    This class implements the Self-Organising Map (SOM) algorithm.
    """

    def __init__(self, input_data: np.ndarray, n_max_iterations: int, width: int, height: int):
        super().__init__(input_data, n_max_iterations, width, height)

    @staticmethod
    def calculate_decay(time_constant: float, t: float) -> float:
        """
        Calculates the decay factor for the neighborhood radius at a given iteration.

        Args:
            time_constant (float): The time constant (λ), which controls the rate of decay.
            t (float): The current iteration step.

        Returns:
            float: The decay factor (0 < decay ≤ 1) to be applied to the neighborhood radius.
        """
        try:
            decay = np.exp(-t / time_constant)
            return decay
        except Exception as e:
            raise RuntimeError(f"Error calculating decay: {e}")

    @staticmethod
    def calculate_learning_rate(t: int, decay: float, a0: float = 0.1) -> float:
        """
        Calculates the learning rate at a given iteration based on an initial rate and decay.

        Args:
            t (int): The current iteration step.
            decay (float): The decay factor for this iteration (typically between 0 and 1).
            a0 (float, optional): The initial learning rate. Must be positive. Default is 0.1.

        Returns:
            float: The updated learning rate for iteration t.
        """
        if a0 <= 0 or decay < 0:
            raise ValueError("Learning rate and decay must be non-negative.")
        return a0 if t == 0 else a0 * decay

    @staticmethod
    def calculate_neighbourhood_radius(t0: float, decay: float) -> float:
        """
        Calculates the neighborhood radius at a given iteration based on initial radius and decay.

        Args:
            t0 (float): The initial neighborhood radius (must be positive).
            decay (float): The decay factor at the current iteration (typically between 0 and 1).

        Returns:
            float: The updated neighborhood radius.
        """
        if t0 <= 0 or decay < 0:
            raise ValueError("t0 and decay must be positive.")
        return t0 * decay

    def calculate_influence(self, bmu: tuple, radius) -> np.ndarray:
        """
        Computes the influence of the Best Matching Unit (BMU) over the surrounding nodes.

        Args:
            bmu (tuple): The coordinates (x, y) of the Best Matching Unit in the grid.
            radius (float): The current neighborhood radius. Must be positive.

        Returns:
            np.ndarray: A 2D array of influence values with shape (height, width),
                        where each value corresponds to the influence of the BMU
                        on that grid location.
        """
        if radius <= 0:
            raise ValueError("Radius must be positive.")

        try:
            # Create a grid of coordinates
            x, y = np.ogrid[0:self.height, 0:self.width]
            # Calculate the Euclidean distance between each node and the BMU
            di = (x - bmu[0]) ** 2 + (y - bmu[1]) ** 2
            # Compute the influence
            theta_t = np.exp(-di / (2 * radius ** 2))
        except Exception as e:
            raise RuntimeError(f"Error calculating influence: {e}")

        return theta_t

    def calculate_t0(self):
        """ The neighbourhood radius at time t0 """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers.")

        try:
            return max(self.width, self.height) / 2
        except Exception as e:
            raise RuntimeError(f"Error calculating t₀: {e}")

    def calculate_time_constant(self, t0: float) -> float:
        """
        Calculates the time constant (lambda) used for exponential decay of the neighborhood radius.

        Args:
            t0 (float): The initial neighborhood radius. Must be a positive value.

        Returns:
            float: The time constant (lambda), used in decay calculations.
        """
        if t0 <= 0:
            raise ValueError("t0 must be a positive value.")

        try:
            return self.n_max_iterations / np.log(t0)
        except ZeroDivisionError:
            raise ZeroDivisionError("log(t0) resulted in zero, causing division error.")
        except Exception as e:
            raise RuntimeError(f"Error calculating time constant: {e}")

    def file_name(self) -> str:
        """
        Generates a descriptive filename based on SOM configuration.

        Returns:
            str: A descriptive filename string (e.g., "10x10 SOM after 100 iters").
        """
        return f"{self.width}x{self.height} SOM after {self.n_max_iterations} iters"

    def find_bmu(self, input_sample: np.ndarray) -> tuple:
        """
        Finds the Best Matching Unit (BMU) for a given input vector.

        Args:
            input_sample (np.ndarray): A 1D input vector of shape (features,).

        Returns:
            tuple: A tuple (x, y) representing the 2D coordinates of the BMU in the SOM grid.
        """
        if input_sample.ndim != 1:
            raise ValueError("Input vector must be 1D.")

        try:
            # Compute the squared Euclidean distance between the input and all weights
            distances = utils.euclidean_distance(self.weights, input_sample)

            # Find the index of the minimum distance (BMU)
            flat_index = np.argmin(distances)

            # Convert flat index to 2D coordinates
            min_node = np.unravel_index(flat_index, (self.height, self.width))
            return min_node
        except Exception as e:
            raise RuntimeError(f"Error finding BMU: {e}")

    def train(self) -> None:
        """
        Trains the Self-Organising Map (SOM) by projecting high-dimensional input data into 2D space.
        """
        try:
            t0 = self.calculate_t0()
            time_constant = self.calculate_time_constant(t0)

            # For each iteration...
            for t in range(self.n_max_iterations):
                decay = self.calculate_decay(time_constant, t)
                radius = self.calculate_neighbourhood_radius(t0, decay)
                lr = self.calculate_learning_rate(t, decay)

                # For each sample...
                for vt in self.input_data:
                    bmu = self.find_bmu(vt)
                    influence = self.calculate_influence(bmu, radius)
                    self.update_weights(vt, influence, lr)
        except Exception as e:
            raise RuntimeError(f"Training failed at iteration {t}: {e}")

    def update_weights(self, input_sample: np.ndarray, influence: np.ndarray, lr: float) -> None:
        """
        Updates the weight vectors in the SOM grid based on the input vector, learning rate, and influence matrix.

        Args:
            input_sample (np.ndarray): The input vector (1D array) being used for this update.
            influence (np.ndarray): A 2D array of influence values (shape: height x width) for each node in the grid.
            lr (float): The current learning rate. Must be non-negative.
        """
        if input_sample.ndim != 1:
            raise ValueError("Input vector must be 1D.")
        if influence.shape != (self.height, self.width):
            raise ValueError("Influence matrix shape mismatch.")
        if lr < 0:
            raise ValueError("Learning rate must be non-negative.")

        try:
            # Reshape input_sample for broadcasting
            input_sample = np.reshape(input_sample, (1, 1, -1))
            # Update weights
            self.weights = (self.weights + lr * influence[..., np.newaxis]
                            * (input_sample - self.weights))
        except Exception as e:
            raise RuntimeError(f"Error updating weights: {e}")
