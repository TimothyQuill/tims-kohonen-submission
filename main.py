import utils
from kohonen import SelfOrganisingMap
import numpy as np

if __name__ == '__main__':

    # Generate data
    input_data = np.random.random((10,3))

    # Using 10x10 SOM at 100 iterations...
    SelfOrganisingMap(input_data, 100, 10, 10).run()

    # ...with 200 iterations...
    SelfOrganisingMap(input_data, 200, 10, 10).run()

    # ...with 500 iterations
    SelfOrganisingMap(input_data, 500, 10, 10).run()

    # Using 100x100 SOM at 1000 iterations...
    SelfOrganisingMap(input_data, 1_000, 100, 100).run()