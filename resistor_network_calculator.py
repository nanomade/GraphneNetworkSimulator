import time
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from resistor_network_calculator_base import ResistorNetworkCalculatorBase


class ResistorNetworkCalculator(ResistorNetworkCalculatorBase):
    def __init__(self, size=10):
        super().__init__(size)

    def create_conductivities_from_image(self, gate_v=0):
        """
        Fill up conductivity matrix from supplied image
        """
        conductivities = {}

        for i in range(1, self.size**2 + 1):
            row = 1 + (i - 1) // self.size
            col = 1 + (i - 1) % self.size

            conductivity = self.calculate_conductivity(row, col, gate_v)
            # print(i, 'Row: ', row, 'Col: ', col, 'G: ', conductivity)
            # g_matrix[i - 1] = conductivity

            # Left of current element
            if col > 1:  # First column has no element to the left
                x = i - 1
                conductivities[(x, i)] = conductivity
                conductivities[(i, x)] = conductivity

            # Right of current element
            if col < self.size:  # Last column has no element to the right
                x = i + 1
                conductivities[(x, i)] = conductivity
                conductivities[(i, x)] = conductivity

            # Upwards of current element
            if row > 1:  # First row has no element above
                x = i - self.size
                conductivities[(x, i)] = conductivity
                conductivities[(i, x)] = conductivity

            # Downwards of current element
            if row < self.size:  # Last row has no element below
                x = i + self.size
                conductivities[(x, i)] = conductivity
                conductivities[(i, x)] = conductivity
        return conductivities

    def calculate_elements(self, conductivities):
        """
        Fill up the sparse NxN matrix
        """
        diagonals = [[], [], [], [], []]
    
        rows = self.size
        for i in range(1, self.size**2 + 1):
            element = 0

            e1 = (i, i - rows)
            if e1 in conductivities:
                diagonals[0].append(conductivities[e1])
                element += conductivities[e1]
            else:
                if e1[1] > 0:
                    diagonals[0].append(0)

            e2 = (i, i - 1)
            if e2 in conductivities:
                diagonals[1].append(conductivities[e2])
                element += conductivities[e2]
            else:
                if e2[1] > 0:
                    diagonals[1].append(0)

            e3 = (i, i + 1)
            if e3 in conductivities:
                diagonals[3].append(conductivities[e3])
                element += conductivities[e3]
            else:
                if e3[1] < self.size**2:
                    diagonals[3].append(0)

            e4 = (i, i + rows)
            if e4 in conductivities:
                diagonals[4].append(conductivities[e4])
                element += conductivities[e4]
            else:
                if e4[1] < self.size**2:
                    diagonals[4].append(0)

            diagonals[2].append(element * -1)

        # for i in range(0, 5):
        #     print('Len diagonal {}: {}'.format(i, len(diagonals[i])))
        c_matrix = sp.sparse.diags(diagonals, [self.size*-1, -1, 0, 1, self.size], format='csc')
        return c_matrix

    def calculate_voltage_distribution(self, gate_v=0, conductivities=None):
        """
        If a conductivity matrix is provided, this will be used rather than the normal
        beaviour of loading from images.
        """
        t = time.time()
        # Conductivities will be calculated according to the standard rules
        # unless a conductivity map has been manually provided
        if conductivities is None:
            conductivities = self.create_conductivities_from_image(gate_v=gate_v)
        print('Create conductivities: ', time.time() - t)

        # In this example current is sourced in upper left corner and
        # drained in lower right corner
        I = np.zeros(shape=(self.size**2, 1), dtype=self.dtype)
        I[0] = 0.01
        I[-1] = -0.01

        t = time.time()
        c_matrix = self.calculate_elements(conductivities)
        print('Calculate elements: ', time.time() - t)

        # Peter's slides mentions finding the inverse and multiply, but
        # this is nummericly more efficient:
        t = time.time()
        v = sp.sparse.linalg.spsolve(c_matrix, I)
        # Direct implementation from slides for comparison
        # c_inv = np.linalg.inv(c_matrix)
        # v = np.matmul(c_inv, I)
        print('spsolve: {:.2f}s'.format(time.time() - t))

        # Re-shape the [N**2x1] vector in to a [NxN] matrix
        self.v_dist = v.reshape(self.size, self.size)

        # Calculate an approximate g-matrix for graphing tools to work
        t = time.time()
        self.g_matrix = self.create_g_matrix(conductivities)
        print('Calculate g_matrix: {:.2f}s'.format(time.time() - t))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import example_matrix

    # Here size must be 3!
    # size = 3
    # conductivities = example_matrix.fixed_conductivity_table()

    # Here size can be chosen more freely
    size = 5
    conductivities = example_matrix.create_conductivities(size)

    RNC = ResistorNetworkCalculator(size=size)
    RNC.create_g_matrix(conductivities)

    print(conductivities)

    print()

    print(RNC.g_matrix)

    exit()

    # RNC.load_doping_map('doping.png')
    # RNC.load_material_maps('conductor.png')

    RNC.calculate_voltage_distribution(conductivities=conductivities)

    plt.imshow(RNC.v_dist)
    plt.colorbar()
    plt.show()
