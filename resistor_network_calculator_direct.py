import time
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from resistor_network_calculator_base import ResistorNetworkCalculatorBase


class DirectResistorNetworkCalculator(ResistorNetworkCalculatorBase):
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
        diagonal1 = []
        diagonal2 = []
        diagonal3 = []
        diagonal4 = []
        diagonal5 = []
    
        c_matrix = np.zeros(shape=(self.size**2, self.size**2), dtype=self.dtype)
        rows = int(c_matrix.shape[0] ** 0.5)
        for i in range(1, c_matrix.shape[0] + 1):
            element = 0

            e1 = (i, i - rows)
            e2 = (i, i - 1)
            e3 = (i, i + 1)
            e4 = (i, i + rows)
            if e1 in conductivities:
                c_matrix[e1[0] - 1, e1[1] - 1] = conductivities[e1]
                diagonal1.append(conductivities[e1])
                element += conductivities[e1]
            if e2 in conductivities:
                c_matrix[e2[0] - 1, e2[1] - 1] = conductivities[e2]
                diagonal2.append(conductivities[e2])
                element += conductivities[e2]
            if e3 in conductivities:
                c_matrix[e3[0] - 1, e3[1] - 1] = conductivities[e3]
                diagonal4.append(conductivities[e3])
                element += conductivities[e3]
            if e4 in conductivities:
                c_matrix[e4[0] - 1, e4[1] - 1] = conductivities[e4]
                diagonal5.append(conductivities[e4])
                element += conductivities[e4]
            # Diagonal element
            c_matrix[i - 1, i - 1] = element * -1
            diagonal3.append(element * -1)

        print('Len diagonal 1: ', len(diagonal1))
        print('Len diagonal 2: ', len(diagonal2))
        print('Len diagonal 3: ', len(diagonal3))
        print('Len diagonal 4: ', len(diagonal4))
        print('Len diagonal 5: ', len(diagonal5))

        print()
        print(diagonal4)


        print()

        print(c_matrix)
        # plt.imshow(c_matrix)
        # plt.colorbar()
        # plt.show()

       
        print()
        print('Diagonal 3:')
        for i in range(0, c_matrix.shape[0]):
            diff = diagonal3[i] - c_matrix[i, i]
            if diff < 1e-9:              
                print(0, end = ' ')
            else:
                print(diff, end = ' ')
        print()
        print()

        print()
        print('Diagonal 4:')
        for i in range(0, c_matrix.shape[0]):
            diff = diagonal4[i] - c_matrix[i + 1, i]
            if diff < 1e-9:              
                print(0, end = ' ')
            else:
                print(diff, end = ' ')
        print()
        print()

            
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
        # c_matrix2 = self.calculate_elements_2(conductivities)
        print('Calculate elements: ', time.time() - t)

        t = time.time()
        # Peter's slides mentions finding the inverse and multiply, but
        # this is nummericly more efficient:
        c_matrix = sp.sparse.csr_matrix(c_matrix)
        print('Convert to sparse: ', time.time() - t)
        t = time.time()
        v = sp.sparse.linalg.spsolve(c_matrix, I)
        print(time.time() - t)
        # Direct implementation from slides for comparison
        # c_inv = np.linalg.inv(c_matrix)
        # v = np.matmul(c_inv, I)

        # Re-shape the [N**2x1] vector in to a [NxN] matrix
        self.v_dist = v.reshape(self.size, self.size)
        print(time.time() - t)

        # print(conductivities)
        
        # Calculate and approximate g-matrix for graphing tools to work
        print('Calculate g_matrix:')
        t = time.time()
        self.g_matrix = self.create_g_matrix(conductivities)
        print(time.time() - t)

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
