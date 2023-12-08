import time
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from resistor_network_calculator_base import ResistorNetworkCalculatorBase


class ResistorNetworkCalculator(ResistorNetworkCalculatorBase):
    def __init__(self, size=10):
        super().__init__(size)

    def calculate_elements(self):
        """
        Fill up the sparse NxN matrix
        """
        # TODO: Apparantly scipy has a sparse-type optimized for
        # diagonal (including diagonals with offsets) matrices
        c_matrix = np.zeros(shape=(self.size**2, self.size**2), dtype=self.dtype)
        for i in range(0, self.size**2):
            element = 0
            row = i // self.size
            col = i % self.size
            if row > 0:
                c_matrix[i, i - self.size] = self.g_matrix[i - self.size]
                element += self.g_matrix[i - self.size]
            if col > 0:
                c_matrix[i, i - 1] = self.g_matrix[i - 1]
                element += self.g_matrix[i - 1]
            if col < self.size - 1:
                c_matrix[i, i + 1] = self.g_matrix[i + 1]
                element += self.g_matrix[i + 1]
            if row < self.size - 1:
                c_matrix[i, i + self.size] = self.g_matrix[i + self.size]
                element += self.g_matrix[i + self.size]
            # Diagonal element
            c_matrix[i, i] = element * -1
        return c_matrix

    def create_conductivities_from_image(self, gate_v=0):
        """
        Fill up conductivity matrix from supplied image
        """
        g_matrix = np.zeros(shape=(self.size**2, 1), dtype=self.dtype)
        for i in range(1, self.size**2 + 1):
            row = 1 + (i - 1) // self.size
            col = 1 + (i - 1) % self.size
            conductivity = self.calculate_conductivity(row, col, gate_v)
            g_matrix[i - 1] = conductivity
        return g_matrix

    def calculate_current_density(self):
        # if self.v_dist is None:
        #     print('Voltage map has not been calculated')
        #     return
        # grad = np.gradient(self.v_dist)
        # mag = np.sqrt(grad[0]**2 + grad[1]**2)
        # current_density = mag * self.g_matrix.reshape(self.size, self.size)
        current_density = self._calculate_current_density()
        return current_density

    def calculate_voltage_distribution(self, gate_v=0):
        t = time.time()
        self.g_matrix = self.create_conductivities_from_image(gate_v=gate_v)
        print('Create conductivities: ', time.time() - t)

        # In this example current is sourced in upper left corner and
        # drained in lower right corner
        I = np.zeros(shape=(self.size**2, 1), dtype=self.dtype)
        I[0] = 0.01
        I[-1] = -0.01

        t = time.time()
        c_matrix = self.calculate_elements()
        print('Calculate elements: ', time.time() - t)

        # Peter's slides mentions finding the inverse and multiply, but
        # this is nummericly more efficient:
        c_matrix = sp.sparse.csr_matrix(c_matrix)
        print('Convert to sparse: ', time.time() - t)
        t = time.time()
        v = sp.sparse.linalg.spsolve(c_matrix, I)
        # Direct implementation from slides for comparison
        # c_inv = np.linalg.inv(c_matrix)
        # v = np.matmul(c_inv, I)

        self.v_dist = v.reshape(self.size, self.size)
