import time
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


class ResistorNetwork():
    def __init__(self, size=10):
        np.set_printoptions(precision=2, suppress=True, linewidth=140)
        self.size = size
        self.v_dist = None

    def _load_image(self, filename, plot_image=False):
        # c_image = Image.open('conductor2.png').convert('L')
        c_image = Image.open(filename).convert('L')
        c_image = c_image.resize((self.size, self.size))
        c_image = np.asarray(c_image)
        if plot_image:
            plt.imshow(c_image)
            plt.colorbar()
            plt.show()
        return c_image

    def calculate_elements(self, conductivities):
        c_matrix = np.zeros(
            shape=(self.size**2, self.size**2),
            dtype=np.float32  # or float64
        )
        rows = int(c_matrix.shape[0]**0.5)

        for i in range(1, c_matrix.shape[0] + 1):
            element = 0
            # row = 1 + (i - 1) // rows
            # col = 1 + (i - 1) % rows

            e1 = (i, i - rows)
            e2 = (i, i - 1)
            e3 = (i, i + 1)
            e4 = (i, i + rows)
            if e1 in conductivities:
                c_matrix[e1[0] - 1, e1[1] - 1] = conductivities[e1]
                element += conductivities[e1]
            if e2 in conductivities:
                c_matrix[e2[0] - 1, e2[1] - 1] = conductivities[e2]
                element += conductivities[e2]
            if e3 in conductivities:
                c_matrix[e3[0] - 1, e3[1] - 1] = conductivities[e3]
                element += conductivities[e3]
            if e4 in conductivities:
                c_matrix[e4[0] - 1, e4[1] - 1] = conductivities[e4]
                element += conductivities[e4]
            # Diagonal element
            c_matrix[i - 1, i - 1] = element * -1

        # plt.imshow(c_matrix)
        # plt.colorbar()
        # plt.show()
        return c_matrix

    def create_conductivities_from_image(self, filename):
        """
        Fill up conductivity matrix from supplied image
        """
        c_image = self._load_image(filename)
        conductivities = {}
        for i in range(1, self.size**2 + 1):
            row = 1 + (i - 1) // self.size
            col = 1 + (i - 1) % self.size

            conductivity = c_image[row - 1][col - 1] / 255.0 + 1e-5
            # print(i, 'Row: ', row, 'Col: ', col, 'G: ', conductivity)

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

    def color_map(self):
        if self.v_dist is None:
            print('Voltage map has not been calculated')
            return
        plt.imshow(self.v_dist)
        plt.colorbar()
        plt.show()

    def plot_surface(self):
        if self.v_dist is None:
            print('Voltage map has not been calculated')
            return
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, self.v_dist, cmap='gray')
        # Show the plot
        plt.show()

    def calculate_voltage_distribution(self, image):
        conductivities = self.create_conductivities_from_image(image)
        # In this example current is sourced in upper left corner and
        # drained in lower right corner
        I = np.zeros(shape=(self.size**2, 1), dtype=np.float32)
        I[0] = 0.01
        I[-1] = -0.01

        c_matrix = self.calculate_elements(conductivities)

        # Peter's slides mentions finding the inverse and multiply, but
        # this is nummericly more efficient:
        v = np.linalg.solve(c_matrix, I)
        # Direct implementation from slides for comparison
        # c_inv = np.linalg.inv(c_matrix)
        # v = np.matmul(c_inv, I)

        # c_matrix = scipy.sparse.csr_matrix(c_matrix)
        # v = scipy.sparse.linalg.spsolve(c_matrix, I)


        self.v_dist = v.reshape(self.size, self.size)

import scipy
RN = ResistorNetwork(80)
t = time.time()
RN.calculate_voltage_distribution('conductor2.png')
print('Run-time: {:.3f}ms'.format((time.time() - t) * 1000))
RN.color_map()
RN.plot_surface()
