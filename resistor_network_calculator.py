import time
import numpy as np
import scipy as sp

from PIL import Image
import matplotlib.pyplot as plt


class ResistorNetworkCalculator():
    def __init__(self, size=10):
        np.set_printoptions(precision=3, suppress=True, linewidth=170)
        self.dtype = np.float32  # or float64

        self.size = size

        self.g_matrix = None

        self.metal_conductivity = 1e-2
        self.minimal_conductivity = 1e-5

        self.graphene_map = None
        self.metal_map = None
        self.doping_map = None

        self.v_dist = None

    def _load_image(self, filename, color, plot_image=False):
        # c_image = Image.open('conductor2.png').convert('L')
        # image = Image.open(filename).convert('L')
        c = {}  # Components
        image = Image.open(filename).convert('RGB')
        image = image.resize((self.size, self.size))
        # Split into 3 channels
        c['red'], c['green'], c['blue'] = image.split()

        image = np.asarray(c[color])
        if plot_image:
            plt.imshow(image)
            plt.colorbar()
            plt.show()
        return image

    def load_doping_map(self, filename):
        doping_map = np.zeros(
            shape=(self.size, self.size),
            dtype=self.dtype
        )
        if filename is None:
            self.doping_map = doping_map
            return

        d_image = self._load_image(filename, 'red')
        for row in range(0, self.size):
            for col in range(0, self.size):
                doping = (d_image[row][col] - 127) * 10**17 / 256.0
                doping_map[row][col] = doping
        self.doping_map = doping_map

    def load_material_maps(self, filename):
        """
        Decodes metal and graphene based on color
        """
        metal_img = self._load_image(filename, 'red')
        self.metal_map = metal_img > 100
        graphene_img = self._load_image(filename, 'green')
        self.graphene_map = graphene_img > 100

    def _calculate_graphene_conductivity(self, row, col, gate_v):
        e = 1.60217663e-19  # Coloumb
        epsilon_0 = 8.8541878128e-12  # F/m
        n_0 = 1e13  # Found on google... - Unit: m^-2

        d = 1e-7  # Parameter, but 100nm is not a bad start - Unit: m
        epsilon_r = 4  # Parameter, but 4 is not a bad start - no unit
        mu = 1  # Parameter, but 1 is not a bad start, Unit: m**2/(v * s)

        n_vg = gate_v * epsilon_0 * epsilon_r / (d * e)
        n_doping = self.doping_map[row - 1][col - 1]
        n_exp = n_vg + n_doping

        n = (n_0**2 + n_exp**2) ** 0.5
        sigma = n * e * mu
        # msg = 'Row: {}. Col: {}, n_doping: {:.3f}.  Sigma: {:.3f}'
        # print(msg.format(row, col, n_doping, sigma))
        return sigma

    def calculate_conductivity(self, row, col, gate_v):
        conductivity = self.minimal_conductivity
        if self.metal_map[row - 1][col - 1] > 0:
            conductivity = self.metal_conductivity
        elif self.graphene_map[row - 1][col - 1] > 0:
            conductivity = self._calculate_graphene_conductivity(row, col, gate_v)
        return conductivity

    def calculate_elements(self):
        """
        Fill up the sparse NxN matrix
        """
        # TODO: Apparantly scipy has a sparse-type optimized for
        # diagonal (including diagonals with offsets) matrices
        c_matrix = np.zeros(
            shape=(self.size**2, self.size**2),
            dtype=self.dtype
        )
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
        if self.v_dist is None:
            print('Voltage map has not been calculated')
            return
        grad = np.gradient(self.v_dist)
        mag = np.sqrt(grad[0]**2 + grad[1]**2)
        current_density = mag * self.g_matrix.reshape(self.size, self.size)
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
