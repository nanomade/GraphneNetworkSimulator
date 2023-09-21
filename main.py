import time
import numpy as np
import scipy as sp

from PIL import Image
import matplotlib.pyplot as plt


class ResistorNetwork():
    def __init__(self, size=10):
        np.set_printoptions(precision=2, suppress=True, linewidth=170)
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
        msg = 'Row: {}. Col: {}, n_doping: {:.3f}.  Sigma: {:.3f}'
        # print(msg.format(row, col, n_doping, sigma))
        return sigma

    def calculate_conductivity(self, row, col, gate_v):
        conductivity = self.minimal_conductivity
        
        if self.metal_map[row - 1][col - 1] > 0:
            conductivity = self.metal_conductivity
        elif self.graphene_map[row - 1][col - 1] > 0:
            conductivity = self._calculate_graphene_conductivity(row, col, gate_v)
        return conductivity
    
    def calculate_elements(self, conductivities):
        """
        Fill up the sparse NxN matrix
        """
        c_matrix = np.zeros(
            shape=(self.size**2, self.size**2),
            dtype=self.dtype
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

    def create_conductivities_from_image(self, gate_v = 0):
        """
        Fill up conductivity matrix from supplied image
        """
        # c_image = self._load_image(filename)
        conductivities = {}
        g_matrix = np.zeros(shape=(self.size**2, 1), dtype=self.dtype)

        
        for i in range(1, self.size**2 + 1):
            row = 1 + (i - 1) // self.size
            col = 1 + (i - 1) % self.size

            # conductivity = c_image[row - 1][col - 1] / 255.0 + 1e-5
            conductivity = self.calculate_conductivity(row, col, gate_v)
            # print(i, 'Row: ', row, 'Col: ', col, 'G: ', conductivity)
            g_matrix[i - 1] = conductivity
            
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

        self.g_matrix = g_matrix
        return conductivities

    def color_map(self):
        if self.v_dist is None:
            print('Voltage map has not been calculated')
            return

        v_map = np.zeros(
            shape=(self.size, self.size),
            dtype=self.dtype
        )
        for i in range(1, self.size):
            for j in range(1, self.size):
                v_map[i][j] = self.v_dist[i - 1][j - 1] - self.v_dist[i][j]

        g_matrix = self.g_matrix.reshape(self.size, self.size)
        # im = plt.imshow(g_matrix)
        # # plt.imshow(self.v_dist)
        # # plt.imshow(v_map)
        # plt.colorbar()
        # plt.show()
        return g_matrix

    def plot_surface(self):
        if self.v_dist is None:
            print('Voltage map has not been calculated')
            return
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)

        # g_matrix = self.g_matrix.reshape(self.size, self.size)

        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, self.v_dist, cmap='gray')
        # ax.plot_surface(x, y, g_matrix, cmap='gray')
        # Show the plot
        plt.show()

    def calculate_voltage_distribution(self, gate_v=0):
        t = time.time()
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
        # t = time.time()
        # v = np.linalg.solve(c_matrix, I)
        # print(time.time() - t)
        # Direct implementation from slides for comparison
        # c_inv = np.linalg.inv(c_matrix)
        # v = np.matmul(c_inv, I)

        t = time.time()
        c_matrix = sp.sparse.csr_matrix(c_matrix)
        print('Convert to sparse: ', time.time() - t)
        t = time.time()
        v = sp.sparse.linalg.spsolve(c_matrix, I)
        print(time.time() - t)
        self.v_dist = v.reshape(self.size, self.size)
        print(time.time() - t)


RN = ResistorNetwork(50)
t = time.time()
RN.load_doping_map('doping.png')
RN.load_material_maps('conductor2.png')

# RN.calculate_voltage_distribution()
# print('Run-time: {:.3f}ms'.format((time.time() - t) * 1000))

RN.calculate_voltage_distribution(gate_v=0)
g_matrix = RN.color_map()
print(g_matrix)
fig, ax = plt.subplots(1, 1)
im = ax.imshow(g_matrix)
# plt.colorbar()
# plt.show()

for gate_v in range(0, 100):
    print(gate_v)
    RN.calculate_voltage_distribution(gate_v=gate_v)
    g_matrix = RN.color_map()
    im.set_data(g_matrix)
    fig.canvas.draw_idle()
    plt.pause(0.1)
# RN.plot_surface()
