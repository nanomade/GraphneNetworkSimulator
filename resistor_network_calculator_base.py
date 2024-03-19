import numpy as np
# np.seterr(all='raise')

from PIL import Image
import matplotlib.pyplot as plt


class ResistorNetworkCalculatorBase:
    def __init__(self, size, current_electrodes):
        np.set_printoptions(precision=4, suppress=True, linewidth=170)
        self.dtype = np.float32  # or float64

        self.size = size
        self.current_in = current_electrodes[0]
        self.current_out = current_electrodes[1]

        self.metal_conductivity = 1e-2
        self.minimal_conductivity = 1e-5

        self.g_matrix = None
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

        image = np.asarray(c[color], dtype=np.uint)
        if plot_image:
            plt.imshow(image)
            plt.colorbar()
            plt.show()
        return image

    def load_doping_map(self, filename):
        doping_map = np.zeros(shape=(self.size, self.size), dtype=self.dtype)
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

    def create_g_matrix(self, conductivities):
        g_matrix_list = {}
        g_matrix = np.zeros(shape=(self.size**2, 1), dtype=self.dtype)
        for i in range(1, self.size**2 + 1):
            g_matrix_list[i] = []
            # row = 1 + (i - 1) // self.size
            # col = 1 + (i - 1) % self.size

            for coord in [
                    (i, i + 1), (i, i - 1), (i, i - self.size), (i, i + self.size)
            ]:
                value = conductivities.get(coord)
                if value is not None:
                    g_matrix_list[i].append(value)
            # print(i, ' row: ', row, '  col: ', col, 'list: ', g_matrix_list[i])

        for i in range(1, self.size**2 + 1):
            elements = g_matrix_list[i]
            g_matrix[i - 1] = sum(elements) / len(elements)

        # print(g_matrix)
        return g_matrix

    def calculate_current_density(self):
        """
        Calculate current density. Independant of the calculation backend,
        this is done on a rectangular grid.
        """
        if self.v_dist is None:
            print('Voltage map has not been calculated')
            return
        grad = np.gradient(self.v_dist)
        mag = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
        current_density = mag * self.g_matrix.reshape(self.size, self.size)
        return current_density

    def calculate_elements(self):
        raise NotImplementedError

    def create_conductivities_from_image(self, gate_v=0):
        raise NotImplementedError

    def calculate_voltage_distribution(self, gate_v=0):
        raise NotImplementedError


if __name__ == "__main__":
    msg = """
    Base class for the two calculation models.

    Also contains the converter from a strict network model into the square
    approximation, example shown below:
    """
