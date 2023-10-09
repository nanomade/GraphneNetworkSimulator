import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from resistor_network_calculator import ResistorNetworkCalculator


"""
TODO-list
These are the next items that needs to be implemented, more will be
added as they show up:

 * Export an animation of the conductivity, current-density and voltage
   as a function of gate sweep
 * Take mobiliy as an input map
 * Take metal and isolator conductivities as parameters rather than
   hard-coded values
 * Implement function to plot voltage between any two points as function
   of gate voltage
 * Investigate potential significant speedup by populating the sparse
   matrix directly. Here we should be able to take advantage of the
   fact that it is a sparse diagonal matrix;
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
 * Make a GUI to operate the program
"""

class RNVisualizer():
    def __init__(self, size):
        self.size = size
        self.rnc = ResistorNetworkCalculator(size)
        self.rnc.load_doping_map('doping.png')
        self.rnc.load_material_maps('conductor2.png')
        # Todo: At some point we should also load a mobility map

    def color_map(self):
        if self.rnc.v_dist is None:
            print('Voltage map has not been calculated')
            return
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Consider to make the figure global to allow for animations
        fig = plt.figure()  # Figsize...

        ax = fig.add_subplot(2, 3, 1)
        ax.text(0.05, 1.10, 'Doping', transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)
        # plt.imshow(self.rnc.g_matrix.reshape(self.size, self.size))
        plt.imshow(self.rnc.doping_map)

        ax = fig.add_subplot(2, 3, 2)
        ax.text(0.05, 1.10, 'Contacts', transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)
        plt.imshow(self.rnc.metal_map)

        ax = fig.add_subplot(2, 3, 3)
        ax.text(0.05, 1.10, 'Graphene', transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)
        plt.imshow(self.rnc.graphene_map)

        # Conductivity
        ax = fig.add_subplot(2, 3, 4)
        ax.text(0.05, 1.10, 'Conductivity', transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)
        plt.imshow(self.rnc.g_matrix.reshape(self.size, self.size))

        # Current Density
        ax = fig.add_subplot(2, 3, 5)
        ax.text(0.05, 1.10, 'Current Density (log scale)', transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)
        current_density = self.rnc.calculate_current_density()
        plt.imshow(current_density, norm=colors.LogNorm())

        # Potential
        ax = fig.add_subplot(2, 3, 6)
        ax.text(0.05, 1.10, 'Potential', transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)
        # plt.imshow(self.rnc.v_dist, norm=colors.LogNorm())
        plt.imshow(self.rnc.v_dist)

        # plt.colorbar()
        plt.show()

    def plot_surface(self):
        if self.rnc.v_dist is None:
            print('Voltage map has not been calculated')
            return
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)

        # g_matrix = self.g_matrix.reshape(self.size, self.size)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, self.rnc.v_dist, cmap='gray')
        # ax.plot_surface(x, y, g_matrix, cmap='gray')
        # Show the plot
        plt.show()


def main():
    # Max with currently available memory is ~200 - this is
    # limited by the memory required to hold the dense version
    # of the calculation matrix. It should be possible to improve
    # this by populating the sparse matrix directly
    rnv = RNVisualizer(150)
    rnv.rnc.calculate_voltage_distribution(gate_v=-1)
    rnv.color_map()

    # rnv.rnc.calculate_voltage_distribution(gate_v=0)
    # fig, ax = plt.subplots(1, 1)
    # im = ax.imshow(rnv.rnc.v_dist)
    # plt.colorbar()
    # plt.show()
    # for gate_v in range(-50, 50):
    #     print(gate_v)
    #     rnv.rnc.calculate_voltage_distribution(gate_v=gate_v)
    #     im.set_data(rnv.rnc.v_dist)
    #     fig.canvas.draw_idle()
    #     plt.pause(0.1)
    # #     # RN.plot_surface()


if __name__ == '__main__':
    main()
