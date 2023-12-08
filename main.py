import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from resistor_network_calculator_direct import DirectResistorNetworkCalculator
from resistor_network_calculator_fast import FastResistorNetworkCalculator


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


class RNVisualizer:
    def __init__(self, size: int = 20, model: str = 'fast'):
        self.size = size
        if model == 'fast':
            self.rnc = FastResistorNetworkCalculator(size)
        else:
            self.rnc = DirectResistorNetworkCalculator(size)

        self.rnc.load_doping_map('doping.png')
        self.rnc.load_material_maps('conductor.png')
        # Todo: At some point we should also load a mobility map

    def color_map(self):
        if self.rnc.v_dist is None:
            print('Voltage map has not been calculated')
            return
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Consider to make the figure global to allow for animations
        fig = plt.figure()  # Figsize...

        # TODO: Show only these if input is from an image
        ax = fig.add_subplot(2, 3, 1)
        params = {'fontsize': 14, 'verticalalignment': 'top', 'bbox': props}
        ax.text(0.05, 1.10, 'Doping', transform=ax.transAxes, **params)
        # plt.imshow(self.rnc.g_matrix.reshape(self.size, self.size))
        plt.imshow(self.rnc.doping_map)

        ax = fig.add_subplot(2, 3, 2)
        ax.text(0.05, 1.10, 'Contacts', transform=ax.transAxes, **params)
        plt.imshow(self.rnc.metal_map)

        ax = fig.add_subplot(2, 3, 3)
        ax.text(0.05, 1.10, 'Graphene', transform=ax.transAxes, **params)
        plt.imshow(self.rnc.graphene_map)

        # Conductivity
        ax = fig.add_subplot(2, 3, 4)
        ax.text(0.05, 1.10, 'Conductivity', transform=ax.transAxes, **params)
        plt.imshow(self.rnc.g_matrix.reshape(self.size, self.size))

        # Current Density
        ax = fig.add_subplot(2, 3, 5)
        ax.text(
            0.05, 1.10, 'Current Density (log scale)', transform=ax.transAxes, **params
        )
        current_density = self.rnc.calculate_current_density()
        plt.imshow(current_density, norm=colors.LogNorm())

        # Potential
        ax = fig.add_subplot(2, 3, 6)
        ax.text(0.05, 1.10, 'Potential', transform=ax.transAxes, **params)
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


def parse_args():
    msg = """
    For model direct, use size values higher then ~90 with caution.
    For model direct, max size for 8GB memory is approximately is ~200 - this is
    limited by the memory required to hold the dense version of the calculation matrix.
    """
    # It should be possible to improve this by populating the sparse matrix directly

    parser = argparse.ArgumentParser(prog='main.py', description=msg)
    parser.add_argument('size', type=int, nargs=1, help='The size of the network')
    parser.add_argument('--gate_v', type=float, nargs=1, default=0, help='Gate voltage')
    parser.add_argument(
        '--model',
        required=False,
        nargs=1,
        choices=['fast', 'direct'],
        help='Calculation backend (default is fast)',
    )
    args = vars(parser.parse_args())

    size = args['size'][0]
    gate_v = args['gate_v'][0]
    if args['model'] is None:
        model = 'fast'
    else:
        model = args['model'][0]

    return size, gate_v, model


def main():
    size, gate_v, model = parse_args()

    rnv = RNVisualizer(size=size, model=model)
    rnv.rnc.calculate_voltage_distribution(gate_v=gate_v)
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
