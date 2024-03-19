import argparse
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


class RNVisualizer:
    def __init__(
            self, size: int, skip_images: bool,
            current_electrodes: ((int, int), (int, int))
    ):
        self.size = size
        self.rnc = ResistorNetworkCalculator(size, current_electrodes)

        if not skip_images:
            self.rnc.load_doping_map('statics/doping.png')
            self.rnc.load_material_maps('statics/conductor.png')

    def color_map(self):
        if self.rnc.v_dist is None:
            print("Voltage map has not been calculated")
            return

        params = {
            "fontsize": 14,
            "verticalalignment": "top",
            "bbox": {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        }
        # Consider to make the figure global to allow for animations
        fig = plt.figure()  # Figsize...

        # TODO: Show only these if input is from an image
        if self.rnc.doping_map is not None:
            ax = fig.add_subplot(2, 3, 1)
            ax.text(0.05, 1.10, "Doping", transform=ax.transAxes, **params)
            # plt.imshow(self.rnc.g_matrix.reshape(self.size, self.size))
            plt.imshow(self.rnc.doping_map)

        if self.rnc.metal_map is not None:
            ax = fig.add_subplot(2, 3, 2)
            ax.text(0.05, 1.10, "Contacts", transform=ax.transAxes, **params)
            plt.imshow(self.rnc.metal_map)

        if self.rnc.graphene_map is not None:
            ax = fig.add_subplot(2, 3, 3)
            ax.text(0.05, 1.10, "Graphene", transform=ax.transAxes, **params)
            plt.imshow(self.rnc.graphene_map)

        # Conductivity
        ax = fig.add_subplot(2, 3, 4)
        ax.text(0.05, 1.10, "Conductivity", transform=ax.transAxes, **params)
        plt.imshow(self.rnc.g_matrix.reshape(self.size, self.size))

        # Current Density
        ax = fig.add_subplot(2, 3, 5)
        ax.text(
            0.05, 1.10, "Current Density (log scale)", transform=ax.transAxes, **params
        )
        current_density = self.rnc.calculate_current_density()
        plt.imshow(current_density, norm=colors.LogNorm())

        # Potential
        ax = fig.add_subplot(2, 3, 6)
        ax.text(0.05, 1.10, "Potential", transform=ax.transAxes, **params)
        # plt.imshow(self.rnc.v_dist, norm=colors.LogNorm())
        plt.imshow(self.rnc.v_dist)
        plt.colorbar()

        plt.show()

    def plot_surface(self):
        if self.rnc.v_dist is None:
            print("Voltage map has not been calculated")
            return
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)

        # g_matrix = self.g_matrix.reshape(self.size, self.size)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, self.rnc.v_dist, cmap="gray")
        # ax.plot_surface(x, y, g_matrix, cmap='gray')
        # Show the plot
        plt.show()


def parse_args():
    def position(s):
        try:
            x, y = map(int, s.split(','))
            return x, y
        except Exception:
            raise argparse.ArgumentTypeError('Position must be x,y')

    msg = 'Use size values higher then 1000 with caution.'
    parser = argparse.ArgumentParser(prog="main.py", description=msg)
    parser.add_argument("size", type=int, nargs=1, help="The size of the network")

    help = 'Current input coordinate (x, y)'
    parser.add_argument('--current_in', help=help, type=position, nargs=1)
    help = 'Current output coordinate (x, y)'
    parser.add_argument('--current_out', help=help, type=position, nargs=1)
    parser.add_argument("--hard-code-network", action="store_true")
    parser.add_argument("--gate_v", type=float,
                        nargs=1, default=[0], help="Gate voltage")
    args = vars(parser.parse_args())

    size = args["size"][0]
    gate_v = args["gate_v"][0]

    current_in = (1, 1)
    if args['current_in'] is not None:
        current_in = args['current_in'][0]
    current_out = (size, size)
    if args['current_out'] is not None:
        current_out = args['current_out'][0]
    current_electrodes = (current_in, current_out)

    if args["hard_code_network"]:
        gate_v = 0
        print("Hardcodet resistor network - images not loaded")
    return size, gate_v, args["hard_code_network"], current_electrodes


def main():
    size, gate_v, hard_coded_network, current_electrodes = parse_args()
    if hard_coded_network:
        from example_matrix import fixed_conductivity_table
        from example_matrix import create_conductivities

        # size must be 3 for fixed_conductivity_table()
        # conductivities = fixed_conductivity_table()
        conductivities = create_conductivities(size)
    else:
        conductivities = None

    rnv = RNVisualizer(
        size=size,
        skip_images=hard_coded_network,
        current_electrodes=current_electrodes
    )

    rnv.rnc.calculate_voltage_distribution(
        gate_v=gate_v,
        conductivities=conductivities
    )
    rnv.color_map()


if __name__ == "__main__":
    main()
