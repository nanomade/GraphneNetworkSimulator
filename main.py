import numpy as np
import matplotlib.pyplot as plt

from resistor_network_calculator import ResistorNetworkCalculator


class RNVisualizer():
    def __init__(self, size):
        self.size = size
        self.rnc = ResistorNetworkCalculator(size)
        self.rnc.load_doping_map('doping.png')
        self.rnc.load_material_maps('conductor2.png')
        # Todo: At some point we should also load a mobility map
        self.rnc.calculate_voltage_distribution(gate_v=25)

    def color_map(self):
        if self.rnc.v_dist is None:
            print('Voltage map has not been calculated')
            return

        # float32 is always enough for plotting
        v_map = np.zeros(
            shape=(self.size, self.size),
            dtype=np.float32
        )
        for i in range(1, self.size):
            for j in range(1, self.size):
                v_map[i][j] = self.rnc.v_dist[i - 1][j - 1] - self.rnc.v_dist[i][j]

        g_matrix = self.rnc.g_matrix.reshape(self.size, self.size)
        # im = plt.imshow(g_matrix)
        plt.imshow(self.rnc.v_dist)
        # # plt.imshow(v_map)
        plt.colorbar()
        plt.show()
        return g_matrix

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
    rnv = RNVisualizer(50)
    rnv.color_map()
    # g_matrix = RN.color_map()
    # print(g_matrix)
    # fig, ax = plt.subplots(1, 1)
    # im = ax.imshow(g_matrix)
    # # plt.colorbar()
    # # plt.show()

    # for gate_v in range(0, 100):
    #     print(gate_v)
    #     RN.calculate_voltage_distribution(gate_v=gate_v)
    #     g_matrix = RN.color_map()
    #     im.set_data(g_matrix)
    #     fig.canvas.draw_idle()
    #     plt.pause(0.1)
    #     # RN.plot_surface()


if __name__ == '__main__':
    main()
