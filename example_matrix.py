def fixed_conductivity_table():
    # A fixed 3x3 array.
    # Upper and lower corner have a higher conductivity, otherwise uniform
    # Use as base for experimentation
    conductivities = {
        (1, 2): 0.1, (2, 1): 0.1,
        (2, 3): 0.01, (3, 2): 0.01,
        (1, 4): 0.01, (4, 1): 0.01,
        (2, 5): 0.01, (5, 2): 0.01,
        (3, 6): 0.01, (6, 3): 0.01,
        (4, 5): 0.01, (5, 4): 0.01,
        (5, 6): 0.01, (6, 5): 0.01,
        (4, 7): 0.01, (7, 4): 0.01,
        (5, 8): 0.01, (8, 5): 0.01,
        (6, 9): 0.01, (9, 6): 0.01,
        (7, 8): 0.01, (8, 7): 0.01,
        (8, 9): 0.1, (9, 8): 0.1,
    }
    return conductivities


def create_conductivities(m_size):
    """
    Create a homogenous network as base for manual modifications.
    """
    conductivities = {}
    for i in range(1, m_size**2 + 1):
        row = 1 + (i - 1) // m_size
        col = 1 + (i - 1) % m_size
        # print(i, row, col)

        # This is an example of way to modify certain parts of the network
        # if (
        #         (m_size/2 - 8 < row < m_size/2 + 7) and
        #         (m_size/2 - 8 < col < m_size/2 + 7)
        # ):
        #     conductivity = 1e-5
        #     print(i)
        # else:
        #     conductivity = 0.01
        conductivity = 0.01

        # Left of current element
        if col > 1:
            # The first column does not have an element to the left
            x = i - 1
            y = i
            conductivities[(x, y)] = conductivity
            conductivities[(y, x)] = conductivity

        # Right of current element
        if col < m_size:
            # The last column does not have an element to the right
            x = i + 1
            y = i
            conductivities[(x, y)] = conductivity
            conductivities[(y, x)] = conductivity

        # Upwards of current element
        if row > 1:
            # The first row does not have an element above
            x = i - m_size
            y = i
            conductivities[(x, y)] = conductivity
            conductivities[(y, x)] = conductivity

        # Downwards of current element
        if row < m_size:
            # The last row does not have an element below
            x = i + m_size
            y = i
            conductivities[(x, y)] = conductivity
            conductivities[(y, x)] = conductivity

    # Metalize the corners
    conductivities[(1, 2)] = 1.0
    conductivities[(2, 1)] = 1.0
    conductivities[(m_size**2 - 1, m_size**2)] = 1.0
    conductivities[(m_size**2, m_size**2 - 1)] = 1.0
    return conductivities


if __name__ == '__main__':
    msg = """
    This file contains two conductivity matrixes that can used to test examples of the
    stringent model of the resistor network as opposed to the square approximation
    typically used.
    """
    print(msg)
    print()
    print(fixed_conductivity_table())

    print()

    print(create_conductivities(4))
