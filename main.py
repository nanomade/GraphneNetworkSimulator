import time
import numpy as np

from PIL import Image
# from matplotlib import image
import matplotlib.pyplot as plt


def calculate_elements(conductivities, c_matrix):
    rows = int(c_matrix.shape[0]**0.5)
    print(rows)
    # for i in range(1, n + 1):
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
    return c_matrix

def create_conductivities(m_size):
    """
    Fill up conductivity matrix with a default value of 0.01
    """
    conductivities = {}
    for i in range(1, m_size**2 + 1):
        row = 1 + (i - 1) // m_size
        col = 1 + (i - 1) % m_size
        print(i, row, col)

        if (
                (m_size/2 - 8 < row < m_size/2 + 7) and
                (m_size/2 - 8 < col < m_size/2 + 7)
        ):
            conductivity = 1e-5
            print(i)
        else:
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
    conductivities[(1,2)] = 1.0
    conductivities[(2,1)] = 1.0
    conductivities[(m_size**2 - 1, m_size**2)] = 1.0
    conductivities[(m_size**2, m_size**2 - 1)] = 1.0
    return conductivities


def create_conductivities_from_image(c_image):
    """
    Fill up conductivity matrix from supplied image
    """
    m_size = c_image.shape[0]
    conductivities = {}
    for i in range(1, m_size**2 + 1):
        row = 1 + (i - 1) // m_size
        col = 1 + (i - 1) % m_size

        conductivity = c_image[col - 1][row - 1] / 255.0 + 1e-5
        print(conductivity)
        # print(i, row, col, conductivity)
        
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
    return conductivities


np.set_printoptions(precision=2, suppress=True, linewidth=140)
t = time.time()
m_size = 6

# c_image = image.imread('trivial_corners.png').convert('L')
# c_image = Image.open('trivial_corners.png').convert('L')
c_image = Image.open('conductor.png').convert('L')
c_image = c_image.resize((m_size, m_size))
c_image = np.asarray(c_image)

# conductivities = create_conductivities(m_size)
conductivities = create_conductivities_from_image(c_image)

# conductivities = np.asarray(c_image)
# print(conductivities.shape)
# conductivities = create_conductivities(m_size)
# print(conductivities.shape)

print(time.time() - t)
# In this example current is sourced in upper left corner and
# drained in lower right corner
I = np.zeros(shape=(m_size**2, 1), dtype=np.float32)
I[0] = 0.001
I[-1] = -0.001

# n = 0
# for c in conductivities:
#    if c[0] > n:
#        n = c[0]
# print(n)
# So far we use square networks
# c_matrix = np.zeros(shape=(n, n), dtype=np.float32)  # or float64
c_matrix = np.zeros(shape=(m_size**2, m_size**2), dtype=np.float32)  # or float64
c_matrix = calculate_elements(conductivities, c_matrix)

plt.imshow(c_matrix)
plt.colorbar()
plt.show()

# plt.imshow(c_matrix)
# plt.colorbar()
# plt.show()

# Peter's slides mentions finding the inverse and multiply, but
# this is nummericly more efficient:
t = time.time()
v = np.linalg.solve(c_matrix, I)
print(time.time() - t)
# Direct implementation from slides for comparison
# c_inv = np.linalg.inv(c_matrix)
# v = np.matmul(c_inv, I)
v_dist = v.reshape(m_size, m_size)

print()
print('Run-time: {:.3f}ms'.format((time.time() - t) * 1000))

plt.imshow(v_dist)
plt.colorbar()
plt.show()


