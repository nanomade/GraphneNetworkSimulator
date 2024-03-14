==========================
Resistor Network Simulator
==========================

Overview
========

This code will simulate a network of resistors in a rectangular network.

Configuration
-------------

Size
++++
The size of the grid is given as an integer command line parameter.

Doping
++++++
Given as a grey-scale (or red-scale) image file. A pixel value of 127 indicates
no doping, and higher og lower values indicate p- og n- doping.

Conductivity
++++++++++++
Given as an image with two color channels, red and green (blue pixels are ignored). Red pixels
indicate metal, green pixels indicate graphene. Pixels with values in both channels are not
allowed.


Interpretation of the configuration images
------------------------------------------

Principally, the software should simulate a network like this::

  1--G--2--G--3--G--4
  |     |     |     |
  G     G     G     G
  |     |     |     |
  5--G--6--G--7--G--8
  |     |     |     |
  G     G     G     G
  |     |     |     |
  9--G-10--G-11--G-12
  |     |     |     |
  G     G     G     G
  |     |     |     |
  13-G-14--G-15--G-16


Voltage is measured in the numbered points, and resistors are named by the corresponding
pair of measurement points.

However, this is a very impractially arrangement, since the resistors are not arranged in
a perfect square. For large network, we could argue that the difference between this ideal
configuration and a perfect square is quite small. Since a proper square grid is much more
convenient, most calculations are performed in a grid like this::

  G(1,1)--G(1,2)--G(1,3)--G(1,4)
  |       |       |       |
  G(2,1)--G(2,2)--G(2,3)--G(2,4)
  |       |       |       |
  G(3,1)--G(3,2)--G(3,3)--G(3,4)
  |       |       |       |
  G(4,1)--G(4,2)--G(4,3)--G(4,4)


The number of measurements points is the same, but the conductivity between each point
is no longer a distinct set of resistors, but an average conductivity in the vicinity
of each measurement point. In the edges this is not exactly the same, but for large networks
the approximation is good.

You can try out the difference between the two models by testing the models provided in
`example_matrix.py` rather than loading maps from images.


Use of the command line tool
============================

Currently, it is not possible to configure the input images at run time, these will always
be loaded from `statics/conductor.png`  and `statics/doping.png` - to change the datafiles,
simply rename your wanted file to fit this.

The command line tool can be executed without arguments, in which case it will output a short
help message::

 > python main.py
 usage: main.py [-h] [--hard-code-network] [--gate_v GATE_V] [--model {fast,direct}] size
 main.py: error: the following arguments are required: size

As seen from the output, only a single argument is required: `size`. If run in this way,
the program will run with a gate voltage of 0 and unsing the fast model.

If the arguement `hard-code-network` is set, the input images will not be loaded,
instead the conductivities hard-coded in `example_matrix.py` will be used. In this case
there will be no gate dependence.

Overview of included files
==========================

This package consists of a number of python files, a short introduction to each file
is given below. Also, each file can be executed individiually, for files mainly intended
as helper-files, this will print a short description as well as some output usefull for
development and debugging.

The included files are:
 * `example_matrix`: Two examples of proper networks as opposed to the square
   approximation otherwise used.
 * `resistor_network_calculator_base`: TODO!!!! Base class for the two calculators 
 * `resistor_network_calculator_direct`: An implentation of the network calculation
   closely following the direct calculation as described in the theory.
 * `resistor_network_calculator_fast`: A much faster implentation that utilises a number
   nummerical optimizations.
 * `main`: The main execuatable file to be used to perform actual simulations.

   
Limitations
-----------

This software is still under development and has a number of limitations:

 * Missing validation of size-input from command prompt
 * Names of input files are harcodet as `doping.png` and `conductor.png`.
 * Current can only be sourced from upper left corner and drained from lower right corner.
 * Currently (and maybe forever), only rectangular networks are supported.

l
