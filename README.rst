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
no doping - higher or lower values indicate p- or n- doping.

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

However, in some regards this is a very impractially arrangement, since the resistors are
not arranged in a perfect square. For large network, we could argue that the difference
between this ideal configuration and a perfect square is quite small. Since a proper square
grid is much more convenient, with regard to feeding input into the program, the actual grid
used in the calculations is based on square input maps::

  G(1,1)--G(1,2)--G(1,3)--G(1,4)
  |       |       |       |
  G(2,1)--G(2,2)--G(2,3)--G(2,4)
  |       |       |       |
  G(3,1)--G(3,2)--G(3,3)--G(3,4)
  |       |       |       |
  G(4,1)--G(4,2)--G(4,3)--G(4,4)


The conductivity between each point is not a distinct set of resistors, but an average conductivity
in the vicinity of each measurement point. In the edges this is not exactly the same, but for large
networks the approximation is good.

You can try out the difference between the two models by testing the models provided in
`example_matrix.py` rather than loading maps from images. The exact algorithm for feeding the square
conductivity networks into the model is found in the function `create_conductivities_from_image()`
in `resistor_network_calculator.py`.


Performance considerations
--------------------------

As explained in the course notes, the simulation involves solving an equation of the form
GxV = I, with V and I both being of the size of the area of the network, ie the square of
the `size` parameter given to the program.

The size of G is the square of this value, ie `size`:sup:`4`. This means that the time and
memory requirement of the calculation could be expected to have a runtime complexety
of O(`size`:sup:`4`). However, scince the intermediate matrix has only 5 non-zero entries pr
row, the total amount of non-zero elements is (almost) exactly 5 * `size`:sup:`2`, which does
not scale as as O(`size`:sup:`4`). Nummerically this can be exploited by using the concept
of sparse matrices - a matrix implementation that does not involve full sized arrays, but
rather a data structure that contains only the non-zero elements. Optimizing this type of
calculations is in itself an active field of research, and in Python two of the currently
best implementaitons are `umfpack` and `paradisio`.

If this program is run in an environment prepared by the provided requirements.txt file,
`umfpack` should be used, but if you experience performance problems, you should investigate
if you are somehow not using the optimized libraries, but rather the non-optimized pure python
implementation. If you are using the correct libraries, at calculation on a network of size
200 should take significantly less than 10s even on very modest hardware.

 * https://github.com/scikit-umfpack/scikit-umfpack
 * https://github.com/haasad/PyPardiso


Use of the command line tool
============================

Currently, it is not possible to configure the input images at run time, these will always
be loaded from `statics/conductor.png`  and `statics/doping.png` - to change the datafiles,
simply rename your wanted file to fit this.

The command line tool can be executed without arguments, in which case it will output a short
help message::

 > python main.py
 usage: main.py [-h] [--current_in CURRENT_IN] [--current_out CURRENT_OUT] [--vmeter_low VMETER_LOW] [--vmeter_high VMETER_HIGH] [--gate_v GATE_V] [--print-extra-output] [--hard-code-network] size

As seen from the output, only a single argument is required: `size`. If run in this way, the
model will use a gate voltage of zero, and place both the volt-meter and source electrodes
in the corners of the sample.

If the arguement `hard-code-network` is set, the input images will not be loaded, instead
the conductivities hard-coded in `example_matrix.py` will be used. In this case there will
be no gate dependence. This feature is usefull if you want to compare an exactly prepared
grid to the approximation usually used when loading conductivities from images files.

Parameters `current_in`, `current_out`, `vmeter_low` and `vmeter_high` positions the four
electrodes. A small patch of metal will be programmatically added around each probe
independant of the conductivities loaded from the images. Syntax for the four parameters
are all of the form `--parameter=y,x`. If an argument is missing, it will default to either
the upper left corner (`current_in` and `vmeter_low`) or the lower right corner (`current_out`
and `vmeter_high`).

The simulated current is always 1mA, but since everything in the model is linear in current,
the result can be freely scaled to whatever current is wanted.

Detailed view of a single gate voltage
--------------------------------------

If a single gate voltage is given by the `--gate_v` parameter (eg `--gate_v=2.2`), a
calculation will be made for this particular gate voltage and the result will be plotted
in a window, additionally the voltage between the two voltmeter electrodes will be printed
in the terminal.

Gate sweep
----------

If a gate range is given by the `--gate_v` parameter in the form of gate_low, gate_high,
stepsize (eg `--gate_v=-2,2,0.1`), a gate sweep will be performed and plotted.
Notice that this involves (gate_high - gate_low)/stepsize calulations and thus it
will take this factor longer time to perform than a single calculation.

Examples
--------

The two configurations from Figure 19 in the notes can be obtained by the following two commands::

 python main.py --gate_v=0 --current_in=125,10 --current_out=125,240 --vmeter_low=5,50 --vmeter_high=5,200 250
 python main.py --gate_v=0 --current_in=245,5 --current_out=245,245 --vmeter_low=5,50 --vmeter_high=5,200 250

A gate sweep of the same configuration can be done like this::
  
 python main.py --gate_v=-5,5,0.1 --current_in=125,10 --current_out=125,240 --vmeter_low=5,50 --vmeter_high=5,200 250

As a reference of performance, this command has an execution time of apprixmately 2.5 minutes on the author's computer.
 
Overview of included files
==========================

This package consists of a number of python files, a short introduction to each file
is given below. Also, each file can be executed individiually, for files mainly intended
as helper-files, this will print a short description as well as some output usefull for
development and debugging.

The included files are:
 * `example_matrix`: Two examples of proper networks as opposed to the square
   approximation otherwise used.
 * `resistor_network_calculator_base`: Base class for the calculator, although
   currently only a single calculator exists. 
 * `resistor_network_calculator`: Contains the actual code that sets the network
   and performs the calculations
 * `main`: The main execuatable file to be used to perform actual simulations.
 * `statics/`: The folder containing the input images. A number of examples
   is provided in the repository.
   
Limitations
===========

This software is still under development and has a number of limitations:

 * Missing validation of size-input from command prompt
 * Names of input files are harcodet as `doping.png` and `conductor.png`.
 * Currently (and maybe forever), only rectangular networks are supported.

