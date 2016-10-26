.. _examples:

========
Examples
========

.. rubric:: Code snippets demonstrating PyTRiP capabilities.

Example 00 - Cube arithmetic
----------------------------

This example demonstrates simple arithmetic on dose- and LET-cubes.
Two dose cubes from two fields are summed to generate a new total dose cube.

The two LET-cubes from the two fields are combined to calculate the total dose-averaged LET in the resulting treatment plan.
All data are saved to disk.

.. literalinclude:: ../examples/example00_basic.py
   :language: python
   :linenos:
   :lines: 19-


Example 01 - Handling structures
--------------------------------

This example shows how one can select a region inside a CTX data cube using a VDX file, and perform some manipulation of it.

.. literalinclude:: ../examples/example01_vdx.py
   :language: python
   :linenos:
   :lines: 19-

Working with dose cubes is fully analogous to the CTX cubes.
	   
Example 02 - TRiP execution
---------------------------

In this example, we demonstrate how to actually perform a treatment plan using TRiP98.
Most of the lines concern with the setup of TRiP.

.. literalinclude:: ../examples/example02_trip98.py
   :language: python
   :linenos:
   :lines: 19-

