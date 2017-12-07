#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
This example demonstrates how to work with SPC files
"""

import logging
import os
import numpy as np
from pytrip import spc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # give some output on what is going on.


# s = spc.SPC('../../tmp/trip/DATA/RF0MM/C.H2O.MeV05000.spc')
# s.read_data()
# 
# s.write_spc("1.spc")
# s.write_spc()
# 
# ss = spc.SPCCollection('../../tmp/trip/DATA/RF0MM/')
# ss.read()
# ss.dirname = '2'
# ss.write()

#
dir_with_energies = 'elettra'
target_dir = 'dir_to_save_spc'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
spc_collection = spc.SPCCollection(target_dir)

for energy_dir in os.listdir(dir_with_energies):
    if not os.path.isdir(os.path.join(dir_with_energies, energy_dir)):
        continue
    dir_with_depths = os.path.join(dir_with_energies, energy_dir)

    fname = "{pp:s}.{tt:s}.{uuu:s}{eeeee:05d}.spc".format(
        pp="1H",
        tt="H2O",
        uuu="MeV",
        eeeee=int(energy_dir)
    )

    spc_object = spc.SPC(os.path.join(target_dir))
    spc_object.endian = 1
    spc_object.energy = float(energy_dir)
    spc_object.filedate = "TODO"
    spc_object.filetype = 'SPCM'
    spc_object.fileversion = '19980704'
    spc_object.ndsteps = 0
    spc_object.norm = 1.0
    spc_object.peakpos = 15.77
    spc_object.projname = "1H"
    spc_object.targname = "H2O"

    for depth_file in os.listdir(dir_with_depths):
        if not os.path.isfile(os.path.join(dir_with_depths, depth_file)):
            continue

        depth_id = int(depth_file.split('.')[-2].split('_')[-1])

        depthblock = spc.DBlock()
        depthblock.dsnorm = 1.0
        depthblock.species = []
        depthblock.depth = depth_id

        datafile = os.path.join(dir_with_depths, depth_file)
        with open(datafile, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            print(i, line)
            z = i + 2

            items = [float(x) for x in line.split()]

            specieblock = spc.SBlock()
            specieblock.z = float(z)
            specieblock.a = specieblock.z
            specieblock.lz = int(specieblock.z)
            specieblock.la = int(specieblock.a)
            specieblock.nc = 0

            specieblock.ebindata = np.arange(start=0, stop=150, step=1, dtype=np.float64)
            specieblock.histdata = np.zeros_like(specieblock.ebindata[:-1],dtype=np.float64)
            specieblock.histdata[:len(items)] = items
            specieblock.rcumdata = np.zeros_like(specieblock.ebindata,dtype=np.float64)
            specieblock.rcumdata[1:] = np.cumsum(specieblock.histdata)
            specieblock.rcumdata /= np.max(specieblock.rcumdata)

            specieblock.dscum = np.sum(specieblock.histdata)

            specieblock.ne = specieblock.histdata.size

            depthblock.species.append(specieblock)
            depthblock.nparts += 1

        spc_object.data.append(depthblock)
        spc_object.ndsteps += 1

    spc_object.write_spc(os.path.join(target_dir, fname))

    test_scp = spc.SPC(os.path.join(target_dir, fname))
    test_scp.read_spc()

    print("2+2")

