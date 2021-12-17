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
This module provides classes to handle SPC data

SPC binary file format is described here:
http://bio.gsi.de/DOCS/TRiP98/PRO/DOCS/trip98fmtspc.html


"""

import os
from enum import Enum
import logging
import numpy as np
import sys


class SPCTagCode(Enum):
    FILETYPE = 1
    FILEVERSION = 2
    FILEDATE = 3
    TARGNAME = 4
    PROJNAME = 5
    BEAM_ENERGY = 6
    PEAK_POSITION = 7
    NORMALIZATION = 8
    NO_OF_DEPTH_STEPS = 9
    DEPTH = 10
    DEPTH_STEP_NORMALIZATION = 11
    NO_OF_PARTICLE_SPECIES = 12
    ZA = 13
    CUMULATED_NO_OF_FRAGMENTS = 14
    NC = 15
    NO_OF_ENERGY_BINS = 16
    ENERGY_BIN_VALUES = 17
    ENERGY_BIN_POINTER = 18
    SPECTRUM_BIN_VALUES = 19
    CUMULATED_SPECTRUM_BIN_VALUES = 20


class SPCCollection(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.data = {}  # key - energy, value - SPC object

    def read(self):
        files = os.listdir(self.dirname)
        for file in files:
            if file.endswith(".spc"):
                energy = int(file.split('.')[-2][3:]) / 100
                print("energy", energy)
                self.data[energy] = SPC(os.path.join(self.dirname, file))
                self.data[energy].read_spc()

    def write(self):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        for spc_object in self.data.values():
            fname = "{pp:s}.{tt:s}.{uuu:s}{eeeee:05d}.spc".format(pp=spc_object.projname,
                                                                  tt=spc_object.targname,
                                                                  uuu="MeV",
                                                                  eeeee=int(100 * spc_object.energy))
            print(fname)
            spc_object.write_spc(os.path.join(self.dirname, fname))


class SPC(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def read_spc(self):
        logging.info("Reading {}".format(self.filename))
        self.read_data()

    def write_spc(self, filename=None):
        fname = filename
        if fname is None:
            fname = "{pp:s}.{tt:s}.{uuu:s}{eeeee:05d}.spc".format(pp=self.projname,
                                                                  tt=self.targname,
                                                                  uuu="MeV",
                                                                  eeeee=int(100 * self.energy))
        with open(fname, "wb") as fd:

            # filetype 1
            # <filetype> is an 80-byte ASCII character string starting with "SPCM" or "SPCI",
            # specifying big ("Motorola") or little ("Intel") endian byte order, respectively.
            tag = Tag(fd)
            tag.code = SPCTagCode.FILETYPE.value
            tag.size = 80
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.filetype, dtype="{:s}a{:d}".format(tag._ste, tag.size)))

            # fileversion 2
            # <fileversion> is an 80-byte ASCII character string specifying the file format version as yyyymmdd.
            # 19980704 is the "classic" format (fixed energy range),
            # whereas 20020107 is reserved for future possible variable energy range.
            tag = Tag(fd)
            tag.code = SPCTagCode.FILEVERSION.value
            tag.size = 80
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.fileversion, dtype="{:s}a{:d}".format(tag._ste, tag.size)))

            # filedate 3
            # <filedate> is an 80-byte ASCII character string with the file creation date
            # as returned by the ctime() function. (<dow> <mmm> <dd> <hh>:<mm>:<ss> <yyyy>)
            tag = Tag(fd)
            tag.code = SPCTagCode.FILEDATE.value
            tag.size = 80
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.filedate, dtype="{:s}a{:d}".format(tag._ste, tag.size)))

            # targname 4, projname 5
            # <targname> and <projname> are the names of target ("H2O") and projectile ("12C6"), respectively.
            # Since both can have any length, they are padded to the right with binary zeroes
            # up to the next 8-byte boundary.
            tag = Tag(fd)
            tag.code = SPCTagCode.TARGNAME.value
            tag.size = 8  # TODO to be padded
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.targname, dtype="{:s}a{:d}".format(tag._ste, tag.size)))

            tag = Tag(fd)
            tag.code = SPCTagCode.PROJNAME.value
            tag.size = 8  # TODO to be padded
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.projname, dtype="{:s}a{:d}".format(tag._ste, tag.size)))

            # no 6 double; beam energy [MeV/u]
            tag = Tag(fd)
            tag.code = SPCTagCode.BEAM_ENERGY.value
            tag.size = 8
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.energy, dtype="{:s}f{:d}".format(tag._ste, tag.size)))

            # no 7, double; peak position [g/cm**2]
            tag = Tag(fd)
            tag.code = SPCTagCode.PEAK_POSITION.value
            tag.size = 8
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.peakpos, dtype="{:s}f{:d}".format(tag._ste, tag.size)))

            # no 8, double; normalization, usually =1
            tag = Tag(fd)
            tag.code = SPCTagCode.NORMALIZATION.value
            tag.size = 8
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.norm, dtype="{:s}f{:d}".format(tag._ste, tag.size)))

            # no 9, 8-byte unsigned integer; number of depth steps.
            tag = Tag(fd)
            tag.code = SPCTagCode.NO_OF_DEPTH_STEPS.value
            tag.size = 8
            tag.endian = self.endian
            tag.set_tag()
            fd.write(np.asarray(self.ndsteps, dtype="{:s}i{:d}".format(tag._ste, tag.size)))

            for dblock in self.data:
                # no 10, double; depth [g/cm**2]
                tag = Tag(fd)
                tag.code = SPCTagCode.DEPTH.value
                tag.size = 8
                tag.endian = self.endian
                tag.set_tag()
                fd.write(np.asarray(dblock.depth, dtype="{:s}f{:d}".format(tag._ste, tag.size)))

                # no 11, double; normalization for this depth step, usually =1.
                tag = Tag(fd)
                tag.code = SPCTagCode.DEPTH_STEP_NORMALIZATION.value
                tag.size = 8
                tag.endian = self.endian
                tag.set_tag()
                fd.write(np.asarray(dblock.dsnorm, dtype="{:s}f{:d}".format(tag._ste, tag.size)))

                # no 12, 8-byte unsigned integer; number of particle species.
                tag = Tag(fd)
                tag.code = SPCTagCode.NO_OF_PARTICLE_SPECIES.value
                tag.size = 8
                tag.endian = self.endian
                tag.set_tag()
                fd.write(np.asarray(dblock.nparts, dtype="{:s}u{:d}".format(tag._ste, tag.size)))

                ebins = {}

                for specie_item_no, specie in enumerate(dblock.species):
                    # no 13, double Z, double A, long Z, long A;
                    tag = Tag(fd)
                    tag.code = SPCTagCode.ZA.value
                    tag.size = 24
                    tag.endian = self.endian
                    tag.set_tag()
                    fd.write(np.asarray([specie.z, specie.a], dtype="{:s}f8".format(tag._ste)))
                    fd.write(np.asarray([specie.lz, specie.la], dtype="{:s}u4".format(tag._ste)))

                    # no 14, double
                    # The scalar Cum value is the cumulated number (running sum) of fragments,
                    # i.e. the species sum over Cum[nE].
                    # This number may exceed 1, since for an incoming primary particle many secondaries are created.
                    tag = Tag(fd)
                    tag.code = SPCTagCode.CUMULATED_NO_OF_FRAGMENTS.value
                    tag.size = 8
                    tag.endian = self.endian
                    tag.set_tag()
                    fd.write(np.asarray([specie.dscum], dtype="{:s}f{:d}".format(tag._ste, tag.size)))

                    # no 15, 8-byte unsigned integer;
                    # <nC> is reserved for later use, so that lateral scattering for each fragment can be included.
                    # At present nC=0.
                    tag = Tag(fd)
                    tag.code = SPCTagCode.NC.value
                    tag.size = 8
                    tag.endian = self.endian
                    tag.set_tag()
                    fd.write(np.asarray([specie.nc], dtype="{:s}u{:d}".format(tag._ste, tag.size)))

                    # no 16, 8-byte unsigned integer;
                    # 8-byte unsigned integer; number of energy bins for this species
                    tag = Tag(fd)
                    tag.code = SPCTagCode.NO_OF_ENERGY_BINS.value
                    tag.size = 8
                    tag.endian = self.endian
                    tag.set_tag()
                    fd.write(np.asarray([specie.ne], dtype="{:s}u{:d}".format(tag._ste, tag.size)))

                    # If a species data block is flagged as EREFCOPY, its energy bins are not stored,
                    # but rather a reference index <lSRef> (0..<nS>-1) to a species with identical energy binning.
                    # This helps to reduce space requirements, since fragment spectra may share the same energy bins.
                    # If the energy bins for the species under consideration are not a reference copy,
                    # <nE>+1 double precision bin border values are stored.
                    compatible_ebins_index = [i for i in ebins if np.array_equal(specie.ebindata, ebins[i])]
                    if not compatible_ebins_index:
                        ebins[specie_item_no] = specie.ebindata

                        # no 17, double; energy bin values
                        tag = Tag(fd)
                        tag.code = SPCTagCode.ENERGY_BIN_VALUES.value
                        tag.size = 8 * (specie.ne + 1)
                        tag.endian = self.endian
                        tag.set_tag()
                        fd.write(np.asarray(specie.ebindata, dtype="{:s}f8".format(tag._ste)))
                    else:
                        # no 18,  8-byte unsigned integer
                        tag = Tag(fd)
                        tag.code = SPCTagCode.ENERGY_BIN_POINTER.value
                        tag.size = 8
                        tag.endian = self.endian
                        tag.set_tag()
                        fd.write(np.asarray([compatible_ebins_index[0]], dtype="{:s}u{:d}".format(tag._ste, tag.size)))

                    # no 19,  double; spectrum bin values
                    tag = Tag(fd)
                    tag.code = SPCTagCode.SPECTRUM_BIN_VALUES.value
                    tag.size = 8 * specie.ne
                    tag.endian = self.endian
                    tag.set_tag()
                    fd.write(np.asarray(specie.histdata, dtype="{:s}f8".format(tag._ste)))

                    # no 20,  double; running cumulated spectrum bin values
                    tag = Tag(fd)
                    tag.code = SPCTagCode.CUMULATED_SPECTRUM_BIN_VALUES.value
                    tag.size = 8 * (specie.ne + 1)
                    tag.endian = self.endian
                    tag.set_tag()
                    fd.write(np.asarray(specie.rcumdata, dtype="{:s}f8".format(tag._ste)))

    def read_data(self):
        with open(self.filedate, "rb") as fd:
            t = Tag(fd)
            db = []

            while t.code < SPCTagCode.NO_OF_DEPTH_STEPS.value:
                t.get_tag()
                self.endian = t.endian
                pl = self.get_payload(fd, t)

            for _ in range(self.ndsteps):
                # no 10, double; depth [g/cm**2]
                t.get_tag()
                pl = self.get_payload(fd, t)

                # construct the main object
                if t.code == SPCTagCode.DEPTH.value:
                    db.append(DBlock())
                    db[-1].depth = pl

                    # no 11, double; normalization for this depth step, usually =1.
                    t.get_tag()
                    if t.code == SPCTagCode.DEPTH_STEP_NORMALIZATION.value:
                        pl = self.get_payload(fd, t)
                        db[-1].dsnorm = pl

                    # no 12, 8-byte unsigned integer; number of particle species.
                    t.get_tag()
                    if t.code == SPCTagCode.NO_OF_PARTICLE_SPECIES.value:
                        pl = self.get_payload(fd, t)
                        db[-1].nparts = pl

                    for _ in range(db[-1].nparts):  # loop over all species
                        # no 13, double Z, double A, long Z, long A;
                        t.get_tag()
                        if t.code == SPCTagCode.ZA.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species.append(SBlock())
                            db[-1].species[-1].z = pl[0]
                            db[-1].species[-1].a = pl[1]
                            db[-1].species[-1].lz = int(pl[2])
                            db[-1].species[-1].la = int(pl[3])

                        # no 14, double
                        # The scalar Cum value is the cumulated number (running sum) of fragments,
                        # i.e. the species sum over Cum[nE].
                        # This number may exceed 1, since for an incoming primary particle many secondaries are created.
                        t.get_tag()
                        if t.code == SPCTagCode.CUMULATED_NO_OF_FRAGMENTS.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].dscum = pl

                        # no 15, 8-byte unsigned integer;
                        # <nC> is reserved for later use, so that lateral scattering for each fragment can be included.
                        # At present nC=0.
                        t.get_tag()
                        if t.code == SPCTagCode.NC.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].nc = pl

                        # no 16, 8-byte unsigned integer;
                        # 8-byte unsigned integer; number of energy bins for this species
                        t.get_tag()
                        if t.code == SPCTagCode.NO_OF_ENERGY_BINS.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].ne = pl

                        # no 17, double; energy bin values
                        t.get_tag()
                        if t.code == SPCTagCode.ENERGY_BIN_VALUES.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].ebindata = pl
                        # no 18,  8-byte unsigned integer
                        if t.code == SPCTagCode.ENERGY_BIN_POINTER.value:  # both tags will not be present
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].ebindata = db[0].species[pl].ebindata

                        # no 19,  double; spectrum bin values
                        t.get_tag()
                        if t.code == SPCTagCode.SPECTRUM_BIN_VALUES.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].histdata = pl

                        # no 20,  double; running cumulated spectrum bin values
                        t.get_tag()
                        if t.code == SPCTagCode.CUMULATED_SPECTRUM_BIN_VALUES.value:
                            pl = self.get_payload(fd, t)
                            db[-1].species[-1].rcumdata = pl

                self.data = db

    def get_payload(self, fd, tag):
        cnt = 1

        if self.endian == 0:
            ste = '<'  # little endian
        else:
            ste = '>'  # big endian

        # filetype 1
        # <filetype> is an 80-byte ASCII character string starting with "SPCM" or "SPCI",
        # specifying big ("Motorola") or little ("Intel") endian byte order, respectively.
        if tag.code == SPCTagCode.FILETYPE.value:
            sdtype = ste + 'a' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0].decode("UTF-8")
            self.filetype = payload
            return payload

        # fileversion 2
        # <fileversion> is an 80-byte ASCII character string specifying the file format version as yyyymmdd.
        # 19980704 is the "classic" format (fixed energy range),
        # whereas 20020107 is reserved for future possible variable energy range.
        if tag.code == SPCTagCode.FILEVERSION.value:
            sdtype = ste + 'a' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0].decode("UTF-8")
            self.fileversion = payload
            return payload

        # filedate 3
        # <filedate> is an 80-byte ASCII character string with the file creation date
        # as returned by the ctime() function. (<dow> <mmm> <dd> <hh>:<mm>:<ss> <yyyy>)
        if tag.code == SPCTagCode.FILEDATE.value:
            sdtype = ste + 'a' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0].decode("UTF-8")
            self.filedate = payload
            return payload

        # targname 4, projname 5
        # <targname> and <projname> are the names of target ("H2O") and projectile ("12C6"), respectively.
        # Since both can have any length, they are padded to the right with binary zeroes
        # up to the next 8-byte boundary.
        if tag.code == SPCTagCode.TARGNAME.value:
            sdtype = ste + 'a' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0].decode("UTF-8")
            self.targname = payload
            return payload

        if tag.code == SPCTagCode.PROJNAME.value:
            sdtype = ste + 'a' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0].decode("UTF-8")
            self.projname = payload
            return payload

        # no 6 double; beam energy [MeV/u]
        if tag.code == SPCTagCode.BEAM_ENERGY.value:
            sdtype = ste + 'f' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            self.energy = payload
            return payload

        # no 7, double; peak position [g/cm**2]
        if tag.code == SPCTagCode.PEAK_POSITION.value:
            sdtype = ste + 'f' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            self.peakpos = payload
            return payload

        # no 8, double; normalization, usually =1
        if tag.code == SPCTagCode.NORMALIZATION.value:
            sdtype = ste + 'f' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            self.norm = payload
            return payload

        # no 9, 8-byte unsigned integer; number of depth steps.
        if tag.code == SPCTagCode.NO_OF_DEPTH_STEPS.value:
            sdtype = ste + 'u' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            self.ndsteps = payload
            return payload

        # no 10, double; depth [g/cm**2]
        if tag.code == SPCTagCode.DEPTH.value:
            sdtype = ste + 'f' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.depth = payload
            return payload

        # no 11, double; normalization for this depth step, usually =1.
        if tag.code == SPCTagCode.DEPTH_STEP_NORMALIZATION.value:
            sdtype = ste + 'f' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.dsnorm = payload
            return payload

        # no 12, 8-byte unsigned integer; number of particle species.
        if tag.code == SPCTagCode.NO_OF_PARTICLE_SPECIES.value:
            sdtype = ste + 'u' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.nparts = payload
            return payload

        # no 13, double Z, double A, long Z, long A;
        if tag.code == SPCTagCode.ZA.value:
            payload = np.zeros(4)

            sdtype = ste + 'f8'
            payload[0] = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))
            payload[1] = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))

            # todo: fix types
            sdtype = ste + 'u4'
            payload[2] = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))
            payload[3] = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))

            return payload

        # no 14, double
        # The scalar Cum value is the cumulated number (running sum) of fragments,
        # i.e. the species sum over Cum[nE].
        # This number may exceed 1, since for an incoming primary particle many secondaries are created.
        if tag.code == SPCTagCode.CUMULATED_NO_OF_FRAGMENTS.value:
            sdtype = ste + 'f' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.dsnorm = payload
            return payload

        # no 15, 8-byte unsigned integer;
        # <nC> is reserved for later use, so that lateral scattering for each fragment can be included.
        # At present nC=0.
        if tag.code == SPCTagCode.NC.value:
            sdtype = ste + 'u' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.nc = payload
            return payload

        # no 16, 8-byte unsigned integer;
        # 8-byte unsigned integer; number of energy bins for this species
        if tag.code == SPCTagCode.NO_OF_ENERGY_BINS.value:
            sdtype = ste + 'u' + str(tag.size)
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.ne = payload
            return payload

        # no 17, double; energy bin values
        if tag.code == SPCTagCode.ENERGY_BIN_VALUES.value:
            sdtype = ste + 'f8'
            cnt = tag.size // 8
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))
            # self.ebindata = payload
            return payload

        # no 18,  8-byte unsigned integer
        # EREF: if tag is set, then use copy of ebin from species os stated in the file.
        if tag.code == SPCTagCode.ENERGY_BIN_POINTER.value:
            sdtype = ste + 'u8'
            cnt = tag.size // 8
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))[0]
            # self.eref = payload
            return payload

        # no 19,  double; spectrum bin values
        # H[nE]: spectrum contents divided by the bin width
        if tag.code == SPCTagCode.SPECTRUM_BIN_VALUES.value:
            sdtype = ste + 'f8'
            cnt = tag.size // 8
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))
            # self.histdata = payload
            return payload

        # no 20,  double; running cumulated spectrum bin values
        if tag.code == SPCTagCode.CUMULATED_SPECTRUM_BIN_VALUES.value:
            sdtype = ste + 'f8'
            cnt = tag.size // 8
            payload = np.fromfile(fd, count=cnt, dtype=np.dtype(sdtype))
            # self.cum = payload
            return payload

        return None


class DBlock(object):  # the depth block
    def __init__(self):
        self.depth = 0.0
        self.norm = 0.0
        self.nparts = 0
        self.species = []  # list of species, for SBlock class


class SBlock(object):  # particle species block
    def __init__(self):
        self.z = 0.0
        self.a = 0.0
        self.lz = 0
        self.la = 0
        self.nc = 0
        self.ne = 0
        self.ebindata = []
        self.histdata = []
        self.rcumdata = []


class Tag(object):
    def __init__(self, fd):
        self.fd = fd
        self.code = 0
        self.size = 0
        self.endian = -1  # 0 little endian, 1 big endian, -1 unknown
        self._ste = ''

    def set_tag(self):
        if self.endian == 1:  # big endian (Motorola)
            self._ste = '>'
        else:
            self._ste = '<'  # little endian (Intel)
        self.fd.write(np.asarray([self.code, self.size], dtype=self._ste + "u4"))

    def get_tag(self):
        if self.endian == -1:
            # try Intel first, little endian
            self.endian = 0
            self._ste = '<'  # little endian read
            code = np.fromfile(self.fd, count=1, dtype=np.dtype(self._ste + 'u4'))[0]
            self.fd.seek(-4, 1)  # rewind 4 bytes

            if code < 1 or code > 20:  # if fail Intel, then:
                self.endian = 1  # big endian
                self._ste = '>'  # big endian read
                code = np.fromfile(self.fd, count=1, dtype=np.dtype(self._ste + 'u4'))[0]
                self.fd.seek(-4, 1)  # rewind 4 bytes, retry
                if code < 1 or code > 20:
                    print("Error: bad format in SPC file.")
                    sys.exit()
                else:
                    # print("Found big-endian format.")
                    pass

        self.code = np.fromfile(self.fd, count=1, dtype=np.dtype(self._ste + 'u4'))[0]
        self.size = np.fromfile(self.fd, count=1, dtype=np.dtype(self._ste + 'u4'))[0]
