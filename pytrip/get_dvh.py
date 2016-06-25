#! /usr/bin/env python

from pytrip.vdx import *


# from numpy import zeros
# ~ from pylab import *

class GetDvh(object):
    'This class produces DVHs.'

    def build_DVH(self, data, _xnorm):
        # data is a flat array

        # number of bins to use:
        _binx = 110
        _biny = 110

        # volume
        _vol = len(data)

        self.bins = _vol  # the amount of bins in VOI
        self.bins_dge95 = 0
        self.bins_dlt95 = 0
        self.bins_dgt107 = 0

        print("Cube size:", _vol)

        _DVHx = arange(0, _binx, 1)  # last element ist 109
        _DVHy = zeros(110)

		# normalize contents:
        data = 100.0 * data / float(_xnorm)

        #	print("DVH: shape", shape(data)


        for _element in data:
            #           print("DVH: element shalpe",shape(_element)
            if _element >= 95:
                self.bins_dge95 += 1
            else:
                self.bins_dlt95 += 1
            if _element > 107.0:
                self.bins_dgt107 += 1
                if _element <= 109.0:  # prevent exceed
                    for iii in range(int(round(_element))):
                        _DVHy[iii] += 1

		# normalize y-scale
        _DVHy = 100.0 * _DVHy / float(_vol)

        self.bins_95_107 = self.bins_dge95 - self.bins_dgt107
        self.percent_95_107 = 100 * self.bins_95_107 / self.bins
        self.percent_gt107 = 100 * self.bins_dgt107 / self.bins
        self.percent_lt95 = 100 * self.bins_dlt95 / self.bins

        return [_DVHx, _DVHy]

    def __init__(self, dos, _xnorm, vdx, _cv):

        # generate maske, if not already done so
        vdx._calc_voi(_cv)

        # extract VOI array containing the data
        _mask_indicies = nonzero(vdx.mask[_cv].flatten())
        print("DVH: vdx mask:", shape(vdx.mask[_cv]))
        print("DVH: mask ind:", shape(_mask_indicies))

        print("DVH: dos shape:", shape(dos.cube.flatten()))
        print("DVH: voi shape:", shape(vdx.mask[_cv].flatten()))

        self.array = take(dos.cube.flatten(), _mask_indicies)[0]  # why this zero?
        print("DVH: array shape:", shape(self.array))
		# print self.array

        self.DVH = self.build_DVH(self.array, _xnorm)
        self.name = vdx.voi[_cv].name

        self.voxel_size = vdx.header.pixel_size * vdx.header.pixel_size * vdx.header.slice_distance
        self.volume = self.bins * self.voxel_size
        self.volume_cm3 = self.volume * 0.001
        self.min = self.array.min()
        self.max = self.array.max()
        self.mean = self.array.mean()
        self.std = self.array.std()

    def analyze(self):
        print("-----------------------------------------------------------------")
        print("                          VOI Analysis")
        print(" Name:", self.name)
        print(" VOI Volume: %.3f cm3" % self.volume_cm3)
        print(" in", self.bins, "bins.")
        print(" Min / Max Dose: %.2f / %.2f" % (self.min, self.max))
        print(" Mean / Stdev  : %.2f / %.2f" % (self.mean, self.std))
        print(" Vol: 95 %% <= D <= 107 %% : %.1f %% in %i bins" % (self.percent_95_107, self.bins_95_107))
        print(" Vol: D > 107 %%          : %.1f %% in %i bins" % (self.percent_gt107, self.bins_dgt107))
        print(" Vol: D < 95 %%           : %.1f %% in %i bins" % (self.percent_lt95, self.bins_dlt95))
        print("-----------------------------------------------------------------")

    def plot(self):

        fig = figure()
        a1 = fig.add_subplot(111, autoscale_on=False)
        title(self.name)
        xlabel("Dose %")
        ylabel("Vol %")
        a1.set_xlim(0, 110)
        a1.set_ylim(0, 110)
        a1.plot(self.DVH[0], self.DVH[1])
        #	leg = a1.legend(self.name)
        show()
