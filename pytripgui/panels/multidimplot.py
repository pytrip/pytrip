"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
import wx
from wx.lib.pubsub import Publisher as pub
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading

from numpy import ogrid, sin
from traits.api import HasTraits, Instance
from traitsui.api import View, Item

from mayavi.sources.api import ArraySource
from mayavi.modules.api import IsoSurface

from mayavi.core.ui.api import SceneEditor, MlabSceneModel


class MultiDimPanel(HasTraits):
    scene = Instance(MlabSceneModel, ())

    # The layout of the panel created by Traits
    view = View(Item('scene', editor=SceneEditor(), resizable=True,
                     show_label=False),
                resizable=True)

    def __init__(self, parent):
        HasTraits.__init__(self)
        # Create some data, and plot it using the embedded scene's engine

        self.parent = parent
        pub.subscribe(self.on_patient_loaded, "patient.loaded")
        pub.subscribe(self.voi_changed, "voi.selection_changed")

    def Init(self):
        pass

    def on_patient_loaded(self, msg):
        self.data = msg.data
        ct_data = self.data.voxelplan_images
        x, y, z = ogrid[0:1:ct_data.dimx, 0:1:ct_data.dimy, 0:1:ct_data.dimz]
        src = ArraySource(scalar_data=ct_data.cube)
        self.scene.engine.add_source(src)
        src.add_module(IsoSurface())

    def voi_changed(self, msg):
        pass
