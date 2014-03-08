#!/usr/bin/env python
# -*- coding: ISO-8859-1 -*-

import wx
from wx.xrc import XmlResource, XRCCTRL, XRCID
from wx.lib.pubsub import Publisher as pub
from dicompyler import guiutil, util
from dicompyler import dvhdata, guidvh
from dicompyler import wxmpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np
import os

def pluginProperties():
    props = {}
    props['name'] = 'LVH'
    props['description'] = ""
    props['author'] = 'Jakob Toftegaard'
    props['version'] = "0.1"
    props['plugin_type'] = 'main'
    props['plugin_version'] = 1
    props['min_dicom'] = ['LVHs']
    props['recommended_dicom'] = ['rtss', 'rtdose', 'rtplan']

    return props

def pluginLoader(parent):
    """Function to load the plugin."""

    # Load the XRC file for our gui resources
    datapath = guiutil.get_data_dir()
    userpath = os.path.join(datapath, 'plugins/TripLVH/lvh.xrc');
    res = XmlResource(userpath)

    panelLVH = res.LoadPanel(parent, 'pluginLVH')
    panelLVH.Init(res,parent)

    return panelLVH

class pluginLVH(wx.Panel):
    """Plugin to display DVH data with adjustable constraints."""

    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)

    def Init(self, res,parent):
        """Method called after the panel has been initialized."""
        self.parent = parent
        self.plotPanel = XRCCTRL(self,'plotPanel')

        self.Bind(wx.EVT_IDLE, self._onIdle)
        #self.Bind(wx.EVT_SIZE, self._onSize)
        self.figure = Figure( None, 100 )
        self.canvas = FigureCanvasWxAgg( self.plotPanel, -1, self.figure )
        self.canvas.callbacks.connect('button_press_event',self.OnMouseDown)
        self.canvas.callbacks.connect('key_press_event',self.OnKeyPress)
        self.canvas.callbacks.connect('scroll_event',self.OnMouseScroll)

        self.ini_plot()

        self.setSize()
        self.SetColor()
        self.selected_voi = None
        # Set up pubsub
        pub.subscribe(self.OnUpdatePatient, 'patient.updated.raw_data')
        pub.subscribe(self.OnStructureCheck, 'structures.checked')
        pub.subscribe(self.OnStructureSelect, 'structure.selected')
    def _onIdle(self,evt):
        self.setSize()
        self.canvas.draw()
    def setSize(self):
        size = self.parent.GetClientSize()
        size[1] = size[1] - 30
        pixels = tuple( size )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )
    def SetColor( self, rgbtuple=None ):
        """Set figure and canvas colours to be the same."""
        if rgbtuple is None:
            rgbtuple = wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ).Get()
        clr = [c/255. for c in rgbtuple]
        self.figure.set_facecolor( clr )
        self.figure.set_edgecolor( clr )
        self.canvas.SetBackgroundColour( wx.Colour( *rgbtuple ) )
    def OnUpdatePatient(self, msg):
        """Update and load the patient data."""
        self.LVHs = msg.data["LVHs"]
        # show an empty plot when (re)loading a patient
    def OnMouseScroll(self,evt):
        if self.selected_voi is None:
            return
        steps = evt.step
        self.nearest_i = self.nearest_i+steps
        lvh = self.LVHs[self.selected_voi]
        if self.nearest_i < 0:
            self.nearest_i = 0
        elif self.nearest_i >= len(lvh[0]):
            self.nearest_i = len(lvh[0])-1

        self.nearest_point = [lvh[0][self.nearest_i],lvh[1][self.nearest_i]]
        self.DrawLVH()
    def OnKeyPress(self,evt):
        return
    def OnMouseDown(self,evt):
        if evt.button is 1:
            self.FindNearestPoint([evt.xdata,evt.ydata])
        elif evt.button is 3:
            self.selected_voi = ""
            self.nearest_point = None

    def OnDestroy(self, evt):
        """Unbind to all events before the plugin is destroyed."""

        pub.unsubscribe(self.OnUpdatePatient)
        pub.unsubscribe(self.OnStructureCheck)
        pub.unsubscribe(self.OnStructureSelect)
    def ini_plot(self):
        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        self.subplot.grid(True)

        self.subplot.set_title("LVH Plot",fontsize=14)
        self.subplot.set_xlabel("LET (keV/um)",fontsize=10)
        self.subplot.set_ylabel("Vol (%)",fontsize=10)
    def OnStructureCheck(self, msg):
        """When a structure changes, update the interface and plot."""

        # Make sure that the volume has been calculated for each structure
        # before setting it

        self.selected_voi = None
        self.selected_vois = []
        self.selected_vois_name = []
        for i in msg.data:
            voi = msg.data[i]
            self.selected_vois.append(voi)
            self.selected_vois_name.append(voi["name"])
        self.DrawLVH()
    def DrawLVH(self):
        self.ini_plot()
        for voi in self.selected_vois:
            data = self.LVHs[voi["name"]]
            color = [float(c)/255 for c in voi["color"]]
            self.subplot.plot(data[0][:],data[1][:],label=voi["name"],color=color)
        if self.selected_voi is not None:
            font = matplotlib.font_manager.FontProperties(size=8)
            self.subplot.plot([self.nearest_point[0]],[self.nearest_point[1]],'o')
            text = "Vol: " + unicode("%.2f"% (self.nearest_point[1]*100)) + "%\nLET: " + unicode("%.2f"%(self.nearest_point[0]))+" keV/um"
            self.subplot.annotate(text, xy=(self.nearest_point[0], self.nearest_point[1]), xytext=(self.nearest_point[0]+10, self.nearest_point[1]),
                                        arrowprops=dict(arrowstyle="->",facecolor='black'),fontproperties=font)

        self.subplot.legend(fancybox=True,prop={'size':8})
        self.canvas.draw()
    def FindNearestPoint(self,pos):
        dist = float('inf')
        nearest = [0.0,0.0]
        self.selected_voi = None
        for key,lvhs in self.LVHs.items():
            if key in self.selected_vois_name:
                i = 0
                for x,y in zip(lvhs[0],lvhs[1]):
                    dist2 = (pos[0]-x)**2+((pos[1]-y)*pos[0])**2
                    if dist > dist2:
                        nearest = [x,y]
                        dist = dist2
                        self.selected_voi = key
                        self.nearest_i = i
                    i= i+1
        self.nearest_point = nearest
        self.DrawLVH()

    def OnStructureSelect(self, msg):
        """Load the constraints for the currently selected structure."""
