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
	self.ini_plot()

	self.setSize()
        self.SetColor()
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

	self.ini_plot()
        for i in msg.data:
            voi = msg.data[i]
            data = self.LVHs[voi["name"]]
            color = [float(c)/255 for c in voi["color"]]
            self.subplot.plot(data[0][:],data[1][:],label=voi["name"],color=color)
	self.subplot.legend(fancybox=True,prop={'size':8})
        self.canvas.draw()
        self.checkedstructures = msg.data

    def OnStructureSelect(self, msg):
        """Load the constraints for the currently selected structure."""
