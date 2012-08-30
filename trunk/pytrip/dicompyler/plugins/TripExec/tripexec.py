#!/usr/bin/env python

import wx, wx.grid
from wx.xrc import XmlResource, XRCCTRL, XRCID
from wx.lib.pubsub import Publisher as pub
from matplotlib import _cntr as cntr
from matplotlib import __version__ as mplversion
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import numpy as np
import os
from dicompyler import guiutil, util

def pluginProperties():
    """Properties of the plugin."""

    props = {}
    props['name'] = 'Trip Exec'
    props['description'] = ""
    props['author'] = 'Jakob Toftegaard'
    props['version'] = "0.1"
    props['plugin_type'] = 'main'
    props['plugin_version'] = 1
    props['min_dicom'] = []
    props['recommended_dicom'] = ['images', 'rtss', 'rtdose']

    return props

def pluginLoader(parent):
    """Function to load the plugin."""

    # Load the XRC file for our gui resources
    datapath = guiutil.get_data_dir()
    userpath = os.path.join(datapath, 'plugins/TripExec/tripexec.xrc')
    res = XmlResource(userpath)

    panelTripExec = res.LoadPanel(parent, 'pluginTripExec')
    panelTripExec.Init(res)

    return panelTripExec

class pluginTripExec(wx.Panel):
    """Plugin to display DICOM image, RT Structure, RT Dose in 2D."""

    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)

    def Init(self, res):
	self.ini_beams_grid()
        self.ini_dose_grid()
        self.btn_run = XRCCTRL(self,"btn_run")
        wx.EVT_BUTTON(self,XRCID('btn_run'),self.run_trip)
        pub.subscribe(self.on_structure_check, 'structures.checked')
    def run_trip(self,evt):
        print "Calculate"
    def ini_dose_grid(self):
        self.grid_dose = XRCCTRL(self,"grid_dose")
        self.dose_data = {}
	self.grid_dose.CreateGrid( 0, 3 )
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(wx.grid.GridCellBoolEditor())
        attr.SetRenderer(wx.grid.GridCellBoolRenderer())
        self.grid_dose.SetColAttr(1,attr)
        self.grid_dose.SetColSize(1,20)

        attr = wx.grid.GridCellAttr()
        attr.SetEditor(wx.grid.GridCellBoolEditor())
        attr.SetRenderer(wx.grid.GridCellBoolRenderer())
        self.grid_dose.SetColAttr(2,attr)
        self.grid_dose.SetColSize(2,20)

        attr = wx.grid.GridCellAttr()
        attr.SetReadOnly(True)
        self.grid_dose.SetColAttr(0,attr)

	self.grid_dose.EnableEditing( True )
	self.grid_dose.EnableGridLines( True )
	self.grid_dose.EnableDragGridSize( True )
	self.grid_dose.SetMargins( 0, 0 )
        self.grid_dose.SetColSize(0,0)
	self.grid_dose.SetColLabelValue( 0, u"Structure Name" )
	self.grid_dose.SetColLabelValue( 1, u"Target" )
	self.grid_dose.SetColLabelValue( 2, u"OAR" )

        self.grid_dose.SetSize((300,200))
    def add_rows_grid_dose(self,values):
        names = []
        for key in values:
            voi_name = values[key]["name"]
            names.append(voi_name)
            if voi_name not in self.dose_data.keys():
                self.grid_dose.AppendRows(1)
                self.grid_dose.SetCellValue(len(values)-1,0,voi_name)
                self.dose_data[voi_name] = {}
        if  len(values) < len(self.dose_data):
            i = 0
            for name in self.dose_data.keys():
                if name not in names:
                    self.grid_dose.DeleteRows(i)
                    del self.dose_data[name]
                else:
                    i += 1
        self.grid_dose.AutoSize()

    def on_structure_check(self,msg):
        self.add_rows_grid_dose(msg.data)
    def ini_beams_grid(self):
        self.grid_beams = XRCCTRL(self,"grid_beams")

	self.grid_beams.CreateGrid( 4, 8 )
	self.grid_beams.EnableEditing( True )
	self.grid_beams.EnableGridLines( True )
	self.grid_beams.EnableDragGridSize( True )
	self.grid_beams.SetMargins( 0, 0 )
        self.grid_beams.SetColSize(0,0)

	self.grid_beams.SetColLabelValue( 0, u"Target" )
	self.grid_beams.SetColLabelValue( 1, u"fwhm" )
	self.grid_beams.SetColLabelValue( 2, u"Gantry" )
	self.grid_beams.SetColLabelValue( 3, u"Couch" )
	self.grid_beams.SetColLabelValue( 4, u"Rastersteps" )
	self.grid_beams.SetColLabelValue( 5, u"zsteps" )
	self.grid_beams.SetColLabelValue( 6, u"doseext" )
	self.grid_beams.SetColLabelValue( 7, u"contourext" )
        self.grid_beams.SetSize((400,300))
	self.grid_beams.AutoSize()
        wx.grid.EVT_GRID_CMD_EDITOR_HIDDEN(self,XRCID('grid_beams'),self.on_change_beams_grid)
    def on_change_beams_grid(self,evt):
        return
    def OnUpdatePatient(self, msg):
        return
class TripFileGenerator:
        def __init__(self):
                self.beams = []
                self.targets = []
                self.bio = False
                self.ion = "Carbon"
                self.plan_name = "temp"
                self.dose = 68.0
                self.iter = 500
                self.eps = 1e-3
                self.geps = 1e-4
                self.bioalg = "ld"
                self.optalg = "cg"
                self.dosealg = "ap"
        def create_input_file(self):
                output = []
                output.append("time / on")
                output.append("sis  * /delete")
                output.append("hlut * /delete")
                output.append("ddd  * /delete")
                output.append("dedx * /delete")
                output.append('dedx "$TRIP98/DATA/DEDX/20040607.dedx" /read')
                output.append('hlut "$TRIP98/DATA/19990211.hlut" / read')
                output.append("scancap / offh2o(1.709) rifi(3) bolus(40.000) minparticles(5000) path(none)")
                if self.ion == "Carbon":
                        output.append('sis "$TRIP98/DATA/SIS/12C.sis" / read')
                        output.append('ddd "$TRIP98/DATA/DDD/12C/RF3MM/12C*" / read')
                        output.append('spc "$TRIP98/DATA/SPC/12C/RF3MM/12C*" / read')
                elif self.ion == "Hydrogen":
                        output.append('sis "$TRIP98/DATA/SIS/1H.sis" / read')
                        output.append('ddd "$TRIP98/DATA/DDD/1H/RF3MM/1H*" / read')
                        output.append('spc "$TRIP98/DATA/SPC/1H/RF3MM/1H*" / read')
                output.append("ct " +self.plan_name + " /read")
                targets = ""
                output.append("vdx " + self.plan_name + "  /read")
                for i,val in enumerate(self.beams):
                        field = "field " + (i+1) + " / new "
                        if "fwhm" in val:
                                field += "fwhm(%.d) "%(val["fwhm"])
                        if "raster" in val:
                                field += "raster(" + val["raster"] + ") "
                        if "couch" in val:
                                field += "couch(" + val["couch"] + ") "
                        if "gantry" in val:
                                field += "gantry(" + val["gantry"] + ") "
                        if "target" in val:
                                field += "target(" + val["target"] + ") "
                        if "contourext" in val:
                                field += "contourext(" + val["contourext"]+ ") "
                        if "doseext" in val:
                                field += "doseext(" + val["doseext"] + ") "
                        if "zstep" in val:
                                field += "zstep(%.d)"%(val["zstep"])
                        output.append(field)

                plan = "plan / dose(%.d)"%(self.dose)
                output.append(plan)


