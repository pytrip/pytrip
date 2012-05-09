import wx
 
from wx.lib.pubsub import Publisher as pub
from wx.xrc import XmlResource, XRCCTRL, XRCID
import os, threading,sys,struct,numpy
from dicompyler import guiutil, util
import pytrip


def pluginProperties():
    props = {}   
    props['name'] = 'APTG file import'                                                                                                 
    props['menuname'] = "as TRiP format"   
    props['description'] = "TRiP file format"                                                                  
    props['author'] = 'Jakob Toftegaard'                        
    props['version'] = 0.1                                                                                                  
    props['plugin_type'] = 'import'                                                                        
    props['plugin_version'] = 1                                                                                                    
    props['min_dicom'] = []                                                                                                       
    props['recommended_dicom'] = ['images', 'rtss', 'rtplan', 'rtdose']  
    return props  

class plugin:
	def __init__(self,parent):
		self.parent = parent
		pub.subscribe(self.OnImportPrefsChange, 'general.dicom')
		pub.sendMessage('preferences.requested.values', 'general.dicom')

	def OnImportPrefsChange(self, msg):
		if (msg.topic[2] == 'import_location'):
			self.path = unicode(msg.data)
		elif (msg.topic[2] == 'import_location_setting'):
			self.import_location_setting = msg.data
        """When the import preferences change, update the values."""

 
	def on_update_patient(self,msg):
		self.data = msg.data;
	def pluginMenu(self,evt):
		dlg = wx.FileDialog(self.parent, defaultDir = self.path,wildcard="Trip Header file (*.hed)|*.hed",message="Choose a Header File")
		patient = {}
		if dlg.ShowModal() == wx.ID_OK:
			hed_path = dlg.GetPath()
			path = os.path.splitext(hed_path)[0]
			if os.path.isfile(path + ".ctx"):
				c = pytrip.ctx2.CtxCube()
				c.read_trip_data_file(path + ".ctx")
				patient["images"] = c.create_dicom()
			if os.path.isfile(path + ".dos"):
				d = pytrip.dos2.DosCube()
				d.read_trip_data_file(path + ".dos")
				patient["rtdose"] = d.create_dicom()
				patient["rtplan"] = d.create_dicom_plan()
				patient["rxdose"] = 100.0
			if os.path.isfile(path + ".vdx"):
				v = pytrip.vdx2.VdxCube("")
				v.import_vdx(path + ".vdx")
				patient["rtss"] = v.create_dicom();
			pub.sendMessage('patient.updated.raw_data', patient)
		dlg.Destroy()
