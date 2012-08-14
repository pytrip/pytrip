import wx
 
from wx.lib.pubsub import Publisher as pub
from wx.xrc import XmlResource, XRCCTRL, XRCID
import os, threading,sys,struct,numpy
from dicompyler import guiutil, util
import pytrip


def pluginProperties():
	props = {}   
	props['name'] = 'TRiP Exec'                                                                                                    
	props['description'] = "TRiP file format"                                                               
	props['author'] = 'Jakob Toftegaard'
	props['version'] = 0.1                                                                                                  
	props['plugin_type'] = 'main'                                                                        
	props['plugin_version'] = 1                                                                                                    
	props['min_dicom'] = []                                                                                                       
	props['recommended_dicom'] = ['images', 'rtss', 'rtplan', 'rtdose']
	return props

def pluginLoader(parent):
	datapath = guiutil.get_data_dir()
	userpath = os.path.join(datapath, 'plugins/TripExec/tripexec.xrc')
    	res = XmlResource(util.GetBasePluginsPath(userpath))

    	panel = res.LoadPanel(parent, 'AptgTripDialog')
    	panel.Init(res)
    	return panel

class AptgTripDialog(wx.Panel):
	def __init__(self):
		pre = wx.PrePanel()
		self.PostCreate(pre)
	def Init(self,res):
		pub.subscribe(self.on_import_prefs_change, 'general.dicom.import_location')	
		pub.sendMessage('preferences.requested.value', 'general.dicom.import_location')
	def on_import_prefs_change(self,msg):
		self.path = unicode(msg.data)
	

