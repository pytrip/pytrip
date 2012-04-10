import wx
 
from wx.lib.pubsub import Publisher as pub
from wx.xrc import XmlResource, XRCCTRL, XRCID
import os, threading,sys,struct,numpy
from dicompyler import guiutil, util
import pytrip


def pluginProperties():
    props = {}   
    props['name'] = 'APTG file export'                                                                                                 
    props['menuname'] = "as TRiP format"   
    props['description'] = "TRiP file format"                                                                  
    props['author'] = 'Jakob Toftegaard'                        
    props['version'] = 0.1                                                                                                  
    props['plugin_type'] = 'export'                                                                        
    props['plugin_version'] = 1                                                                                                    
    props['min_dicom'] = []                                                                                                       
    props['recommended_dicom'] = ['images', 'rtss', 'rtplan', 'rtdose']  
    return props  

class plugin:
	def __init__(self,parent):
		self.parent = parent
		pub.subscribe(self.on_update_patient,"patient.updated.raw_data")    
		datapath = guiutil.get_data_dir()
		userpath = os.path.join(datapath, 'plugins/TripExport/aptg_export.xrc');
		self.res = XmlResource(userpath)
	def on_update_patient(self,msg):
		self.data = msg.data;
	def pluginMenu(self,evt):
		data = self.data
		dlgAptgDialog = self.res.LoadDialog(self.parent,"AptgExportDialog")
		dlgAptgDialog.init()
		if dlgAptgDialog.ShowModal() == wx.ID_OK:
			fullpath = os.path.join(dlgAptgDialog.path,dlgAptgDialog.filename)
			dlgProgress = guiutil.get_progress_dialog(wx.GetApp().GetTopWindow(),"Creating Trip files ...")
			self.t = threading.Thread(target=self.CreateOutputThread,args=(dlgAptgDialog,data,fullpath,dlgProgress.OnUpdateProgress))
			self.t.start()
			dlgProgress.ShowModal()
			dlgProgress.Destroy()
		else:
			pass
		dlgAptgDialog.Destroy()
	def CreateOutputThread(self,dialog,data,fullpath,progressFunc):
		length = 7
		if dialog.set_ctx:
			ctx = pytrip.ctx2.CtxCube()
			wx.CallAfter(progressFunc, 1, length,"Convert dicom to ctx")
			try:
				ctx.read_dicom(data)
				wx.CallAfter(progressFunc, 2, length,"Write header file")
				ctx.write_trip_header(fullpath + ".hed")
				wx.CallAfter(progressFunc, 3, length,"Write Ctx data")
				ctx.write_trip_data(fullpath + ".ctx")
			except:
				print "No CT data"
		if dialog.set_dos:
			dos = pytrip.dos2.DosCube()
			try:
				dos.read_dicom(data)
				wx.CallAfter(progressFunc, 4, length,"Write DOS data")
				ctx.write_trip_data(fullpath + ".dos")
			except:
				print "No Dose data"
		if dialog.set_vdx:
			vdx = pytrip.vdx2.VdxCube("")
			wx.CallAfter(progressFunc, 5, length,"Convert dicom to vdx")
			try:
				vdx.read_dicom(data)
				wx.CallAfter(progressFunc, 6, length,"Write Vdx data")
				vdx.write_to_trip(fullpath + ".vdx")
			except:
				print "No Contour data"
			wx.CallAfter(progressFunc, 7, length,"Done")


class AptgExportDialog(wx.Dialog):
	def __init__(self):
		pre = wx.PreDialog()
		self.PostCreate(pre)
	def init(self):
		self.txtDICOMFolder = XRCCTRL(self,'txtDICOMFolder')
		self.btnFolderBrowse = XRCCTRL(self,'btnFolderBrowse')
		self.txtName = XRCCTRL(self,'txtName');
		self.check_dos = XRCCTRL(self,'checkDOS');
		self.check_ctx = XRCCTRL(self,'checkCTX');
		self.check_vdx = XRCCTRL(self,'checkVDX');

		
		wx.EVT_BUTTON(self,XRCID('btnFolderBrowse'),self.on_folder_browse)
		wx.EVT_BUTTON(self,wx.ID_OK,self.on_generate)
		
		pub.subscribe(self.on_import_prefs_change, 'general.dicom.import_location')	
		pub.sendMessage('preferences.requested.value', 'general.dicom.import_location')
 
	def on_folder_browse(self,evt):
		dlg = wx.DirDialog(
			self, defaultPath=self.path,
			message="Choose a folder to save the CTX files")
		if dlg.ShowModal() == wx.ID_OK:
			self.path = dlg.GetPath()
			self.txtOutputFolder = dlg.GetPath()
		dlg.Destroy()
	def on_import_prefs_change(self,msg):
		self.path = unicode(msg.data)
		self.txtDICOMFolder.SetValue(self.path)
	def on_generate(self,evt):
		self.filename = self.txtName.GetValue();
		self.set_vdx = self.check_vdx.IsChecked()
		self.set_dos = self.check_dos.IsChecked()
		self.set_ctx = self.check_ctx.IsChecked()

		self.EndModal(wx.ID_OK)	

