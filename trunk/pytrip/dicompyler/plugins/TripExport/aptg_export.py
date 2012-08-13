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
		pub.subscribe(self.on_structure_change,"structures.checked")

		datapath = guiutil.get_data_dir()
		userpath = os.path.join(datapath, 'plugins/TripExport/aptg_export.xrc');
		self.structures = {}

		self.res = XmlResource(userpath)
	def on_structure_change(self,msg):
		self.structures = msg.data;

	def on_update_patient(self,msg):
		self.data = msg.data
		
	def pluginMenu(self,evt):
		dlgAptgDialog = self.res.LoadDialog(self.parent,"AptgExportDialog")
		dlgAptgDialog.init(self.structures)
		if dlgAptgDialog.ShowModal() == wx.ID_OK:
			fullpath = os.path.join(dlgAptgDialog.path,dlgAptgDialog.filename)
			dlgProgress = guiutil.get_progress_dialog(wx.GetApp().GetTopWindow(),"Creating Trip files ...")
			self.t = threading.Thread(target=self.CreateOutputThread,args=(dlgAptgDialog,self.data,self.structures,dlgAptgDialog.filename,fullpath,dlgProgress.OnUpdateProgress))
			self.t.start()
			dlgProgress.ShowModal()
			dlgProgress.Destroy()
		else:
			pass
		dlgAptgDialog.Destroy()
	def CreateOutputThread(self,dialog,data,structures,filename,fullpath,progressFunc):
		length = 8
		ctx = pytrip.ctx2.CtxCube()
		wx.CallAfter(progressFunc, 1, length,"Convert dicom to ctx")
		ctx.read_dicom(data)
		ctx.patient_name = filename

		if dialog.set_ctx:
		
			try:
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
			vdx = pytrip.vdx2.VdxCube("",ctx)
			wx.CallAfter(progressFunc, 5, length,"Convert dicom to vdx")
			
			vdx.read_dicom(data,structures.keys())
			wx.CallAfter(progressFunc, 6, length,"Write Vdx data")
			vdx.write_to_trip(fullpath + ".vdx")
			wx.CallAfter(progressFunc, 6, length,"Write header file")

		if dialog.create_dose_cube:
			vois = dialog.preset_dosecubes
			if len(vois) > 0:
				cubes = []
				maxdose = vois[max(vois.iterkeys(), key=lambda k: vois[k])]
				print maxdose
				for voi,dose in vois.items():
					cube = pytrip.dos2.DosCube(ctx)
					cube.load_from_structure(vdx.get_voi_by_name(voi),dose/maxdose*1000)
					cubes.append(cube)
				for i in range(1,len(cubes)):
					cubes[0].merge(cubes[i])
				cubes[0].write_trip_data(fullpath + ".target_dose.dos")
		ctx.write_trip_header(fullpath + ".hed")
		wx.CallAfter(progressFunc, 7, length,"Done")


class AptgExportDialog(wx.Dialog):
	def __init__(self):
		pre = wx.PreDialog()
		self.PostCreate(pre)
	def init(self,data):
                self.data = data
                self.init_general_tab()
		self.init_dosecube_tab()
		
		pub.subscribe(self.on_import_prefs_change, 'general.dicom.import_location')	
		pub.sendMessage('preferences.requested.value', 'general.dicom.import_location')

        def init_general_tab(self):
                self.txtDICOMFolder = XRCCTRL(self,'txtDICOMFolder')
		self.btnFolderBrowse = XRCCTRL(self,'btnFolderBrowse')
		self.txtName = XRCCTRL(self,'txtName');
		self.check_dos = XRCCTRL(self,'checkDOS')
		self.check_ctx = XRCCTRL(self,'checkCTX')
		self.check_vdx = XRCCTRL(self,'checkVDX')
                wx.EVT_BUTTON(self,XRCID('btnFolderBrowse'),self.on_folder_browse)
		wx.EVT_BUTTON(self,wx.ID_OK,self.on_submit)


        def init_dosecube_tab(self):
                self.listStructures = XRCCTRL(self,'listStructures')
                self.txtTargetDose = XRCCTRL(self,'txtTargetDose')
		self.btnSetDose = XRCCTRL(self,'btnSetDose')
                self.panel_target_dose = XRCCTRL(self,'panel_target_dose')
                self.check_generate_dosecube = XRCCTRL(self,'check_generate_dosecube')
                self.panel_target_dose.Enable(0)

                wx.EVT_BUTTON(self,XRCID('btnSetDose'),self.on_dose_set)
        	wx.EVT_LISTBOX(self,XRCID('listStructures'),self.list_structures_item_selected)
                wx.EVT_CHECKBOX(self,XRCID('check_generate_dosecube'),self.show_targetdose)
                
                self.preset_dosecubes = {}

        	for item in self.data:
                	self.listStructures.Append(self.data[item]["name"])
        def show_targetdose(self,evt): 
                if self.check_generate_dosecube.IsChecked():
                        self.panel_target_dose.Enable(1)
                else:
                        self.panel_target_dose.Enable(0)
                     
        def on_dose_set(self,evt):
                key =  self.listStructures.GetString(self.listStructures.GetSelection())
                self.preset_dosecubes[key] = float(self.txtTargetDose.GetValue()) 
        def list_structures_item_selected(self,evt):
                self.txtTargetDose.SetValue("")
                if evt.GetString() in self.preset_dosecubes:
                        self.txtTargetDose.SetValue(str(self.preset_dosecubes[evt.GetString()]))
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
	def on_submit(self,evt):
		self.filename = self.txtName.GetValue();
		self.set_vdx = self.check_vdx.IsChecked()
		self.set_dos = self.check_dos.IsChecked()
		self.set_ctx = self.check_ctx.IsChecked()
		self.create_dose_cube = self.check_generate_dosecube.IsChecked()

		self.EndModal(wx.ID_OK)	

