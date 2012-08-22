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
		datapath = guiutil.get_data_dir()
		userpath = os.path.join(datapath, 'plugins/TripImport/tripimport.xrc')
		self.res = XmlResource(userpath)

	def OnImportPrefsChange(self, msg):
		if (msg.topic[2] == 'import_location'):
			self.path = unicode(msg.data)
		elif (msg.topic[2] == 'import_location_setting'):
			self.import_location_setting = msg.data
        """When the import preferences change, update the values."""
 
	def on_update_patient(self,msg):
		self.data = msg.data;
	def pluginMenu(self,evt):
                dlgAptgDialog = self.res.LoadDialog(self.parent,"AptgImportDialog")
		dlgAptgDialog.init()
		if dlgAptgDialog.ShowModal() == wx.ID_OK:
			self.without_progress(dlgAptgDialog)
			"""dlgProgress = guiutil.get_progress_dialog(wx.GetApp().GetTopWindow(),"Importing TRiP Files ...")
			self.t = threading.Thread(target=self.CreateOutputThread,args=(dlgAptgDialog,dlgAptgDialog,dlgProgress.OnUpdateProgress))
			self.t.start()
			dlgProgress.ShowModal()
			dlgProgress.Destroy()"""

                       
		else:
			pass
		dlgAptgDialog.Destroy()
	def without_progress(self,dlgAptgDialog):
		patient = {}
		length = 3
                v = pytrip.vdx2.VdxCube("")
                if dlgAptgDialog.checkCTX.GetValue() is True:
			c = pytrip.ctx2.CtxCube()
			c.read_trip_data_file(dlgAptgDialog.cleanpath + ".ctx")
			patient["images"] = c.create_dicom()
                if dlgAptgDialog.checkDOS.GetValue() is True:
			d = pytrip.dos2.DosCube()
			d.read_trip_data_file(dlgAptgDialog.cleanpath + ".dos",dlgAptgDialog.check_multiply_dose.GetValue())
			dose = dlgAptgDialog.txtDose.GetValue()
                        d.target_dose = float(dose)
			patient["rtdose"] = d.create_dicom()
                        patient["rxdose"] = float(dose)
                        patient["rtplan"] = d.create_dicom_plan()

                if dlgAptgDialog.checkVDX.GetValue() is True:
			v.import_vdx(dlgAptgDialog.cleanpath + ".vdx")
			patient["rtss"] = v.create_dicom()
		if dlgAptgDialog.checkLET.GetValue() is True:
			l = pytrip.let.LETCube()
			l.read_trip_data_file(dlgAptgDialog.cleanpath + ".dosemlet.dos")
                        lvhs = {}
                        for voi in v.vois:
                                lvhs[voi.name] = l.calculate_lvh(voi)
			patient["LVHs"] = lvhs
			patient["letdata"] = l
		pub.sendMessage('patient.updated.raw_data', patient)
	def CreateOutputThread(self,dialog,dlgAptgDialog,progressFunc):
		 patient = {}
		 length = 3
                 if dlgAptgDialog.checkCTX.GetValue() is True:
			wx.CallAfter(progressFunc,1,length,"Import CTX")
			c = pytrip.ctx2.CtxCube()
			c.read_trip_data_file(dlgAptgDialog.cleanpath + ".ctx")
			patient["images"] = c.create_dicom()
                 if dlgAptgDialog.checkDOS.GetValue() is True:
			wx.CallAfter(progressFunc,2,length,"Import DOS")
			d = pytrip.dos2.DosCube()
			d.read_trip_data_file(dlgAptgDialog.cleanpath + ".dos",dlgAptgDialog.check_multiply_dose.GetValue())
			dose = dlgAptgDialog.txtDose.GetValue()
                        d.target_dose = float(dose)
			patient["rtdose"] = d.create_dicom()
                        patient["rxdose"] = float(dose)
                        patient["rtplan"] = d.create_dicom_plan()

                 if dlgAptgDialog.checkVDX.GetValue() is True:
			wx.CallAfter(progressFunc,3,length,"Import VDX")
			v = pytrip.vdx2.VdxCube("")
			v.import_vdx(dlgAptgDialog.cleanpath + ".vdx")
			patient["rtss"] = v.create_dicom()
		 pub.sendMessage('patient.updated.raw_data', patient)


class AptgImportDialog(wx.Dialog):
        def __init__(self):
                pre = wx.PreDialog()
                self.PostCreate(pre)
        def is_number(self,number):
                try:
                        float(number)
                        return True
                except ValueError:
                        return False

        def validate(self):
                valid = True
                error_message = ""
                if self.checkDOS.IsChecked() and not self.is_number(self.txtDose.GetValue()):
                        valid = False
                        error_message = "Dose should be a number"
                if not valid:
                        wx.MessageBox(error_message,'Wrong Input',wx.OK|wx.ICON_ERROR)
                return valid

        def init(self):
                self.txtImportFolder = XRCCTRL(self,'txtImportFolder')
                self.btnBrowse = XRCCTRL(self,'btnBrowse')
                self.btnImport = XRCCTRL(self,'btnImport')
                self.checkCTX = XRCCTRL(self,'checkCTX')
                self.checkDOS = XRCCTRL(self,'checkDOS')
                self.checkVDX = XRCCTRL(self,'checkVDX')
                self.checkLET = XRCCTRL(self,'checkLET')
                self.txtDose = XRCCTRL(self,'txtDose')
		self.check_multiply_dose = XRCCTRL(self,'check_multiply_dose')

                self.labelCTX = XRCCTRL(self,'labelCTX')
                self.labelDOS = XRCCTRL(self,'labelDOS')
                self.labelVDX = XRCCTRL(self,'labelVDX')
                self.labelLET = XRCCTRL(self,'labelLET')


                self.reset_dialog() 
                wx.EVT_BUTTON(self,XRCID('btnBrowse'),self.on_folder_browse)
                wx.EVT_BUTTON(self,XRCID('btnImport'),self.on_import)
                wx.EVT_CHECKBOX(self,XRCID('checkDOS'),self.on_checkdos_click)
		
                pub.subscribe(self.on_import_prefs_change, 'general.dicom')	
		pub.sendMessage('preferences.requested.values', 'general.dicom')
        def on_import(self,evt):

                if self.validate():
                        self.EndModal(wx.ID_OK)
        def on_checkdos_click(self,evt):
                if evt.IsChecked() is True:
                        self.txtDose.Enable(1)
                else:
                        self.txtDose.Enable(0)
        def reset_dialog(self):
                self.checkCTX.Enable(0)
                self.checkDOS.Enable(0)
                self.checkLET.Enable(0)
                self.checkVDX.Enable(0)
                self.labelCTX.SetLabel("")
                self.labelDOS.SetLabel("")
                self.labelLET.SetLabel("")
                self.labelVDX.SetLabel("")
                self.txtDose.Enable(0)

        def on_import_prefs_change(self,msg):
                print msg.data
                if (msg.topic[2] == 'import_location'):
                    self.path = unicode(msg.data)
                elif (msg.topic[2] == 'import_location_setting'):
                    self.import_location_setting = msg.data

        def on_folder_browse(self,evt):
                dlg = wx.FileDialog(self, defaultDir = self.path,wildcard="Trip Header file (*.hed)|*.hed",message="Choose a Header File")
                if dlg.ShowModal() == wx.ID_OK:
                        self.reset_dialog()
                        self.path = dlg.GetPath()
                        self.txtImportFolder.SetValue(self.path)
                        self.cleanpath = os.path.splitext(self.path)[0]
                        self.filename = os.path.basename(self.cleanpath)
                        if os.path.isfile(self.cleanpath + ".ctx"):
                                self.checkCTX.Enable(1)
                                self.checkCTX.SetValue(1)
                                self.labelCTX.SetLabel(self.filename + ".ctx")
                        if os.path.isfile(self.cleanpath + ".dos"):
                                self.checkDOS.Enable(1)
                                self.checkDOS.SetValue(1)
                                self.txtDose.Enable(1)
                                self.labelDOS.SetLabel(self.filename + ".dos")
                        if os.path.isfile(self.cleanpath + ".vdx"):
                                self.checkVDX.Enable(1)
                                self.checkVDX.SetValue(1)
                                self.labelVDX.SetLabel(self.filename + ".vdx")
                        if os.path.isfile(self.cleanpath + ".dosemlet.dos"):
                                self.checkLET.Enable(1)
                                self.checkLET.SetValue(1)
                                self.labelLET.SetLabel(self.filename + ".dosemlet.dos")
                        if (self.import_location_setting == "Remember Last Used"):
                                pub.sendMessage('preferences.updated.value',
                                                {'general.dicom.import_location':dlg.GetDirectory()})
                                pub.sendMessage('preferences.requested.values', 'general.dicom')
                dlg.Destroy()


