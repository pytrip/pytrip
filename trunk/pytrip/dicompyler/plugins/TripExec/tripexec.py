#!/usr/bin/env python

import wx, wx.grid
import pytrip
import tarfile
import paramiko
import shutil
from wx.xrc import XmlResource, XRCCTRL, XRCID
from wx.lib.pubsub import Publisher as pub
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
	self.ini_general()
	self.ini_output()
	self.ini_opt()
	self.txt_log = XRCCTRL(self,'txt_log')
	self.structures = {}
	self.path = "/home/jato/dicomtemp/"
	self.plan_name = "temp"
        self.btn_run = XRCCTRL(self,"btn_run")
        wx.EVT_BUTTON(self,XRCID('btn_run'),self.run_trip)		
	pub.subscribe(self.on_update_patient,"patient.updated.raw_data")
        pub.subscribe(self.on_structure_check, 'structures.checked')
    def on_update_patient(self,msg):
	self.data = msg.data
    def get_field_data(self):
	data = self.grid_beams.GetTable()
	out = []
	for y in range(data.GetNumberRows()):
		line = {}
		for x in range(data.GetNumberCols()):
			val = data.GetValue(y,x)
			if val != "":
				line[data.GetColLabelValue(x).lower()] = val
		if len(line) == 0:
			break
		out.append(line)
	return out
    def get_dose_data(self):
	data = self.grid_dose.GetTable()
	out = []
	for y in range(data.GetNumberRows()):
		line = {}
		for x in range(data.GetNumberCols()):
			val = data.GetValue(y,x)
			line[data.GetColLabelValue(x).lower()] = val
		out.append(line)
	return out
    def write_to_log(self,txt):
	self.txt_log.AppendText(txt)
	print txt
    def run_trip(self,evt):
	self.filepath = self.path + self.plan_name
	if os.path.exists(self.path):
		shutil.rmtree(self.path)
	os.makedirs(self.path)
	server = self.txt_server.GetValue()
	username = self.txt_username.GetValue()
	password = self.txt_password.GetValue()
        generator = TripFileGenerator()
	#setup beams
	fields = self.get_field_data()
	dose_info = self.get_dose_data()
	oar_list = []
	for dos in dose_info:
		if dos["oar"] == '1':
			oar_list.append({"voi_name":dos["structure name"],"dose":dos["max dose fraction"]})

	generator.beams = fields
	generator.oar = oar_list
	generator.dose = float(self.txt_dose.GetValue())
	generator.out_dose = self.check_out_dose.GetValue()
	generator.out_let = self.check_out_let.GetValue()
	generator.bioalg = self.drop_bioalg.GetStringSelection()
	generator.phys_bio = self.drop_phys_bio.GetStringSelection()
	generator.dosealg = self.drop_dosealg.GetStringSelection()
	generator.optalg = self.drop_optalg.GetStringSelection()
	generator.iterations = self.txt_iterations.GetValue()
	generator.eps = self.txt_eps.GetValue()
	generator.geps = self.txt_geps.GetValue()
	generator.ion = self.drop_projectile.GetStringSelection()

	princip = self.drop_opt_princip.GetStringSelection()
	if princip == 'H2OBased':
		generator.h2obased = True
		generator.ctbased = False
	elif princip == 'CTBased':
		generator.h2obased = False
		generator.ctbased = True
	self.write_to_log("Creating TRiP input file\n")
	generator.create_input_file(self.path + "plan.exec")
	ctx = pytrip.ctx2.CtxCube()
	ctx.read_dicom(self.data)
	ctx.patient_name = self.plan_name
	self.write_to_log("Writing header file\n")
	ctx.write_trip_header(self.filepath + ".hed")
	self.write_to_log("Writing ctx file\n")
	ctx.write_trip_data(self.filepath + ".ctx")
	vdx = pytrip.vdx2.VdxCube("",ctx)
	self.write_to_log("Writing vdx file\n")
	vdx.read_dicom(self.data,self.structures.keys())
	for dos in dose_info:
		voi = vdx.get_voi_by_name(dos["structure name"])
		if dos["target"] == '1':
			voi.type = '1'
		else:
			voi.type = '0'
	vdx.write_to_trip(self.filepath + ".vdx")

	self.write_to_log("Compress Files\n")
	tar = tarfile.open("/home/jato/temp.tar.gz","w:gz")
	tar.add(self.path,arcname='dicomtemp')
	tar.close()
	transport = paramiko.Transport((server,22))
	transport.connect(username=username,password=password)

	sftp = paramiko.SFTPClient.from_transport(transport)
	
	self.write_to_log("Copy files to cluster\n")
	sftp.put('/home/jato/temp.tar.gz','temp.tar.gz')
	sftp.close()
	transport.close()
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(server,username=username,password=password)
	self.write_to_log("Run Trip\n")
	stdin, stdout, stderr = ssh.exec_command("bash run_trip")
	for line in stdout:	
		self.write_to_log(line)
	ssh.close()

	transport = paramiko.Transport((server,22))
	transport.connect(username=username,password=password)

	sftp = paramiko.SFTPClient.from_transport(transport)
	self.write_to_log("Copy from cluster\n")	
	sftp.get('temp.tar.gz','/home/jato/temp.tar.gz')
	sftp.close()
	transport.close()
	output_folder = self.path
	if os.path.exists(output_folder):
		shutil.rmtree(output_folder)
	tar = tarfile.open("/home/jato/temp.tar.gz","r:gz")
	self.write_to_log("Extract output files\n")
	tar.extractall("/home/jato/")
	self.write_to_log('Done')
	patient = {}
	c = pytrip.ctx2.CtxCube()
	c.read_trip_data_file(self.filepath + ".ctx")
	patient["images"] = c.create_dicom()
	d = pytrip.dos2.DosCube()
	d.read_trip_data_file(self.filepath + ".phys.dos")
        d.target_dose = generator.dose
	patient["rtdose"] = d.create_dicom()
        patient["rxdose"] = float(d.target_dose)
        patient["rtplan"] = d.create_dicom_plan()
	v = pytrip.vdx2.VdxCube("")
	v.import_vdx(self.filepath + ".vdx")
	patient["rtss"] = v.create_dicom()
	pub.sendMessage('patient.updated.raw_data', patient)

    def ini_output(self):
	self.check_out_dose = XRCCTRL(self,"check_out_dose")
	self.check_out_let = XRCCTRL(self,"check_out_let")

    def ini_opt(self):
	self.txt_iterations = XRCCTRL(self,"txt_iterations")
	self.txt_eps = XRCCTRL(self,'txt_eps')
	self.txt_geps = XRCCTRL(self,'txt_geps')
	self.drop_phys_bio = XRCCTRL(self,"drop_opt_method")
	self.drop_opt_princip = XRCCTRL(self,"drop_opt_principle")
	self.drop_dosealg = XRCCTRL(self,"drop_dosealg")
	self.drop_bioalg = XRCCTRL(self,"drop_opt_bioalg")
	self.drop_optalg = XRCCTRL(self,"drop_optalg")
    def ini_general(self):
	self.txt_password = XRCCTRL(self,"txt_password")    
	self.txt_server = XRCCTRL(self,"txt_server")    
	self.txt_username = XRCCTRL(self,"txt_username")
    def ini_dose_grid(self):
        self.grid_dose = XRCCTRL(self,"grid_dose")
	self.txt_dose = XRCCTRL(self,"txt_dose")
        self.dose_data = {}
	self.grid_dose.CreateGrid( 0, 4 )
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
	self.grid_dose.SetColLabelValue( 3, u"Max Dose Fraction")

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
	self.structures = msg.data
        self.add_rows_grid_dose(msg.data)
    def ini_beams_grid(self):
	self.drop_projectile = XRCCTRL(self,"drop_projectile")
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
		self.oar = []
                self.bio = False
                self.ion = "Carbon"
                self.plan_name = "temp"
                self.dose = 68.0
                self.iterations = 500
		self.phys_bio = "phys"
                self.eps = 1e-3
                self.geps = 1e-4
                self.bioalg = "ld"
                self.optalg = "cg"
                self.dosealg = "ap"
                self.out_dose = True
                self.out_let = True
                self.h2obased = True
                self.ctbased = False
        def create_input_file(self,path):
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
			self.proj = 'C'
                elif self.ion == "Hydrogen":
                        output.append('sis "$TRIP98/DATA/SIS/1H.sis" / read')
                        output.append('ddd "$TRIP98/DATA/DDD/1H/RF3MM/1H*" / read')
                        output.append('spc "$TRIP98/DATA/SPC/1H/RF3MM/1H*" / read')
			self.proj = 'H'
                output.append("ct " +self.plan_name + " /read")
                targets = ""
                output.append("voi " + self.plan_name + "  /read")
		
                for i,val in enumerate(self.beams):
                        field = "field " + str(i+1) + " / new "
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
			field += 'proj(' + self.proj + ')'
                        output.append(field)
		for oar in self.oar:
			output.append("voi " + oar["voi_name"] + " / maxdosefraction(" + oar["dose"] + ") oarset")
                plan = "plan / dose(%.d)"%(self.dose)
                output.append(plan)
                opt = "opt / field(*) "

                if self.h2obased is True:
                        opt += "H2Obased "
		opt += "iterations(" + self.iterations +  ") " 
                opt += "dosealg(" + self.dosealg + ") "
                opt += "" + self.phys_bio.lower() + " "
                opt += "geps(" + str(self.geps) + ") "
                opt += "eps(" + str(self.eps) + ") "
		opt += "optalg(" + self.optalg + ") "

                output.append(opt)
                if self.out_dose is True:
                        output.append('dose "' + self.plan_name + '." /calculate field(*) write')
                if self.out_let is True:
                        output.append('dose "' + self.plan_name + '." /calculate dosemeanlet write')

                out = "\n".join(output) + "\n"
                f = open(path,"w+")
                f.write(out)
                f.close()
