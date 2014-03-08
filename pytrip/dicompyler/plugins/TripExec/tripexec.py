#!/usr/bin/env python

import wx, wx.grid
import pytrip
import res_text
import tarfile
import paramiko
import shutil
import threading
import numpy
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib
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
    props['recommended_dicom'] = ['images', 'rtss']

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

    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
    
    def Init(self, res):
        self.ini_rbe()
        self.fields = fieldClass(self)
        self.trip_listbook = XRCCTRL(self,"trip_listbook")
        
        self.tab_general = res.LoadPanel(self.trip_listbook, 'panel_general')
        self.trip_listbook.AddPage(self.tab_general,"General")
        self.tab_general.Init(res,self)
        
        self.tab_voi = res.LoadPanel(self.trip_listbook, 'panel_voi')
        self.trip_listbook.AddPage(self.tab_voi,"Vois")
        self.tab_voi.Init(res,self)
        
        self.tab_field = res.LoadPanel(self.trip_listbook, 'panel_field')
        self.trip_listbook.AddPage(self.tab_field,"Field")
        self.tab_field.Init(res,self,self.fields)
        
        self.tab_field2 = res.LoadPanel(self.trip_listbook, 'panel_field2')
        self.trip_listbook.AddPage(self.tab_field2,"Field 2")
        self.tab_field2.Init(res,self,self.fields)
        
        self.tab_opt = res.LoadPanel(self.trip_listbook, 'panel_opt')
        self.trip_listbook.AddPage(self.tab_opt,"Optimization")
        self.tab_opt.Init(res,self)
        
        self.tab_output = res.LoadPanel(self.trip_listbook, 'panel_output')
        self.trip_listbook.AddPage(self.tab_output,"Output")
        self.tab_output.Init(res,self)
        
        self.tab_run = res.LoadPanel(self.trip_listbook, 'panel_run')
        self.trip_listbook.AddPage(self.tab_run,"Run")

        self.txt_log = XRCCTRL(self,'txt_log')

        self.plan_name = "temp"
        self.btn_run = XRCCTRL(self,"btn_run")

        self.preferences = [{'Settings':[{'name':'Working Directory',
                'type':'directory',
                'callback':'tripexec.settings.working_directory',
                'default':'~/'}]}]
        wx.EVT_BUTTON(self,XRCID('btn_run'),self.run_trip)
        pub.subscribe(self.on_update_patient,"patient.updated.raw_data")

        pub.subscribe(self.on_settings_change, 'tripexec.settings')
        pub.sendMessage('preferences.requested.values', 'tripexec.settings')
    def ini_rbe(self):
        self.rbe_files = res_text.rbe_files
    
    def on_settings_change(self,msg):
        if (msg.topic[2] == 'working_directory'):
            self.path = msg.data + "/dicomtemp/"
            
    def on_update_patient(self,msg):
        self.data = msg.data
        if not hasattr(self,"paths_cube"):
            ct_data = pytrip.ctx2.CtxCube()
            ct_data.read_dicom(self.data)
            self.paths_cube = pytrip.paths.DensityCube(ct_data)
        if not hasattr(self,"vdx"):
            self.vdx = pytrip.vdx2.VdxCube("",self.paths_cube)
            self.vdx.read_dicom(self.data)
        
    def write_to_log(self,txt):
        self.txt_log.AppendText(txt)
        
    def run_trip(self,evt):
        self.t = threading.Thread(target=self.run_trip_thread)
        self.t.start()
    def run_trip_thread(self):
        executer = TripFieldExecuter(self)
        executer.execute()
class fieldClass(wx.grid.PyGridTableBase):
    def __init__(self,parent):
        wx.grid.PyGridTableBase.__init__(self)
        self.labels = ["Target","fwhm","Gantry","Couch","Rastersteps","zsteps","doseext","contourext"]
        self.observers = []
        self.parent = parent
        self.fields = []
        self.default_field = {"gantry":0.0,"couch":90.0}
    def SetValue(self, row, col, valstr):
        self.fields[row][self.labels[col].lower()] = valstr
        return True
    def IsEmptyCell(self, row, col):
        if row > len(self.fields) or self.labels[col].lower() not in self.fields[row].keys():
            return True
        else:
            return False
    #pygridtablebase functions
    def GetNumberRows(self):
        return self.get_number_of_fields()
    def GetNumberCols(self):
        return 8
    def GetValue(self,row,col):
        field = self.fields[row]
        key = self.labels[col].lower()
        if key in field.keys():
            return field[key]
        return ""
    def GetColLabelValue(self, col):
        return self.labels[col]
        
    
    def add_new_field(self,field):
        if not "target" in field.keys():
            target_name = self.parent.tab_voi.get_target_name()
            if target_name != "":
                center = self.parent.vdx.get_voi_by_name(target_name).calculate_center()
                field["target"] = "%.2f,%.2f,%.2f"%(center[0],center[1],center[2])
        self.fields.append(field)
        self.updated()
        msg = wx.grid.GridTableMessage(self, wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED,1)
        self.GetView().ProcessTableMessage(msg)
    def remove_field(self,number):
        self.fields.pop(number-1)
        msg = wx.grid.GridTableMessage(self, wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED,self.GetNumberRows(),1)
        self.GetView().ProcessTableMessage(msg)
        if self.get_number_of_fields() is 0:
            self.add_new_field(self.default_field.copy())
        self.updated()
    def get_field(self,field):
        return self.fields[field-1]
    def get_fields(self):
        return self.fields
    def get_number_of_fields(self):
        return len(self.fields)
    def add_observer(self,observer):
        if not observer in self.observers:
            self.observers.append(observer)
    def remove_observer(self,observer):
        try:
            self.observers.remove(observer)
        except ValueError:
            pass
    def updated(self):
        for observer in self.observers:
            observer.notify()

class outputPanel(wx.Panel):
    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
    def Init(self,res,parent):
        self.check_out_dose = XRCCTRL(self,"check_out_dose")
        self.check_out_dose.SetToolTip(wx.ToolTip(res_text.tooltip["check_out_dose"]))
        self.check_out_let = XRCCTRL(self,"check_out_let")
        self.check_out_let.SetToolTip(wx.ToolTip(res_text.tooltip["check_out_let"]))
    def get_output_data(self):
        output = {}
        output["dose"] = self.check_out_dose.GetValue()
        output["let"] = self.check_out_let.GetValue()
        return output
class optPanel(wx.Panel):
    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
    def Init(self,res,parent):
        self.txt_iterations = XRCCTRL(self,"txt_iterations")
        self.txt_iterations.SetToolTip(wx.ToolTip(res_text.tooltip["txt_iterations"]))
        self.txt_eps = XRCCTRL(self,'txt_eps')
        self.txt_eps.SetToolTip(wx.ToolTip(res_text.tooltip["txt_eps"]))
        self.txt_geps = XRCCTRL(self,'txt_geps')
        self.txt_geps.SetToolTip(wx.ToolTip(res_text.tooltip["txt_geps"]))
        self.drop_phys_bio = XRCCTRL(self,"drop_opt_method")
        self.drop_phys_bio.SetToolTip(wx.ToolTip(res_text.tooltip["drop_opt_method"]))
        self.drop_opt_princip = XRCCTRL(self,"drop_opt_principle")
        self.drop_opt_princip.SetToolTip(wx.ToolTip(res_text.tooltip["drop_opt_principle"]))

        self.drop_dosealg = XRCCTRL(self,"drop_dosealg")
        self.drop_dosealg.SetToolTip(wx.ToolTip(res_text.tooltip["drop_dosealg"]))
        self.drop_bioalg = XRCCTRL(self,"drop_opt_bioalg")
        self.drop_bioalg.SetToolTip(wx.ToolTip(res_text.tooltip["drop_opt_bioalg"]))
        self.drop_optalg = XRCCTRL(self,"drop_optalg")
        self.drop_optalg.SetToolTip(wx.ToolTip(res_text.tooltip["drop_optalg"]))
    def get_opt_data(self):
        opt_data = {}
        opt_data["iterations"] = self.txt_iterations.GetValue()
        opt_data["eps"] = self.txt_eps.GetValue()
        opt_data["geps"] = self.txt_geps.GetValue()
        opt_data["phys_bio"] = self.drop_phys_bio.GetStringSelection()
        opt_data["opt_princip"] = self.drop_opt_princip.GetStringSelection()
        opt_data["dosealg"] = self.drop_dosealg.GetStringSelection()
        opt_data["bioalg"] = self.drop_bioalg.GetStringSelection()
        opt_data["optalg"] = self.drop_optalg.GetStringSelection()
        return opt_data
    
class fieldPanel(wx.Panel):
    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
    def notify(self):
        self.update = True
    def OnPaint(self,evt):
        self.grid_beams.AutoSize()
        pass
    def Init(self,res,parent,fielddata):
        fielddata.add_observer(self)
        self.fielddata = fielddata
        self.update = True
        self.drop_projectile = XRCCTRL(self,"drop_projectile")
        self.grid_beams = XRCCTRL(self,"grid_beams")
        self.grid_beams.SetTable(self.fielddata)
        self.grid_beams.GetGridColLabelWindow().Bind(wx.EVT_MOTION,self.on_mouse_over_col_label_grid_beams)
        self.grid_beams.SetWindowStyle(wx.HSCROLL)
        self.grid_beams.SetSize((400,300))
        self.grid_beams.AutoSize()
        wx.grid.EVT_GRID_CMD_EDITOR_HIDDEN(self,XRCID('grid_beams'),self.on_change_beams_grid)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    
    def on_change_beams_grid(self,evt):
        return

    def on_mouse_over_col_label_grid_beams(self,event):
        col_tooltip = res_text.tooltip["grid_beam"]
        x = event.GetX()
        col = self.grid_beams.XToCol(x)
        self.grid_beams.GetGridColLabelWindow().SetToolTip(wx.ToolTip(col_tooltip[col]))
class field2Panel(wx.Panel):
    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
    def notify(self):
        n_fields = self.fielddata.get_number_of_fields()
        if n_fields > self.drop_fields.GetCount()-1:
            self.drop_fields.Insert("Field " + str(n_fields),n_fields-1)
            self.drop_fields.SetSelection(n_fields-1)
        if n_fields < self.drop_fields.GetCount()-1:
            self.drop_fields.Delete(self.drop_fields.GetCount()-2)
            self.drop_fields.SetSelection(self.drop_fields.GetSelection()-1)
        self.load_field(self.fielddata.get_field(self.drop_fields.GetSelection()+1))
                 
    def on_drop_field_changed(self,evt):
        obj = evt.GetEventObject()
        if obj.GetCount()-1 is obj.GetSelection():
            self.fielddata.add_new_field(self.fielddata.default_field.copy())
        else:
            field = self.fielddata.get_field(obj.GetSelection()+1)
            self.load_field(field)
    def load_field(self,field):
        self.spin_gantry.SetValue(str(field["gantry"]))
        self.spin_couch.SetValue(str(field["couch"]))
        self.slider_gantry.SetValue(int(field["gantry"]*10))
        self.slider_couch.SetValue(int(field["couch"]*10))
    def Init(self,res,parent,fielddata):
        fielddata.add_observer(self)
        self.parent = parent
        self.fielddata = fielddata
        self.couch_angle = 91
        self.gantry_angle = 0
        
        self.btn_delete_field = XRCCTRL(self,"btn_delete_field")
        self.btn_delete_field.Bind(wx.EVT_BUTTON,self.on_delete_field_clicked)
        
        self.drop_fields = XRCCTRL(self,'drop_field')
        self.drop_fields.Bind(wx.EVT_CHOICE,self.on_drop_field_changed)
        self.slider_gantry = XRCCTRL(self,'slider_gantry')
        self.slider_couch = XRCCTRL(self,'slider_couch')
        self.spin_gantry = XRCCTRL(self,'spin_gantry')
        self.spin_couch = XRCCTRL(self,'spin_couch')

        self.slider_gantry.SetValue(self.gantry_angle*10)
        self.slider_couch.SetValue(self.couch_angle*10)

        self.panel_plotfield = XRCCTRL(self,'panel_plotfield')
        self.spin_gantry_panel = XRCCTRL(self,'spin_gantry_panel')
                
        self.spin_couch = XRCCTRL(self,"spin_couch")
        self.spin_gantry = XRCCTRL(self,"spin_gantry")
        self.spin_couch.Bind(wx.EVT_LEAVE_WINDOW,self.validate_angle_textctrl)
        self.spin_gantry.Bind(wx.EVT_LEAVE_WINDOW,self.validate_angle_textctrl)
        
        self.spin_gantry.Bind(wx.EVT_KILL_FOCUS,self.on_spin_gantry_changed)
        self.spin_couch.Bind(wx.EVT_KILL_FOCUS,self.on_spin_couch_changed)
        
        self.spin_gantry.SetValue(str(self.gantry_angle))
        self.spin_couch.SetValue(str(self.couch_angle))
        self.fieldplot = "None"
        self.panel_plotfield = XRCCTRL(self,'panel_plotfield')
        self.drop_fieldplot = XRCCTRL(self,'drop_fieldplot')
        self.drop_fieldplot.Bind(wx.EVT_CHOICE,self.on_drop_fieldplot_changed)
        self.slider_gantry.Bind(wx.EVT_SLIDER,self.on_slider_gantry_changed)
        self.slider_couch.Bind(wx.EVT_SLIDER,self.on_slider_couch_changed)
        
        fielddata.add_new_field(self.fielddata.default_field.copy())
    def on_delete_field_clicked(self,evt):
        self.fielddata.remove_field(self.drop_fields.GetSelection()+1)
    def validate_angle_textctrl(self,evt):
        value = evt.GetEventObject().GetValue()
        try:
            value = float(value)
        except:
            evt.GetEventObject().SetValue("0")			

    def on_drop_fieldplot_changed(self,evt):
        self.fieldplot = evt.GetString()
        if self.fieldplot == "None":
            self.panel_plotfield.DestroyChildren()
            return
        elif self.fieldplot == "CT":
            return
        elif self.fieldplot == "Density":
            target_name = self.parent.tab_voi.get_target_name()
            if target_name == "":
                self.drop_fieldplot.SetSelection(0)
                return
            self.field_target_name = target_name
            self.figure = Figure( None, 100 )
            self.canvas = FigureCanvasWxAgg( self.panel_plotfield, -1, self.figure )
            self.setSize()
            self.figure.clf()
            self.subplot = self.figure.add_subplot(111)
            self.Bind(wx.EVT_SIZE, self._setSize)
            self.field_angle_changed_density()
    def _setSize(self,evt):
        self.setSize()
        evt.Skip()
    def get_active_field(self):
        return self.drop_fields.GetSelection()+1
    def field_angle_changed(self):
        field = self.fielddata.get_field(self.get_active_field())
        field["gantry"] = float(self.spin_gantry.GetValue())
        field["couch"] = float(self.spin_couch.GetValue())
        self.fielddata.updated()
        if self.fieldplot == "None":
            return
        elif self.fieldplot == "CT":
            return
        elif self.fieldplot == "Density":
             self.field_angle_changed_density()
    def setSize(self):
        size = self.panel_plotfield.GetClientSize()
        pixels = tuple( size )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )
    def plot_density(self,plotdata,contour):
        self.subplot.cla()
        self.subplot.imshow(plotdata,vmax=10.0)
        self.subplot.plot(contour[:,1]/self.parent.paths_cube.pixel_size,contour[:,0]/self.parent.paths_cube.pixel_size,'r')
        self.subplot.set_ylim([0,len(plotdata)])
        self.subplot.set_xlim([0,len(plotdata[0])])
        self.canvas.draw()
    def field_angle_changed_density(self):
        if hasattr(self,"density_thread"):
            if self.density_thread.is_alive():
                self.density_thread.join()
        voi = self.parent.vdx.get_voi_by_name(self.field_target_name)
        self.density_thread = threading.Thread(target=self.calculate_density_thread,args=(voi,self.gantry_angle,self.couch_angle,self.plot_density))
        self.density_thread.start()
    def calculate_density_thread(self,voi,gantry_angle,couch_angle,plotfunction):
        dense = pytrip.paths.DensityProjections(self.parent.paths_cube)
        dense_data,start,basis = dense.calculate_projection(voi,gantry_angle,couch_angle)
        gradient = numpy.gradient(dense_data)
        data = (gradient[0]**2+gradient[0]**2)**0.5
        contour = voi.get_2d_projection_on_basis(basis,start)
        wx.CallAfter(plotfunction,data,contour)
        
    def on_slider_gantry_changed(self,evt):
        value = float(evt.GetEventObject().GetValue())/10
        self.spin_gantry.SetValue(str(value))
        self.gantry_angle = value
        self.field_angle_changed()
    def on_slider_couch_changed(self,evt):
        value = float(evt.GetEventObject().GetValue())/10
        self.spin_couch.SetValue(str(value))
        self.couch_angle = value
        self.field_angle_changed()
    def on_spin_couch_changed(self,evt):
        value = evt.GetEventObject().GetValue()
        self.slider_couch.SetValue(float(value)*10)
        self.couch_angle = float(value)
        self.field_angle_changed()
    def on_spin_gantry_changed(self,evt):
        value = evt.GetEventObject().GetValue()
        self.slider_gantry.SetValue(float(value)*10)
        self.gantry_angle = float(value)
        self.field_angle_changed()

class voiPanel(wx.Panel):
    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
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
    def get_target_name(self):
        target_data = self.get_dose_data()
        target_name = ""
        for line in target_data:
            if line["target"] == '1':
                target_name = line["structure name"]
        return target_name
    def on_structure_check(self,msg):
        self.structures = msg.data
        self.add_rows_grid_dose(msg.data)
    def Init(self,res,parent):
        self.structures = {}
        self.parent = parent
        self.grid_dose = XRCCTRL(self,"grid_dose")
        self.txt_dose = XRCCTRL(self,"txt_dose")
        self.dose_data = {}
        self.grid_dose.CreateGrid( 0, 5 )
        self.grid_dose.Bind(wx.grid.EVT_GRID_CELL_CHANGE,self.on_cell_change_grid_dose)
        self.grid_dose.GetGridColLabelWindow().Bind(wx.EVT_MOTION,self.on_mouse_over_col_label_dose_grid)
        self.grid_dose.GetGridColLabelWindow().SetToolTipString("")
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

        attr = wx.grid.GridCellAttr()
        attr.SetReadOnly(True)
        self.grid_dose.SetColAttr(3,attr)

        attr = wx.grid.GridCellAttr()
        attr.SetEditor(wx.grid.GridCellChoiceEditor(self.parent.rbe_files.keys()))
        self.grid_dose.SetColAttr(4,attr)



        self.grid_dose.EnableEditing( True )
        self.grid_dose.EnableGridLines( True )
        self.grid_dose.EnableDragGridSize( True )
        self.grid_dose.SetMargins( 0, 0 )
        self.grid_dose.SetColSize(0,0)
        self.grid_dose.SetColLabelValue( 0, u"Structure Name" )
        self.grid_dose.SetColLabelValue( 1, u"Target" )
        self.grid_dose.SetColLabelValue( 2, u"OAR" )
        self.grid_dose.SetColLabelValue( 3, u"Max Dose Fraction")
        self.grid_dose.SetColLabelValue( 4, u"Tissue Type")

        self.grid_dose.SetSize((300,200))
        
        pub.subscribe(self.on_structure_check, 'structures.checked')
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
            for i in range(self.grid_dose.GetNumberRows()):
                name =  self.grid_dose.GetTable().GetValue(i,0)
                if name not in names:
                    del self.dose_data[name]
                    self.grid_dose.DeleteRows(i)
                else:
                    i += 1
        self.grid_dose.AutoSize()
        
    def on_mouse_over_col_label_dose_grid(self,event):
        col_tooltip = res_text.tooltip["grid_dose"]
        x = event.GetX()
        col = self.grid_dose.XToCol(x)
        self.grid_dose.GetGridColLabelWindow().SetToolTip(wx.ToolTip(col_tooltip[col]))
    def on_cell_change_grid_dose(self,event):
        col = event.Col
        row = event.Row
        if col is 1:
            value = self.grid_dose.GetTable().GetValue(row,col)
            if value == "1":
                attr = self.grid_dose.GetOrCreateCellAttr(row,2)
                attr.SetReadOnly(True)
                attr = self.grid_dose.GetOrCreateCellAttr(row,3)
                attr.SetReadOnly(True)
            else:
                attr = self.grid_dose.GetOrCreateCellAttr(row,2)
                attr.SetReadOnly(False)
        if col is 2:
            value = self.grid_dose.GetTable().GetValue(row,col)
            if value == "1":
                attr = self.grid_dose.GetOrCreateCellAttr(row,1)
                attr.SetReadOnly(True)
                attr = self.grid_dose.GetOrCreateCellAttr(row,3)
                attr.SetReadOnly(False)
            else:
                attr = self.grid_dose.GetOrCreateCellAttr(row,1)
                attr.SetReadOnly(False)
                attr = self.grid_dose.GetOrCreateCellAttr(row,3)
                attr.SetReadOnly(True)
        
class generalPanel(wx.Panel):
    def __init__(self):
        pre = wx.PrePanel()
        # the Create step is done by XRC.
        self.PostCreate(pre)
    def Init(self,res,parent):
        self._parent = parent
        self.txt_password = XRCCTRL(self,"txt_password")
        self.txt_password.SetToolTip(wx.ToolTip(res_text.tooltip["txt_password"]))
        self.txt_server = XRCCTRL(self,"txt_server")
        self.txt_server.SetToolTip(wx.ToolTip(res_text.tooltip["txt_server"]))
        self.txt_username = XRCCTRL(self,"txt_username")
        self.txt_username.SetToolTip(wx.ToolTip(res_text.tooltip["txt_username"]))
    def get_server_settings(self):
        settings = {}
        settings["server"] = self.txt_server.GetValue()
        settings["username"] = self.txt_username.GetValue()
        settings["password"] = self.txt_password.GetValue()
        return settings
class TripFieldExecuter:
    def __init__(self,parent):
        self.parent = parent
        self.path = parent.path
        self.plan_name = parent.plan_name
    def execute(self):
        self.prepare_folder()
        self.prepare_server()
        self.create_trip_exec_file()
        self.convert_files_to_voxelplan()
        self.compress_files()
        self.copy_files_to_server()
        self.run_ssh_command("bash run_trip")
        self.copy_back_from_server()
        self.decompress_data()
        self.visualize_data()
    def prepare_folder(self):
        self.filepath = self.path + self.plan_name
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)
    def prepare_server(self):
        self.server_settings = self.parent.tab_general.get_server_settings()
        
    def create_trip_exec_file(self):
        generator = TripFileGenerator()
        #setup beams
        fields = self.parent.fields.get_fields()
        self.dose_info = self.parent.tab_voi.get_dose_data()
        oar_list = []
        for dos in self.dose_info:
            if dos["oar"] == '1':
                oar_list.append({"voi_name":dos["structure name"],"dose":dos["max dose fraction"]})

        generator.beams = fields
        generator.oar = oar_list
        generator.dose = float(self.parent.tab_voi.txt_dose.GetValue())
        
        output_data = self.parent.tab_output.get_output_data()
        
        generator.out_dose = output_data["dose"]
        generator.out_let = output_data["let"]
        
        opt_data = self.parent.tab_opt.get_opt_data()
        
        generator.bioalg = opt_data["bioalg"]
        generator.phys_bio = opt_data["phys_bio"]
        generator.dosealg = opt_data["dosealg"]
        generator.optalg = opt_data["optalg"]
        generator.iterations = opt_data["iterations"]
        generator.eps = opt_data["eps"]
        generator.geps = opt_data["geps"]
        generator.ion = self.parent.tab_field.drop_projectile.GetStringSelection()

        princip = self.parent.tab_opt.drop_opt_princip.GetStringSelection()
        if princip == 'H2OBased':
            generator.h2obased = True
            generator.ctbased = False
        elif princip == 'CTBased':
            generator.h2obased = False
            generator.ctbased = True
        wx.CallAfter(self.parent.write_to_log,"Creating TRiP input file\n")
        generator.create_input_file(self.path + "plan.exec")
    def convert_files_to_voxelplan(self):
        ctx = pytrip.ctx2.CtxCube()
        ctx.read_dicom(self.parent.data)
        ctx.patient_name = self.plan_name
        wx.CallAfter(self.parent.write_to_log,"Writing header file\n")
        ctx.write_trip_header(self.filepath + ".hed")
        wx.CallAfter(self.parent.write_to_log,"Writing ctx file\n")
        ctx.write_trip_data(self.filepath + ".ctx")
        vdx = pytrip.vdx2.VdxCube("",ctx)
        wx.CallAfter(self.parent.write_to_log,"Writing vdx file\n")
        vdx.read_dicom(self.parent.data,self.parent.tab_voi.structures.keys())
        for dos in self.dose_info:
            voi = vdx.get_voi_by_name(dos["structure name"])
            if dos["target"] == '1':
                voi.type = '1'
            else:
                voi.type = '0'
        vdx.write_to_trip(self.filepath + ".vdx")
    def compress_files(self):
        wx.CallAfter(self.parent.write_to_log,"Compress Files\n")
        tar = tarfile.open("/home/jato/temp.tar.gz","w:gz")
        tar.add(self.path,arcname='dicomtemp')
        tar.close()
    def copy_files_to_server(self):
        login = self.server_settings
        transport = paramiko.Transport((login["server"],22))
        transport.connect(username=login["username"],password=login["password"])

        sftp = paramiko.SFTPClient.from_transport(transport)

        wx.CallAfter(self.parent.write_to_log,"Copy files to cluster\n")
        sftp.put('/home/jato/temp.tar.gz','temp.tar.gz')
        sftp.close()
        transport.close()
    def run_ssh_command(self,cmd):
        login = self.server_settings
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(login["server"],username=login["username"],password=login["password"])
        self.parent.write_to_log("Run Trip\n")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        for line in stdout:
            wx.CallAfter(self.parent.write_to_log,line)
        ssh.close()
    def copy_back_from_server(self):
        transport = paramiko.Transport((login["server"],22))
        transport.connect(username=login["username"],password=login["password"])
        sftp = paramiko.SFTPClient.from_transport(transport)
        wx.CallAfter(self.parent.write_to_log,"Copy from cluster\n")
        sftp.get('temp.tar.gz','/home/jato/temp.tar.gz')
        sftp.close()
        transport.close()
    def decompress_data(self):
        output_folder = self.path
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        tar = tarfile.open("/home/jato/temp.tar.gz","r:gz")
        wx.CallAfter(self.parent.write_to_log,"Extract output files\n")
        tar.extractall("/home/jato/")
        wx.CallAfter(self.parent.write_to_log,'Done')
    def visualize_data(self):
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
        wx.CallAfter(pub.sendMessage,'patient.updated.raw_data', patient)

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
                field += "couch(" + str(val["couch"]) + ") "
            if "gantry" in val:
                field += "gantry(" + str(val["gantry"]) + ") "
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
            output.append("voi " + oar["voi_name"].replace(" ","_") + " / maxdosefraction(" + oar["dose"] + ") oarset")
        output.append("voi * /list")
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
