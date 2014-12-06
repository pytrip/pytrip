"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
import wx
import copy
import numpy as np
import sys
if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
    from wx.lib.pubsub import setuparg1
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub


from util import *
from pytrip import dicomhelper
from pytrip.error import *
from pytrip.ctx import CtxCube
from pytrip.vdx import VdxCube
from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.ctimage import *
import time
from rbehandler import *
import json
from dose import *
from tripexecparser import *
import threading
import pytrip.tripexecuter as pte
import gc
from closeobj import *
notify = 1
def set_notify(a):
    global notify
    notify = a
def get_notify():
    return notify
class PytripData:
    def __init__(self):
        self.plans = TripPlanCollection()
        self.rbe = RBEHandler()
        self.patient_name = ""
        pub.subscribe(self.resend_data,"patient.request")
    def resend_data(self,msg):
        pub.sendMessage("patient.loaded.resend",self)
    def load_trip_exec(self,path):
        t = TripExecParser(self)
        t.parse_file(path)
    def get_rbe(self):
        return self.rbe
    def execute_trip(self,plan):
        executer = TripExecuter(self.get_images().get_modified_images(plan),self.get_rbe())
        t = threading.Thread(target=executer.execute,args=(plan,))
        t.start()
        pub.sendMessage("gui.triplog.open",executer)
        executer.visualize_data()
        executer.clean_up()
    def on_patient_update(self,msg):
        self.load_from_dicom(msg.data)
    def get_images(self):
        return self.ct_images
    def get_image_cube(self):
        return self.ct_images.get_voxelplan().cube
    def get_image_dimensions(self):
        c = self.ct_images.get_voxelplan()
        return [c.dimx,c.dimy,c.dimz]
    def delete(self):
        if hasattr(self,"structures"):
            del self.structures
    def save(self,path):
        out = self.plans.save()
        out["loaded_path"] = self.loaded_path
        with open(path,"w+") as fp:
            json.dump(out,fp,sort_keys=True,indent=4)
    def load(self,path):
        set_notify(False)
        with open(path,mode='r') as fp:
            data = json.load(fp)
        self.loaded_data = data
        load_image = self.load_images(data["loaded_path"])
        if load_image:
            self.plans.load(self.loaded_data["plans"],self.structures)
        set_notify(True)
        pub.sendMessage("patient.load",self)
        if len(self.plans.get_plans()):
            self.plans.set_active_plan(self.plans.get_plans()[0])
        
        
    def load_images(self,path):
        if not os.path.exists(path):
            return False
        if os.path.isdir(path):
            self.load_from_dicom(path)
        elif os.path.splitext(path)[1] == ".hed":
            self.load_from_voxelplan(path)
        return True
        
    def patient_load(self):
        if get_notify():
            pub.sendMessage("patient.load",self)
        
    
    def load_from_dicom(self,path,threaded=True):
        dcm = dicomhelper.read_dicom_folder(path)
        self.loaded_path = path
        close = CloseObj()
        if threaded:
            self.t = threading.Thread(target=self.load_from_dicom_thread,args=(dcm,close))
            self.t.start()
            pub.sendMessage("gui.wait.open",close)
        else:
            self.load_from_dicom_thread(dcm)
        
    
    def load_from_dicom_thread(self,dicom,close= None):
        if dicom.has_key('images'):
            c = CtxCube()
            c.read_dicom(dicom)
            self.ct_images = CTImages(c)
            self.patient_name = c.patient_name
        
        self.structures = VoiCollection(self)
        if dicom.has_key('rtss'):
            structures = VdxCube("",c)
            structures.read_dicom(dicom)
            for voi in structures.vois:
                self.structures.add_voi(Voi(voi.get_name(),voi),0)
        if not close is None:
            wx.CallAfter(close.close)

        wx.CallAfter(self.patient_load)
        
    def load_ctx_cube(self,cube):
        self.patient_name = cube.patient_name
        self.ct_images = CTImages(cube)
        self.structures = VoiCollection(self)
        wx.CallAfter(self.patient_load)
        
    def load_voi(self,voi,selected = False):
        v = Voi(voi.get_name(),voi)
        if selected:
            v.toogle_selected()
        self.structures.add_voi(v,0)
        pub.sendMessage("patient.voi.added",v)
    def load_from_voxelplan(self,path,threaded=True):
        self.loaded_path = path
        close = CloseObj()
        if threaded:
            self.t = threading.Thread(target=self.load_from_voxelplan_thread,args=(path,close))
            self.t.start()
            pub.sendMessage("gui.wait.open",close)
        else:
            self.load_from_voxelplan_thread(path)
        
    
    def load_from_voxelplan_thread(self,path,close=None):
        clean_path = os.path.splitext(path)[0]
        if os.path.exists(clean_path+".ctx"):
            c = CtxCube()
            c.read(clean_path+".ctx")
            self.ct_images = CTImages(c)
        else:
            raise InputError("No Images")
        self.structures = VoiCollection(self)
        if os.path.exists(clean_path+".vdx"):
            structures = VdxCube("",c)
            structures.read(clean_path+".vdx")
            for voi in structures.vois:
                self.structures.add_voi(Voi(voi.get_name(),voi),0)
        self.patient_name = c.patient_name
        if not close is None:
            wx.CallAfter(close.close)
        wx.CallAfter(self.patient_load)
        
    def get_plans(self):
        return self.plans
    def get_vois(self):
        return self.structures
        
class TripExecuter(pte.TripExecuter):
    def __init__(self,images,rbe):
        super(TripExecuter,self).__init__(images,rbe)
    def log(self,txt):
        txt = txt.replace("\n","")
        for l in self.listeners:
            wx.CallAfter(l.write,txt)
    def finish(self):
        for l in self.listeners:
            wx.CallAfter(l.finish)
class Voi(pte.Voi):
    def __init__(self,name,voi):
        self.icon = None
        super(Voi,self).__init__(name,voi)
    
    def get_icon(self):
        return self.icon
    def set_icon(self,icon):
        self.icon = icon    
    def toogle_selected(self):
        super(Voi,self).toogle_selected()
        pub.sendMessage("voi.selection_changed",self)
    
class TripVoi(pte.TripVoi):
    def __init__(self,voi):
        super(TripVoi,self).__init__(voi)
    def toogle_plan_selected(self,plan):
        super(TripVoi,self).toogle_plan_selected(plan)
        pub.sendMessage("plan.voi.plan_selected_change",{"voi":self,"plan":plan})
    def load(self,data):
        for i in data.keys():
            setattr(self,i,data[i])
        
    def set_max_dose_fraction(self,max_dose_fraction):
        super(TripVoi,self).set_max_dose_fraction(max_dose_fraction)
        pub.sendMessage('plan.voi.max_dose_fraction.changed',self)
        
    def set_dose(self,dose):
        super(TripVoi,self).set_dose(dose)
        pub.sendMessage('plan.voi.dose.changed',self)

class Field(pte.Field):
    def __init__(self,name):
        self.selected = False
        super(Field,self).__init__(name)
    def toogle_selected(self,plan):
        super(Field,self).toogle_selected()
        pub.sendMessage("plan.field.selection_changed",{"field":self,"plan":plan})
    def load(self,data):
        for i in data.keys():
            setattr(self,i,data[i])
            
    def set_name(self,name):
        super(Field,self).set_name(name)
        self.name = name
    def set_gantry(self,angle):
        super(Field,self).set_gantry(angle)
        pub.sendMessage('plan.field.gantry.changed',self)    
    
    def set_couch(self,angle):
        super(Field,self).set_couch(angle)
        pub.sendMessage('plan.field.couch',self)    
        
class FieldCollection(pte.FieldCollection):
    def __init__(self,plan):
        super(FieldCollection,self).__init__(plan)
    def load(self,data):
        for i in data["fields"]:
            f = Field(i["name"])
            self.add_field(f)
            f.load(i)
        
    def add_field(self,field):
        super(FieldCollection,self).add_field(field)
        if get_notify():
            pub.sendMessage('plan.field.added',{"plan":self.plan,"field":field})

    def remove_field(self,field):
        super(FieldCollection,self).remove_field(field)
        pub.sendMessage('plan.field.deleted',{"plan":self.plan,"field":field})
    
class VoiCollection(pte.VoiCollection):
    def __init__(self,parent):
        super(VoiCollection,self).__init__(parent)
    def load(self,data,structures):
        for voi in data["vois"]:
            d = structures.get_voi_by_name(voi["name"])
            d = self.add_voi(d)
            d.load(voi)
    def add_voi(self,voi,notify=1):
        if get_class_name(self.parent) == "TripPlan":
            voi = TripVoi(voi)
        if super(VoiCollection,self).add_voi(voi) and notify and get_notify():
            pub.sendMessage("plan.voi.added",{"plan":self.parent,"voi":voi})
        return voi
    def move_voi(self,voi,step):
        if super(VoiCollection,self).move_voi(voi,step):
            pub.sendMessage("plan.voi.moved",{"plan":self.parent,"voi":voi,"step":step})
    def delete_voi(self,voi,notify=1):
        self.vois.remove(voi)
        if notify:
            if get_class_name(self.parent) == "TripPlan":
                pub.sendMessage("plan.voi.remove",{"plan":self.parent,"voi":voi})
    def destroy(self):
        pass
class TripPlan(pte.TripPlan):
    def __init__(self,name="",comment=""):
        super(TripPlan,self).__init__(name,comment)
        self.fields = FieldCollection(self)
        self.vois = VoiCollection(self)
    def load(self,data,structures):
        for key in data.keys():
            if key == "fields":
                self.fields.load(data[key])
            elif key == "vois":
                self.vois.load(data[key],structures)
            else:
                setattr(self,key,data[key])
    def active_dose_change(self,dos):
        super(TripPlan,self).active_dose_change(dos)
        if get_notify():
            pub.sendMessage("plan.dose.active_changed",{"plan":self,"dose":dos})
        
    def load_let(self,path):
        let = super(TripPlan,self).load_let(path)
        pub.sendMessage("plan.let.added",{"plan":self,"let":let,"name":"LET"})
    def remove_let(self,let):
        super(TripPlan,self).remove_let(path)
        pub.sendMessage("plan.let.removed",{"plan":self,"let":let})
    

    def remove_dose(self,dos):
        super(TripPlan,self).remove_dose(dos)
        pub.sendMessage("plan.dose.removed",{"plan":self,"dose":dos})
    def remove_dose_by_type(self,type):
        dos = super(TripPlan,self).remove_dose_by_type(type)
        if not dos is None:
            pub.sendMessage("plan.dose.removed",{"plan":self,"dose":dos})
    def add_dose(self,dos,t=""):
        if type(dos) is DosCube:
            dos = DoseCube(dos,t)
        super(TripPlan,self).add_dose(dos)
        wx.CallAfter(pub.sendMessage,"plan.dose.added",{"plan":self,"dose":dos})
        
        
    def set_active_dose(self,dos):
        super(TripPlan,self).set_active_dose(dos)
        wx.CallAfter(pub.sendMessage,"plan.dose.active_changed",{"plan":self,"dose":dos})    
        
    def set_name(self,name):
        if super(TripPlan,self).set_name(name):
            pub.sendMessage("plan.renamed",self)
            return True
        return False

class TripPlanCollection(pte.TripPlanCollection):
    def __init__(self):
        super(TripPlanCollection,self).__init__()
    def load(self,data,structures):
        for plan in data:
            p = TripPlan(plan["name"],plan["comment"])
            self.add_plan(p)
            p.load(plan,structures)
            
    def add_plan(self,plan):
        super(TripPlanCollection,self).add_plan(plan)
        if get_notify():
            pub.sendMessage("plan.new",plan)
            pub.sendMessage("plan.active.changed",plan)
    
    def set_active_plan(self,plan):
        super(TripPlanCollection,self).set_active_plan(plan)
        if get_notify():
            pub.sendMessage("plan.active.changed",plan)
    def delete_plan(self,plan):
        new_plan = super(TripPlanCollection,self).delete_plan(plan)
        if not new_plan is None:
            pub.sendMessage("plan.active.changed",new_plan)
        pub.sendMessage("plan.deleted",plan)

    
