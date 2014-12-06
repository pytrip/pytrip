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
from data import *
from util import *
import guihelper
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


class LeftMenuTree(wx.TreeCtrl):
    def __init__(self, *args, **kwargs):
        super(LeftMenuTree,self).__init__(*args, **kwargs)
        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK,self.on_leftmenu_rightclick)
        self.Bind(wx.EVT_TREE_END_LABEL_EDIT,self.end_edit)
        self.Bind(wx.EVT_TREE_BEGIN_DRAG,self.begin_drag)
        self.Bind(wx.EVT_TREE_END_DRAG,self.end_drag)
        self.context_menu = {"images":[{"text":"View"}],
            "image":[{"text":"View","callback":self.show_image}],
            "plans":[{"text":"New Plan","callback":self.new_plan},{"text":"New Empty Plan","callback":self.new_empty_plan}],
            "TripPlan":[{"text":"Set Active","callback":self.plan_set_active},
                        {"text":"Add Field","callback":self.plan_add_field},
                        {"text":"Export","type":"submenu","submenu":[
                            {"text":"Voxelplan","callback":self.plan_export_exec},
                            {"text":"Cube","callback":self.plan_export_cube}
                        ]},
                        {"text":"Import","type":"submenu","submenu":[
                            {"text":"Import Dose (Voxelplan)","callback":self.plan_load_dose_voxelplan},
                            {"text":"Import LET (Voxelplan)","callback":self.plan_load_let_voxelplan}
                        ]},
                        {"text":"Calculate","type":"submenu","submenu":[
                            {"text":"Execute TRiP","callback":self.plan_run_trip}
                        ]},
                        {"text":"Edit","callback":self.edit_label},
                        {"text":"Delete","callback":self.delete_plan},
                        {"text":"Properites","callback":self.tripplan_properties}],
            "DoseCube":[{"text":"Delete","callback":self.plan_remove_dose},
                        {"text":"Set Active For Plan","callback":self.plan_set_active_dose},
                        {"text":"Properties","callback":self.plan_dose_properties}
                    ],
            "LETCube":[{"text":"Delete","callback":self.plan_remove_let}],
                
            "Voi":self.generate_voi_menu,
            "TripVoi":[{"text":"Select","type":"check","value":"is_selected","callback":self.toogle_selected_voi},
                    {"text":"Target","type":"check","value":"is_target","callback":self.plan_toogle_target},
                    {"text":"OAR","type":"check","value":"is_oar","callback":self.plan_toogle_oar},
                    {"text":"Move Up","callback":self.plan_up_voi},
                    {"text":"Move Down","callback":self.plan_down_voi},
                    {"text":"Delete","callback":self.plan_delete_voi},
                    {"text":"Properties","callback":self.tripvoi_properties}],
            "MainVoi":[{"text":"Select","type":"check","value":"is_selected","callback":self.toogle_selected_voi},
                    {"text":"Add To Plan","type":"submenu","submenu":self.plan_submenu}],
            
            "FieldCollection":[{"text":"Add Field","callback":self.plan_add_field}],
            "Field":[{"text":"Delete","callback":self.plan_delete_field},
                    {"text":"Properties","callback":self.plan_properties_field}]
            
            }
        pub.subscribe(self.on_patient_load,"patient.load")
        pub.subscribe(self.voi_added,"patient.voi.added")
        pub.subscribe(self.plan_added,"plan.new")
        pub.subscribe(self.plan_renamed,"plan.renamed")
        pub.subscribe(self.plan_deleted,"plan.deleted")
        pub.subscribe(self.plan_voi_added,"plan.voi.added")
        pub.subscribe(self.plan_voi_removed,"plan.voi.remove")
        pub.subscribe(self.plan_field_added,"plan.field.added")
        pub.subscribe(self.plan_field_deleted,"plan.field.deleted")
        pub.subscribe(self.plan_dose_add,"plan.dose.added")
        pub.subscribe(self.plan_dose_removed,"plan.dose.removed")
        pub.subscribe(self.plan_let_add,"plan.let.added")
        pub.subscribe(self.plan_let_removed,"plan.let.removed")
        pub.subscribe(self.plan_voi_moved,"plan.voi.moved")
        
        
        
        
        pub.subscribe(self.on_import_path_change_dicom,"general.import.dicom_path")
        pub.subscribe(self.on_import_path_change_voxelplan,"general.import.voxelplan_path")
        pub.sendMessage("settings.value.request","general.import.dicom_path")
        pub.sendMessage("settings.value.request","general.import.voxelplan_path")
        self.prepare_icons()
    def prepare_icons(self):
        self.icon_size = (16,16)
        self.image_list = wx.ImageList(self.icon_size[0],self.icon_size[1])
        self.AssignImageList(self.image_list)
        
    def toogle_selected_voi(self,evt):
        voi = self.GetItemData(self.selected_item).GetData()
        voi.toogle_selected()
    def on_import_path_change_dicom(self,msg):
        self.dicom_path = msg.data
        if self.dicom_path is None:
            self.dicom_path = ""
    def on_import_path_change_voxelplan(self,msg): 
        self.voxelplan_path = msg.data
        if self.voxelplan_path is None:
            self.voxelplan_path = ""
    def show_image(self,evt):
        a = plan = self.GetItemData(self.selected_item).GetData()
        id = int(a.split(" ")[1])
        pub.sendMessage("2dplot.image.active_id",id)
    def plan_view_dose(self,evt):
        dose = self.GetItemData(self.selected_item).GetData()
        pub.sendMessage("2dplot.dose.set",dose)
    def get_parent_plan_data(self,node):
        item = node
        while True:
            data = self.GetItemData(item).GetData()
            if get_class_name(data) == "TripPlan":
                return data
            item = self.GetItemParent(item)
            if item is None:
                return None
    def delete_node_from_data(self,parent,data):
        child,cookie = self.GetFirstChild(parent)
        while child:
            if self.GetItemData(child).GetData() is data:
                
                self.Delete(child)
            child,cookie = self.GetNextChild(parent,cookie)
    def set_label_from_data(self,parent,data,text):
        child,cookie = self.GetFirstChild(parent)
        while child:
            if self.GetItemData(child).GetData() is data:
                self.SetItemText(child,text)
                break
            child,cookie = self.GetNextChild(parent,cookie)
    def get_child_from_data(self,parent,data):
        child,cookie = self.GetFirstChild(parent)
        while child:
            if self.GetItemData(child).GetData() is data:
                return child
            child,cookie = self.GetNextChild(parent,cookie)
        return None
    def search_by_data(self,root,data):
        data = None
        item,cookie = self.GetFirstChild(root)
        while item:
            if self.GetItemData(item).GetData() is data:
                return self.GetItemData(item).GetData()
            if self.GetChildrenCount(item) > 0:
                data = self.search_by_data(item,data)
            if data is not None:
                return data
            item,cookie = self.GetNextChild(item,cookie)
        return data
    def plan_properties_field(self,evt):
        field = self.get_field_from_node()
        pub.sendMessage("gui.field.open",field)
    
    def plan_dose_properties(self,evt):
        dosecube = self.GetItemData(self.selected_item).GetData() 
        pub.sendMessage("gui.dose.open",dosecube)
    
    def tripplan_properties(self,evt):
        plan = self.GetItemData(self.selected_item).GetData()
        pub.sendMessage("gui.tripplan.open",plan)    
    
    def tripvoi_properties(self,evt):
        voi = self.GetItemData(self.selected_item).GetData()
        pub.sendMessage("gui.tripvoi.open",voi)    
    def plan_field_deleted(self,msg):
        plan = msg.data["plan"]
        field = msg.data["field"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        fields_node = self.get_child_from_data(plan_node,plan.get_fields())
        self.Delete(self.get_child_from_data(fields_node,field))
        if self.GetChildrenCount(fields_node) is 0:
            self.Delete(fields_node) 
    def plan_load_dose_voxelplan(self,evt):
        plan = self.GetItemData(self.selected_item).GetData()
        dlg = wx.FileDialog(
			self,
            defaultFile=self.voxelplan_path,
            wildcard="Voxelplan headerfile (*.hed)|*.hed|",
			message="Choose headerfile")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            pub.sendMessage("settings.value.updated",{"general.import.voxelplan_path":path})
            plan.load_dose(path,"phys")
    def plan_load_let_voxelplan(self,evt):
        plan = self.GetItemData(self.selected_item).GetData()
        dlg = wx.FileDialog(
			self,
            defaultFile=self.voxelplan_path,
            wildcard="Voxelplan headerfile (*.hed)|*.hed|",
			message="Choose headerfile")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            pub.sendMessage("settings.value.updated",{"general.import.voxelplan_path":path})
            plan.load_let(path)
        
    def plan_delete_field(self,evt):
        fields = self.GetItemData(self.GetItemParent(self.selected_item)).GetData()
        field = self.get_field_from_node()
        fields.remove_field(field)
    def get_field_from_node(self,node = None):
        if node is None:
            node = self.selected_item
        return self.GetItemData(node).GetData()
    def plan_dose_add(self,msg):
        plan = msg.data["plan"]
        dose = msg.data["dose"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        doselist = self.get_or_create_child(plan_node,"Dose","dose")
        self.get_or_create_child(doselist,dose.get_type(),dose)
    
    def plan_dose_removed(self,msg):
        plan = msg.data["plan"]
        dose = msg.data["dose"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        doselist = self.get_or_create_child(plan_node,"Dose","dose")
        dose = self.get_child_from_data(doselist,dose)
        self.Delete(dose)
        if self.GetChildrenCount(doselist) is 0:
            self.Delete(doselist) 
    def plan_export_exec(self,evt):
        plan = self.GetItemData(self.selected_item).GetData()
        pub.sendMessage("gui.tripexport.open",plan)
    def plan_export_cube(self,evt):
        plan = self.GetItemData(self.selected_item).GetData()
        pub.sendMessage("gui.tripcubeexport.open",plan)
        
    
    def plan_let_add(self,msg):
        plan = msg.data["plan"]
        let = msg.data["let"]
        name = msg.data["name"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        let = self.get_or_create_child(plan_node,"LET",let)
    
    def plan_let_removed(self,msg):
        plan = msg.data["plan"]
        let = msg.data["let"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        let = self.get_or_create_child(plan_node,"",let)
        self.Delete(let)
        
    def plan_remove_let(self,evt):
        plan = self.get_parent_plan_data(self.selected_item)
        plan.remove_let(self.GetItemData(self.selected_item).GetData())
            
    def plan_remove_dose(self,evt):
        plan = self.get_parent_plan_data(self.selected_item)
        plan.remove_dose(self.GetItemData(self.selected_item).GetData())
    
    def plan_set_active_dose(self,evt):
        plan = self.get_parent_plan_data(self.selected_item)
        plan.set_active_dose(self.GetItemData(self.selected_item).GetData())

    def plan_field_added(self,msg):
        plan = msg.data["plan"]
        field = msg.data["field"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        fields = self.get_or_create_child(plan_node,"Fields",plan.get_fields())
        data = wx.TreeItemData()
        data.SetData(field)
        self.AppendItem(fields,field.get_name(),data=data)
    def plan_run_trip(self,evt):
        plan = self.GetItemData(self.selected_item).GetData()
        self.data.execute_trip(plan)
    def plan_add_field(self,evt):
        plan = self.get_parent_plan_data(self.selected_item)
        plan.add_field(Field(""))
    def plan_set_active(self,evt):
        plan = self.get_parent_plan_data(self.selected_item)
        self.data.plans.set_active_plan(plan)
    def plan_toogle_oar(self,evt):
        voi = self.GetItemData(self.selected_item).GetData()
        voi.toogle_oar()
    def plan_toogle_target(self,evt):
        voi = self.GetItemData(self.selected_item).GetData()
        voi.toogle_target()
    def plan_delete_voi(self,evt):        
        vois_item = self.GetItemParent(self.selected_item)
        vois_data = self.GetItemData(vois_item).GetData()
        voi = self.GetItemData(self.selected_item).GetData()
        vois_data.delete_voi(voi)
    
    def plan_up_voi(self,evt):
        vois_item = self.GetItemParent(self.selected_item)
        vois_data = self.GetItemData(vois_item).GetData()
        voi = self.GetItemData(self.selected_item).GetData()
        vois_data.move_voi(voi,-1)
    def plan_down_voi(self,evt):
        vois_item = self.GetItemParent(self.selected_item)
        vois_data = self.GetItemData(vois_item).GetData()
        voi = self.GetItemData(self.selected_item).GetData()
        vois_data.move_voi(voi,1)
        
        
    def plan_voi_removed(self,msg):
        plan = msg.data["plan"]
        voi = msg.data["voi"]
        plan_node = self.get_child_from_data(self.plans_node,plan)
        vois_node = self.get_or_create_child(plan_node,"",plan.get_vois())
        item = self.get_or_create_child(vois_node,"",voi)
        self.Delete(item)
        if self.GetChildrenCount(vois_node) is 0:
            self.Delete(vois_node)
                
    def plan_voi_added(self,msg):
        plan = msg.data["plan"]
        voi = msg.data["voi"]
        node = self.get_child_from_data(self.plans_node,plan)
        item = self.get_or_create_child(node,"Structures",plan.get_vois())
        data = wx.TreeItemData()
        data.SetData(voi)
        i2 = self.AppendItem(item,voi.get_name(),data=data)
        self.SetItemImage(i2, voi.get_voi().get_icon(), wx.TreeItemIcon_Normal)
        self.Expand(item)
        self.Expand(self.GetItemParent(item))
    def plan_voi_moved(self,msg):
        plan = msg.data["plan"]
        voi = msg.data["voi"]
        step = msg.data["step"]
        node = self.get_child_from_data(self.plans_node,plan)
        item = self.get_or_create_child(node,"Structures",plan.get_vois())
        child = self.get_child_from_data(item,voi)
        child2 = child
        if step < 0:
            for i in range(abs(step)+1):
                child2 = self.GetPrevSibling(child2)
        elif step > 0:
            for i in range(abs(step)):
                child2 = self.GetNextSibling(child2)
        data = wx.TreeItemData()
        data.SetData(voi)
        item = self.InsertItem(item,child2,self.GetItemText(child),data=data)
        self.SetItemImage(item, voi.get_voi().get_icon(), wx.TreeItemIcon_Normal)
        self.Delete(child)
        
    def get_or_create_child(self,parent,text,data):
        item = self.get_child_from_data(parent,data)
        if item:
            return item
        treedata = wx.TreeItemData()
        treedata.SetData(data)
        item = self.AppendItem(parent,text,data=treedata)
        return item 
        
    def begin_drag(self,evt):
        self.drag_data = self.GetItemData(evt.GetItem()).GetData()
        self.drag_item = evt.GetItem()
        if get_class_name(self.drag_data) in ["Voi","TripVoi"]:
            evt.Allow()
    def end_drag(self,evt):
        data = self.GetItemData(evt.GetItem()).GetData()
        class_name = get_class_name(data)
        if get_class_name(self.drag_data) == "Voi":
            if class_name in ["TripPlan","VoiCollection"]:
                data.add_voi(self.drag_data)
            if class_name == "TripVoi":
                item = self.GetItemData(self.GetItemParent(self.GetItemParent(evt.GetItem()))).GetData()
                if get_class_name(item) == "TripPlan":
                    item.add_voi(self.drag_data)
        elif get_class_name(self.drag_data) == "TripVoi":
            if class_name == "TripVoi":
                end = self.get_index_of(evt.GetItem())
                vois = self.GetItemData(self.GetItemParent(evt.GetItem())).GetData()
            elif class_name == "VoiCollection":
                end = 0
                vois = data
            else:
                return
            start = self.get_index_of(self.drag_item)
            step = end-start
            vois.move_voi(self.drag_data,step)
            
    def get_index_of(self,item):
        parent = self.GetItemParent(item)
        data = self.GetItemData(item).GetData()
        child,cookie = self.GetFirstChild(parent)
        i = 0
        n = self.GetChildrenCount(parent)
        while self.GetItemData(child).GetData() is not data and i < n:
            child,cookie = self.GetNextChild(parent,cookie)
            i += 1
        return i
            
    def plan_added(self,msg):
        data = wx.TreeItemData()
        data.SetData(msg.data)
        self.AppendItem(self.plans_node,msg.data.name,data=data)
        self.Expand(self.plans_node)
    def plan_renamed(self,msg):
        self.set_label_from_data(self.plans_node,msg.data,msg.data.get_name())
    def plan_deleted(self,msg):
        self.delete_node_from_data(self.plans_node,msg.data)

    
    def delete_plan(self,evt):
        data = self.GetItemData(self.selected_item).GetData()
        self.data.plans.delete_plan(data)
        
    def edit_label(self,evt):
        self.EditLabel(self.selected_item)
    def end_edit(self,evt):
        item = self.GetItemData(evt.GetItem()).GetData()
        if len(evt.GetLabel()) is 0 or not hasattr(item,"set_name") or not item.set_name(evt.GetLabel()):
            evt.Veto()
    def voi_added(self,msg):
        voi = msg.data
        data = wx.TreeItemData()
        data.SetData(voi)
        item = self.AppendItem(self.structure_node,voi.get_name(),data=data)
        img = self.image_list.Add(guihelper.get_empty_bitmap(self.icon_size[0],self.icon_size[1],voi.get_color()))
        voi.set_icon(img)
        self.SetItemImage(item, img, wx.TreeItemIcon_Normal)
    def on_patient_load(self,msg):
        self.data = msg.data
        self.populate_tree()
    def populate_tree(self):
        self.DeleteAllItems()
        self.rootnode = self.AddRoot(self.data.patient_name)
        data = wx.TreeItemData()
        data.SetData("structures")
        self.structure_node = self.AppendItem(self.rootnode,"Structures",data=data)
        data = wx.TreeItemData()
        data.SetData("plans")
        self.plans_node = self.AppendItem(self.rootnode,"Plans",data=data)
        ctx = self.data.get_images().get_voxelplan()
        for voi in self.data.get_vois():
            data = wx.TreeItemData()
            data.SetData(voi)
            item = self.AppendItem(self.structure_node,voi.get_name(),data=data)
            img = self.image_list.Add(guihelper.get_empty_bitmap(self.icon_size[0],self.icon_size[1],voi.get_color()))
            voi.set_icon(img)
            self.SetItemImage(item, img, wx.TreeItemIcon_Normal)
        for plan in self.data.get_plans():
            data = wx.TreeItemData()
            data.SetData(plan)
            p_id = self.AppendItem(self.plans_node,plan.name,data=data)
            if len(plan.get_vois()):
                item = self.get_or_create_child(p_id,"Structures",plan.get_vois())
                for voi in plan.get_vois():
                    node = self.get_child_from_data(self.plans_node,plan)
                    item = self.get_or_create_child(node,"Structures",plan.get_vois())
                    data = wx.TreeItemData()
                    data.SetData(voi)
                    i2 = self.AppendItem(item,voi.get_name(),data=data)
                    self.SetItemImage(i2, voi.get_voi().get_icon(), wx.TreeItemIcon_Normal)
                    self.Expand(item)
                    self.Expand(self.GetItemParent(item))
            if len(plan.get_fields()):
                fields = self.get_or_create_child(p_id,"Fields",plan.get_fields())
                for field in plan.get_fields():
                    data = wx.TreeItemData()
                    data.SetData(field)
                    self.AppendItem(fields,field.get_name(),data=data)
        self.Expand(self.rootnode)
        self.Expand(self.plans_node)

    def new_plan(self,evt):
        plan = TripPlan()
        self.data.plans.add_plan(plan)
        for voi in self.data.get_vois():
            plan.add_voi(voi)
        plan.add_field(Field("Field 1"))
            
    def new_empty_plan(self,evt):
        self.data.plans.add_plan(TripPlan())
    def generate_voi_menu(self,node):
        data = self.GetItemData(self.GetItemParent(self.GetItemParent(node)))
        if data is not None and get_class_name(data.GetData()) == "TripPlan":
            return self.context_menu["TripVoi"]
        return self.context_menu["MainVoi"]
    def on_leftmenu_rightclick(self,evt):
        tree = evt.GetEventObject()
        selected_data = tree.GetItemData(evt.GetItem()).GetData()
        if type(selected_data) is str:
            menu_name = selected_data.split(" ")[0]
        else:
            menu_name = get_class_name(selected_data)
        if menu_name in self.context_menu:
            show_menu = False
            self.selected_item = evt.GetItem()
            
            menu_points = self.context_menu[menu_name]
            if type(menu_points) is not list:
                menu_points = menu_points(self.selected_item)
             
            menu = self.build_menu(menu_points,selected_data)
            self.PopupMenu(menu,evt.GetPoint())
            menu.Destroy()
    def build_menu(self,menu_points,selected_data):
        menu = wx.Menu()
        for menu_item in menu_points:
            id = wx.NewId()
            if "require" in menu_item:
                if getattr(selected_data,menu_item["require"])() is None:
                    continue
            if not "type" in menu_item:
                item = wx.MenuItem(menu,id,menu_item["text"])
                menu.AppendItem(item)
            elif menu_item["type"] == "check":
                item = menu.AppendCheckItem(id,menu_item["text"])
                if getattr(selected_data,menu_item["value"])() is True:
                    item.Check()
            elif menu_item["type"] == "submenu":
                if type(menu_item["submenu"]) is list:
                    item = self.build_menu(menu_item["submenu"],selected_data)
                else:
                    item = menu_item["submenu"]()
                item = menu.AppendSubMenu(item,menu_item["text"])
            if "callback" in menu_item:
                wx.EVT_MENU(self,id,menu_item["callback"])
        return menu
    def plan_add_voi(self,evt):
        name = evt.GetEventObject().GetLabel(evt.GetId())
        self.data.plans.get_plan(name).add_voi(self.GetItemData(self.selected_item).GetData())
    def plan_submenu(self):
        submenu = wx.Menu()
        for plan in self.data.plans:
            id = wx.NewId()
            item = wx.MenuItem(submenu,id,plan.get_name())
            submenu.AppendItem(item)
            wx.EVT_MENU(submenu,id,self.plan_add_voi)
        return submenu
        
    
