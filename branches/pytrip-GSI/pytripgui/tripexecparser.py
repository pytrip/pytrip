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
import os,re
import data
from util import *

class TripExecParser:
    def __init__(self,data):
        self.data = data
    def parse_file(self,path):
        self.folder = os.path.dirname(path)
        with open(path,"r") as fp:
            data = fp.read()
        data = data.split("\n")
        for line in data:
            if line.find("ct") is 0:
                self.load_ct(line)
            if line.find("field") is 0:
                self.load_field(line)
            if line.find("plan") is 0:
                self.load_plan(line)
            if line.find("opt") is 0:
                self.load_plan(line)

    def load_ct(self,line):
        items = line.split("/")
        if len(items) > 1:
            path = items[0].split()[1]
            args = items[1].split()
            if "read" in args:
                ctx_file = os.path.splitext(path)[0] + ".ctx"
                path = find_path(ctx_file,self.folder)
                if not path is None:
                    self.data.load_from_voxelplan(path)
                    self.plan = data.TripPlan()
                    self.data.get_plans().add_plan(self.plan)
    def try_set(self,obj,arg,dic):    
        arg = arg.lower()
        func = get_func_from_string(arg)
        if not func is None:
            arguments = get_args_from_string(arg)
            if hasattr(obj,"set_"+func):
                getattr(obj,"set_"+func)(*arguments)
            elif func in dic.keys():
                getattr(obj,dic[func])(*arguments)
    def load_plan(self,line):
        items = line.split("/")
        dic = {"dosealgorithm":"set_dose_algorithm","optalgorithm":"set_opt_algorithm","bioalgorithm":"set_bio_algorithm"}
        if len(items) > 1:
            args = items[1].split()
            if hasattr(self,"plan"):
                for arg in args:
                    if self.try_set(self.plan,arg,dic) is None:
                        pass #should set i yourself 
                        
    def load_field(self,line):
        items = line.split("/")
        dic = {"raster":"set_rasterstep","proj":"set_projectile","doseext":"set_doseextension","contourext":"set_contourextension"}
        if len(items) > 1:
            args = items[1].split()
            if "new" in args:
                if hasattr(self,"plan"):
                    field = data.Field("Field %s"%(items[0].split()[1]))
                    self.plan.add_field(field)
                    for arg in args:
                        if self.try_set(field,arg,dic) is None:
                            pass #should set i yourself 
                        
                        
            
        
