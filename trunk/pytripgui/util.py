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
import os,sys
def get_class_name(item):
    return item.__class__.__name__
def get_resource_path(res):
    return os.path.join(os.path.join(get_main_dir(),"res"),res)
def get_main_dir():
##    if "_MEIPASS2" in os.environ:
##        return os.environ["_MEIPASS2"]
##    else:
##        return os.path.dirname(__file__)
    if getattr(sys, 'frozen', False):
        return os.environ.get(
            "_MEIPASS2",
           os.path.abspath(".")
        )    
    else:
        return os.path.dirname(__file__)
    

def get_default_plugin_path():
    path = os.path.join(get_user_directory(),"plugins")
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def get_base_plugin_path():
    return os.path.join(get_main_dir(),"baseplugins")
def get_user_directory():
    path = os.path.join(os.path.expanduser("~"),".pytrip")
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def find_path(path,active_folder=None):
    path = os.path.expandvars(path)
    if os.path.exists(path):
        return path
    elif not active_folder is None and os.path.exists(os.path.join(active_folder,path)):
        return os.path.join(active_folder,path)
    return None
def get_args_from_string(string):
    temp = string[string.find("(")+1:string.find(")")]
    return temp.split(",")
def get_func_from_string(string):
    idx = string.find("(") 
    if idx > -1:
        return string[0:idx]
    return None
def IsMSWindows():
    """Are we running on Windows?

    @rtype: Bool"""
    return wx.Platform=='__WXMSW__'
def IsMac():
    """Are we running on Mac

    @rtype: Bool"""
    return wx.Platform=='__WXMAC__'

