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
import util
import sys
if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub

import os,imp
class PluginManager:
    def __init__(self):
        self.plugin_path = util.get_default_plugin_path()
        pub.subscribe(self.on_plugin_path_change,"general.plugin.path")
        pub.sendMessage("settings.value.request","general.plugin.path")
    def load_modules(self):
        basedir = util.get_base_plugin_path()
        modules = []
        for module in os.listdir(basedir):
            modules.append(module)
        for module in os.listdir(self.plugin_path):
            modules.append(module)
        self.plugins = []
        loaded_modules = []
        for module in modules:
            
            name = module.split('.')[0]
            if not name in loaded_modules:
                if not (name == '__init__' or name == ''):
                    try:
                        f, filename, description = imp.find_module(module, [self.plugin_path, basedir])
                        self.plugins.append(imp.load_module(module,f,filename,description))
                        if not (description[2] == imp.PKG_DIRECTORY):
                            f.close()
                    except ImportError as e:
                        pass
                    
    def get_plugins_by_type(self,type):
        list = []
        for plugin in self.plugins:
            if plugin.pluginProperties()["plugin_type"] == type:
                list.append(plugin)
        return list
    def on_plugin_path_change(self,msg):
        if msg.data is not None:
            self.plugin_path = msg.data
    
