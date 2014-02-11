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
import numpy as np

def get_empty_bitmap(width=32,height=32,color=[0,0,0] ):
    if len(color) is 0:
        color = [0,0,0]
    data = np.zeros((width,height,3),'uint8')
    data[:,:,] = color
    image = wx.EmptyImage(width,height)
    image.SetData(data.tostring())
    return image.ConvertToBitmap()
