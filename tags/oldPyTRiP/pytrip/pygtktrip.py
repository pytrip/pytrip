#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import sys
import os

import matplotlib
matplotlib.use('GTK')

from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.widgets import SpanSelector
from pylab import *

#from numpy import arange

try:
    import pygtk
    pygtk.require("2.0")
except:
    pass

try:
    import gtk
    import gtk.glade
except:
    sys.exit(1)

# DICOM handeling is only optional.
try:
    import dicom
    _dicom_loaded = True
except:
    _dicom_loaded = False
# 3D plotting is also only optional.
# Mayavi does not have a GTK backend, but one could run it in a different thread (and window).
try:
    from enthought.mayavi import mlab
    _mlab_loaded = True
except:
    _mlab_loaded = False


from pytrip import Vdx
from pytrip import CtxCube
from pytrip import DoseCube
from pytrip import __file__ # Hmmm, wonder if this is the right way to do it.

class PyGtkTRiP:
    """Smerg Smerg Smerg"""
    def __init__(self):

        # index fo which ones have been loaded.
        self.vdx = False
        self.hed = False  # special, see below.
        self.ctx = False
        self.dos = False
        self.tdos = False
        self.let = False
        
        # About self.hed: Any file loaded can have a header, even the
        # dicom files get one, when they are imported. However, only
        # one header can be used as a reference when building the
        # coordinate system and all the references herein.
        # This header is the one in self.hed.
        # So far, the policiy is to simply assign self.hed to the last
        # file which have been loaded to this particular header.
        # TODO: come up with a better solution?

        # NEW version: Load any number of data obejcts.

        self.do = [] # placeholder for all objects.
        self.cdo = -1 # placeholder for selected object
        
        #Set the Glade file
        path = os.path.dirname(__file__)
        self.gladefile = os.path.join(path,"pygtktrip.glade")
        self.wTree = gtk.Builder()
        self.wTree.add_from_file(self.gladefile)

        #Create our dictionay and connect it
        dic = { "on_button_test_clicked": self.on_button_test_clicked,
                #"on_button1_clicked" : self.on_button1_clicked,

                "on_button_resetvoi_clicked" : self.on_button_resetvoi_clicked,
                "on_button_writevoi_clicked" : self.on_button_writevoi_clicked,
                "on_button_buildcube_clicked" : self.on_button_buildcube_clicked,
                "on_button_oer_clicked" : self.on_button_oer_clicked,
                "on_notebook2_switch_page": self.change_page,
		"on_menu_openctx_activate": self.open_ctx,
		"on_menu_openvdx_activate": self.open_vdx,
		"on_menu_opendos_activate": self.open_dos,
		"on_menu_openlet_activate": self.open_let,
                "on_menu_opendicom_activate": self.open_dicom,
		"on_menu_quit_activate": self.my_quit,
                "on_menu_new_activate": self.foo,
                "on_menu_about_activate": self.foo,
                "on_window1_destroy_event" : self.my_quit,
                "on_window1_delete_event"  : self.my_quit,
                "on_drawingarea1_expose_event": self._do_expose,
                "on_spinbutton2_value_changed": self.change_dose,
                "on_spinbutton3_value_changed": self.change_prio,
                "on_drawingarea1_button1_press_event": self._do_expose,
		"on_checkbutton_ct_toggled": self.checkbutton_view,
		"on_checkbutton_voi_toggled": self.checkbutton_view,
                "on_checkbutton_cvoi_toggled": self.checkbutton_view,
		"on_checkbutton_dose_toggled": self.checkbutton_view,
                "on_checkbutton_tdose_toggled": self.foo, # To be implemented
		"on_checkbutton_let_toggled": self.checkbutton_view,
		"on_checkbutton_oer_toggled": self.checkbutton_view,
                "on_combobox1_changed": self.combobox1_change,
                "on_treeview1_cursor_changed": self.treeview1_select
             
                #"on_treeview1_select_cursor_row": self.treeview1_select,
                #"on_treeview1_toggle_cursor_row": self.treeview1_select,
                #"on_treeview1_row_activated": self.treeview1_select
               }
        
	self.statusbar1 = self.wTree.get_object("statusbar1")

	self.checkbutton_ct = self.wTree.get_object("checkbutton_ct")
	self.checkbutton_voi = self.wTree.get_object("checkbutton_voi")
        self.checkbutton_cvoi = self.wTree.get_object("checkbutton_cvoi")
	self.checkbutton_dose = self.wTree.get_object("checkbutton_dose")
        self.checkbutton_tdose = self.wTree.get_object("checkbutton_tdose")
	self.checkbutton_let = self.wTree.get_object("checkbutton_let")
	self.checkbutton_oer = self.wTree.get_object("checkbutton_oer")

        self.menu_opendicom = self.wTree.get_object("menu_opendicom")

        self.button_test = self.wTree.get_object("button_test")
        
        self.button1 = self.wTree.get_object("button1")
        self.button_oer = self.wTree.get_object("button_oer")
	self.spinbutton2 = self.wTree.get_object("spinbutton2")
        self.spinbutton3 = self.wTree.get_object("spinbutton3")
	
	self.radiobutton3 = self.wTree.get_object("radiobutton3")

        self.radiobutton_oer0 =  self.wTree.get_object("radiobutton_oer0")
        self.radiobutton_oer1 =  self.wTree.get_object("radiobutton_oer1")
        self.radiobutton_oer2 =  self.wTree.get_object("radiobutton_oer2")

        self.label_od_name = self.wTree.get_object("label_od_name")
        self.label_od_type = self.wTree.get_object("label_od_type")
        self.label_od_filename = self.wTree.get_object("label_od_filename")
        self.label_od_dimx = self.wTree.get_object("label_od_dimx")
        self.label_od_dimy = self.wTree.get_object("label_od_dimy")
        self.label_od_dimz = self.wTree.get_object("label_od_dimz")
        self.label_od_min  = self.wTree.get_object("label_od_min")
        self.label_od_max  = self.wTree.get_object("label_od_max")
        self.label_od_size = self.wTree.get_object("label_od_size")
        self.label_od_byteo = self.wTree.get_object("label_od_byteo")
        self.label_od_dtype = self.wTree.get_object("label_od_dtype")

        self.notebook2 = self.wTree.get_object("notebook2")
        self.hbox6 = self.wTree.get_object("hbox6") # trans
        self.hbox8 = self.wTree.get_object("hbox8") # sagg
        self.hbox9 = self.wTree.get_object("hbox9") # coronal

        self.label_hed = self.wTree.get_object("label_hed")
        self.label_vdx = self.wTree.get_object("label_vdx")
        self.label_dos = self.wTree.get_object("label_dos")
        self.label_ctx = self.wTree.get_object("label_ctx")

        self.treeview1 = self.wTree.get_object("treeview1")
        self.treeview2 = self.wTree.get_object("treeview2")
        self.combobox1 = self.wTree.get_object("combobox1")
        self.frame_oer = self.wTree.get_object("frame_oer")
        self.frame4 = self.wTree.get_object("frame4")

        self.window = self.wTree.get_object("window")
        self.hbox1 = self.wTree.get_object("hbox1")
	
        # default values 
	self.cx = 128 # bin
	self.cy = 128 # bin
	self.cz = 50  # bin 
        self.myslice = self.cz
        self.myaxis = "z"
        #self.radiobutton3.set_active(True)
        self.myvoi = 10                 # this is the GTV
        self.menu_opendicom.set_sensitive(_dicom_loaded)

        self.setup_plot()
        
        # now that the GUI is completed, connect the signals.
        # Dont do it before, emitted signals activate callbacks, which need
        # the above defaults.
        #self.wTree.signal_autoconnect(dic)
        self.wTree.connect_signals(dic)


        
    def setup_plot(self):
	print "--------- SETUP FIGURE ---------------"        
        self.fig  = Figure()   # Transversal
        self.fig2 = Figure()  # Sagittal
        self.fig3 = Figure()  # Coronal

        self.fig.set_facecolor('black')
        self.fig2.set_facecolor('black')
        self.fig3.set_facecolor('black')
        
        self.ax  = self.fig.add_subplot(111, autoscale_on=False)
        self.ax2 = self.fig2.add_subplot(111, autoscale_on=False)
        self.ax3 = self.fig3.add_subplot(111, autoscale_on=False)
        
        self.canvas  = FigureCanvasGTKAgg(self.fig) # a gtk.DrawingArea
        self.canvas2 = FigureCanvasGTKAgg(self.fig2) # a gtk.DrawingArea
        self.canvas3 = FigureCanvasGTKAgg(self.fig3) # a gtk.DrawingArea

	self.fig.canvas.mpl_connect('button_press_event', self.canvas_onclick)
	self.fig2.canvas.mpl_connect('button_press_event', self.canvas_onclick)
	self.fig3.canvas.mpl_connect('button_press_event', self.canvas_onclick)

	self.fig.canvas.mpl_connect('motion_notify_event', self.canvas_onmotion)
        self.fig2.canvas.mpl_connect('motion_notify_event', self.canvas_onmotion)
        self.fig3.canvas.mpl_connect('motion_notify_event', self.canvas_onmotion)

        self.fig.canvas.mpl_connect('scroll_event', self.canvas_scroll)
        self.fig2.canvas.mpl_connect('scroll_event', self.canvas_scroll)
        self.fig3.canvas.mpl_connect('scroll_event', self.canvas_scroll)
        
        print "SETUP: box pack start"
        #gtk.gdk.flush()        
        self.hbox6.pack_start(self.canvas, True, True)   # trans
        self.hbox8.pack_start(self.canvas2, True, True)  # sagg
        self.hbox9.pack_start(self.canvas3, True, True)  # coronal

        print "SETUP: done box pack start"
        #gtk.gdk.flush()

        # set the limits of the X and Y axis.
        # TODO: find more elegant solution ?
        self.ax.set_xlim(300,0)
        self.ax.set_ylim(300,0)

        self.ax2.set_xlim(300,0)
        self.ax2.set_ylim(300,0)

        self.ax3.set_xlim(300,0)
        self.ax3.set_ylim(300,0)

        # setup aspect and labels
        self.ax.set_aspect(1.0)
        self.ax2.set_aspect(1.0)
        self.ax3.set_aspect(1.0)
        #self.ax.set_xlabel('[mm]')
        #self.ax.set_ylabel('[mm]')

        # TODO: eeeeeeeeehh...
        #majorLocator   = MultipleLocator(1)
        #majorFormatter = FormatStrFormatter('%d')
        #minorLocator   = MultipleLocator(1.5)
        #self.ax.xaxis.set_minor_locator(minorLocator)

        # what does this do ?
        # self.ax.set_animated(True)        
        self.canvas.set_size_request(512, 512)
        self.canvas2.set_size_request(512, 512)
        self.canvas3.set_size_request(512, 512)
    
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        
        self.canvas.show()
        self.canvas2.show()
        self.canvas3.show()
                        
	self.window.window.set_cursor(None)
        # Det kan enhver komme og paastaa. :-/
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.background = self.canvas2.copy_from_bbox(self.ax2.bbox)
        self.background = self.canvas3.copy_from_bbox(self.ax3.bbox)

        # init for self.cax and self.ccanvas
        self.cax = self.ax
        self.ccanvas = self.canvas
	self.update_labels()
        
        print "SETUP_PLOT: done all."
        gtk.gdk.flush()

    def _do_expose(self, widget, event):
        print "_do_expose"
        self.canvas.draw()
        self.canvas.show()
                
    #def on_button1_clicked(self, widget):
        #print "Hello World!"
        #t = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
        #gtk.main_quit()

    def my_quit(self, widget, *foo):  # why foo?
        print "Bye bye"
        gtk.main_quit()

    def change_page(self,foo,bar,current_page):
        if self.no_data() == True:
            return()  # exit without doing anything.
        if current_page == 0: # Transversal
            self.myaxis = "z"
            self.cax = self.ax   # this wont go well if tabs are detached.
            self.myslice = self.cz
            self.ccanvas = self.canvas
        if current_page == 1: # Sagittal
            self.myaxis = "y"
            self.cax = self.ax2
            self.myslice = self.cy
            self.ccanvas = self.canvas2
        if current_page == 2: # Coronal
            self.myaxis = "x"
            self.cax = self.ax3
            self.myslice = self.cx
            self.ccanvas = self.canvas3            
        self.update_plot()
            
        
    def update_plot(self):
        #print " -------- update_plot() -----------"    
        #print "  update_plot(): self.myslice: ", self.myslice
        self.window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
        #fig = self.fig.get_figure()        
        #self.fig.set_edgecolor('white')
        #ax = self.fig.gca()
        self.cax.cla()
            
        # ---------------- VOI --------------------------------
	if (self.checkbutton_voi.get_active()) and (self.vdx != False):
            _ok_to_plot = False
            _slice_list = []
            for idx in range(len(self.voilist)):
                if self.voilist[idx].visible:  # if voi is set in treeview frame
                    self.X,self.Y,self.vData3 = self.vdx.get_slice( \
                        self.myaxis,self.myslice, idx)
                    # multiply with dose
                    _slice_list.append(self.vData3 * self.voilist[idx].dose) 
                    _ok_to_plot = True
            if _ok_to_plot:
                for _slice in _slice_list:
                    # but only plot, if this slice holds data
                    if _slice.max() > _slice.min() :
                        self.levels_voi = arange(0.0,  \
                                                 200.0,\
                                                 2)                    
                        self.cax.contourf(self.X,self.Y,_slice, \
                                         self.levels_voi,alpha=0.5,\
                                         antialiased=True,linewidths=None)
                self.cax.set_xlim(self.X.max(), self.X.min())
                self.cax.set_ylim(self.Y.max(), self.Y.min())

        # -------- alternative plotting: contours ------------------
        # this will only work if we are in the transversal viewing mode.
        # THIS plot will force some axis, which will cause flicker when
        # switching between dose and non dose view. The fix was realized
        # when moving this to the top. The succseeding CTX plot will
        # remove the axis.
        if (self.checkbutton_cvoi.get_active()) and (self.vdx != False):
            if self.myaxis == "z":
                for idx in range(len(self.voilist)):
                    # if show VOI is set in treeview.
                    if self.voilist[idx].visible: 
                        vertices = self.vdx.get_slice_vertices(self.myslice,idx)
                        if vertices != None:
                            self.cax.plot(vertices[:,0],vertices[:,1],'go-') 
                        else:
                            pass
                    else:
                        pass

            self.cax.set_xlim(self.vdx.rxmax, self.vdx.rxmin)
            self.cax.set_ylim(self.vdx.rymax, self.vdx.rymin)
            #self.cax.set_xlim(self.vdx.rxmin, self.vdx.rxmax)
            #self.cax.set_ylim(self.vdx.rymin, self.vdx.rymax)
                

        # ---------- CTX ---------------
	if (self.checkbutton_ct.get_active()) and (self.ctx != False):
            self.X,self.Y,self.cData = self.ctx.get_slice(self.myaxis,
                                                          self.myslice)
            self.levels_ct = arange(self.ctx.min+10,self.ctx.max+10,50.0)
            # not needed?
            #X = [self.X[0,0],self.X[self.X.shape[0]-1,0]]
            #Y = [self.Y[0,0],self.Y[self.Y.shape[0]-1,0]]
            # extent tells what the x and y coordinates are,
            # so matplotlib can make propper assignment.
            self.cax.imshow(self.cData,cmap=cm.gray,interpolation='bilinear',\
                           origin="lower",
                           extent=[self.X.min(),self.X.max(),
                                   self.Y.min(),self.Y.max()])
            self.cax.set_xlim(self.X.max(), self.X.min())
            self.cax.set_ylim(self.Y.max(), self.Y.min())


        # ------------------ DOSE ----------------------------        


        if (self.checkbutton_dose.get_active()) and (self.dos != False):
            self.X,self.Y,self.dData = self.dos.get_slice( \
                self.myaxis,
                self.myslice)

            # check if there is something to plot
            #print "NB:", self.dos.min, self.dos.max, \
            #             self.dData.min(), self.dData.max()
            if (self.dData.max() > 0):
                #cmap1 = cm.get_cmap()
                cmap1 = cm.jet
                cmap1.set_under("k",alpha=0.0)  # seems to be broken! :-( -- Sacrificed goat, now working - David
                cmap1.set_over("k",alpha=0.0)
		cmap1.set_bad("k",alpha=0.0)  # Sacrificial knife here
		tmpdat=ma.masked_where(self.dData <= self.dos.min ,self.dData) # Sacrifical goat
		im = self.cax.imshow(tmpdat,
                               interpolation='bilinear',
                               cmap=cmap1,
                               #norm=Normalize(vmin=self.dos.min+10,vmax=self.dos.max, clip=False)
                               alpha=0.7,  
                               origin="lower",
                               extent=[self.X.min(),self.X.max(),
                                       self.Y.min(),self.Y.max()])
                               
                self.cax.set_xlim(self.X.max(), self.X.min())
                self.cax.set_ylim(self.Y.max(), self.Y.min())
                #colorbar(im, extend='both', orientation='vertical', shrink=0.8)
        # ------------------ TARGET DOSE ----------------------------        
        if (self.checkbutton_tdose.get_active()) and (self.dos != False):
            self.X,self.Y,self.dData = self.dos.get_slice( \
                self.myaxis,
                self.myslice)
            cmap1 = cm.get_cmap()
            cmap1.set_under(color='k', alpha=0.0)
            
            self.cax.imshow(self.dData,
                           interpolation='bilinear',
                           cmap=cmap1,
                           alpha=0.8,
                           origin="lower",
                           extent=[self.X.min(),self.X.max(),
                                   self.Y.min(),self.Y.max()],
                           vmin=self.dData.min()+2,
                           vmax=self.dData.max()+1)
                           #vmin=10,
                           #vmax=520)
            self.cax.set_xlim(self.X.max(), self.X.min())
            self.cax.set_ylim(self.Y.max(), self.Y.min())

        # ------------------ LET ----------------------------     
        if (self.checkbutton_let.get_active()) and (self.let != False):
            self.X,self.Y,self.lData = self.let.get_slice( \
                self.myaxis,
                self.myslice)
            cmap1 = cm.get_cmap()
            cmap1.set_under(color='k', alpha=0.0)
            self.cax.imshow(self.lData,
                           interpolation='bilinear',
                           cmap=cmap1,
                           alpha=0.8,
                           origin="lower",
                           extent=[self.X.min(),self.X.max(),
                                   self.Y.min(),self.Y.max()],
                           vmin=self.lData.min()+2,
                           vmax=self.lData.max()+1)
            self.cax.set_xlim(self.X.max(), self.X.min())
            self.cax.set_ylim(self.Y.max(), self.Y.min())

        # TODO ---------------- OER -----------------------------
        if (self.checkbutton_oer.get_active()) and (self.oer != False):
            self.X,self.Y,self.oData = self.oer.get_slice( \
                self.myaxis,
                self.myslice)
            cmap1 = cm.Spectral
            cmap1.set_under(color='k', alpha=0.0)
            self.cax.imshow(self.oData,
                           interpolation='bilinear',
                           cmap=cmap1,
                           alpha=0.8,
                           origin="lower",
                           extent=[self.X.min(),self.X.max(),
                                   self.Y.min(),self.Y.max()],
                           vmin=self.oData.min(),
                           vmax=self.oData.max())
            self.cax.set_xlim(self.X.max(), self.X.min())
            self.cax.set_ylim(self.Y.max(), self.Y.min())
        # ---------------------------------------------------------
        if (self.no_data()):
            self.cax.set_xlim(300,0)
            self.cax.set_ylim(300,0)       

        self.update_labels()
        self.plot_text()
        self.cax.axis('off')
        self.ccanvas.draw()
        self.ccanvas.show()
	self.window.window.set_cursor(None)

    def update_labels(self):                
        #print "------- update_labels() -------------------"
        # ---------------- VDX ---------------------
        if self.vdx is not False:
            self.frame4.set_sensitive(True)
            self.checkbutton_voi.set_sensitive(True)
            if self.myaxis == "z":                
                self.checkbutton_cvoi.set_sensitive(True)
                # TODO: instead of forcing this,
                # state set by user should be remembered.
                self.checkbutton_cvoi.set_active(True)
            else:
                self.checkbutton_cvoi.set_sensitive(False)
                self.checkbutton_cvoi.set_active(False) 
            self.label_vdx.set_markup("VOIs: %s" % os.path.basename(self.fname_vdx) )
        else:
            self.frame4.set_sensitive(False)
            self.checkbutton_voi.set_sensitive(False)
            self.checkbutton_cvoi.set_sensitive(False)
            self.checkbutton_voi.set_active(False)
            self.checkbutton_voi.set_active(False)
            self.label_vdx.set_markup("VOIs: (none loaded)")
        # -------------- CTX -------------------------------
        if self.ctx is not False:
            self.checkbutton_ct.set_sensitive(True)
            self.label_ctx.set_markup("CT: %s" % os.path.basename(self.fname_ctx) )
        else:
            self.checkbutton_ct.set_sensitive(False)
            self.checkbutton_ct.set_active(False)
            self.label_ctx.set_markup("CT: (none loaded)")

        # -------------- DOS --------------------
        if self.dos is not False:
            self.checkbutton_dose.set_sensitive(True)
            self.label_dos.set_markup("Dose: %s" % os.path.basename(self.fname_dos) )
        else:
            self.label_dos.set_markup("Dose: (none loaded)")
            self.checkbutton_dose.set_sensitive(False)
            self.checkbutton_dose.set_active(False)
        # ---------------- TDOS ---------------------
        if self.tdos is not False:
            self.checkbutton_tdose.set_sensitive(True)
        else:
            self.checkbutton_tdose.set_sensitive(False)
            self.checkbutton_tdose.set_active(False)
        # ------------- LET --------------
        if self.let is not False:
            self.checkbutton_let.set_sensitive(True)
        else:
            self.checkbutton_let.set_sensitive(False)
            self.checkbutton_let.set_active(False)
            
        # deprecated ?
        if self.hed != False:
            self.label_hed.set_markup("HED: %s" % os.path.basename(self.fname_hed) )
        else:
            self.label_hed.set_markup("HED: (none loaded)")

        # NEW LABEL SET
        # cdo = current data object
        if self.cdo >= 0:
            self.label_od_name.set_markup(str(self.do[self.cdo].name))
            self.label_od_type.set_markup(str(self.do[self.cdo].type))
            # TODO: do better.
            if self.do[self.cdo].filename != None:
                self.label_od_filename.set_markup( \
                    str(os.path.basename(self.do[self.cdo].filename)))
            else:
                self.label_od_filename.set_markup(str(None))                
            self.label_od_dimx.set_markup(str(self.do[self.cdo].header.dimx)) 
            self.label_od_dimy.set_markup(str(self.do[self.cdo].header.dimy)) 
            self.label_od_dimz.set_markup(str(self.do[self.cdo].header.dimz)) 
            self.label_od_min.set_markup(str(self.do[self.cdo].min)) 
            self.label_od_max.set_markup(str(self.do[self.cdo].max)) 
            self.label_od_size.set_markup(str("none"))
            self.label_od_byteo.set_markup(str(self.do[self.cdo].header.byte_order))
            self.label_od_dtype.set_markup(str(self.do[self.cdo].header.data_type))

    def change_to_axis1(self,foo):
        """ Coronal view """                
        self.myaxis = "x"
        self.update_plot()

    def change_to_axis2(self,foo):
        """ Sagittal view """
        self.myaxis = "y"
        self.update_plot()

    def change_to_axis3(self,foo):
        """ Transversal view """
        self.myaxis = "z"
        self.update_plot()    
        
    def canvas_onclick(self,event):
	print 'CLICK button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
		event.button, event.x, event.y, event.xdata, event.ydata)

    def canvas_onmotion(self,event):
        if self.no_data() == False:
            if ((event.xdata != None) and (event.ydata!= None)):
                mystr = 'x=%d, y=%d, xdata=%.2f mm, ydata=%.2f mm' \
                        %(event.x, event.y, event.xdata,event.ydata)
                self.statusbar1.push(0,mystr)
                self._xdata = event.xdata
                self._ydata = event.ydata

    def canvas_scroll(self,event):
        if self.no_data() == False:
            if self.myaxis == "x":
                self.cx += event.step
                self.myslice = self.cx
            if self.myaxis == "y":
                self.cy += event.step
                self.myslice = self.cy
            if self.myaxis == "z":
                self.cz += event.step
                self.myslice = self.cz
            self.update_plot()
	
    def checkbutton_view(self,foo):
        self.update_plot()

    def choose_file(self, _suffix, _multi=False):
        """ File chooser for '_suffix' type files.
        _multi=True will return a filename list.
        _multi=False returns a single filename """
    	chooser = gtk.FileChooserDialog(title=None,
                                        action=gtk.FILE_CHOOSER_ACTION_OPEN,
	buttons=(gtk.STOCK_CANCEL,
                 gtk.RESPONSE_CANCEL,
                 gtk.STOCK_OPEN,
                 gtk.RESPONSE_OK))
	chooser.set_select_multiple(_multi) # True, when DICOM.
        # TODO: multiple file support for DOS,CTX,VDX.
	filter = gtk.FileFilter()
	filter.add_pattern(_suffix)
	chooser.add_filter(filter)
	response = chooser.run()	
	if response == gtk.RESPONSE_OK:
            if _multi:
                filenames = chooser.get_filenames()
                chooser.destroy()
                return(filenames)
            else:
                filename = chooser.get_filename()
                chooser.destroy()
                return(filename)
	elif response == gtk.RESPONSE_CANCEL:
            print "Closed, no files selected"
            chooser.destroy()
        gtk.gdk.flush()

    def open_ctx(self,foo):        
	filename = self.choose_file("*.ctx")
        if filename == None:
            return()
        if os.path.isfile(filename):
            self.window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
            while gtk.events_pending():
                gtk.main_iteration()
            self.ctx = CtxCube(filename)
            self.do.append(self.ctx) # New version
            self.cdo += 1
            self.hed = self.ctx.header
            self.fname_ctx = filename #TODO: use name in object instead.
            self.fname_hed = os.path.splitext(filename)[0]+".hed"
            self.checkbutton_ct.set_active(True)
            self.statusbar1.push(0,"Opened %s" % filename)
            self.window.window.set_cursor(None)
            while gtk.events_pending():
                gtk.main_iteration()
            self.update_plot()

    def open_vdx(self,foo):
	filename = self.choose_file("*.vdx")
        if filename == None:
            return()        
        if os.path.isfile(filename):
            self.window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
            while gtk.events_pending():
                gtk.main_iteration()
            self.vdx = Vdx(filename)
            self.do.append(self.vdx) #
            self.cdo += 1
            self.hed = self.vdx.header
            self.fname_vdx = filename
            self.fname_hed = os.path.splitext(filename)[0]+".hed"
            self.populate_voitree()
            self.statusbar1.push(0,"Opened %s" % filename)
            self.window.window.set_cursor(None)
            while gtk.events_pending():
                gtk.main_iteration()
            # set either fast contour mode (Transversal only) or slow fill mode.
            if self.myaxis == "z":
                # this one does update plot?
                self.checkbutton_cvoi.set_active(True) 
            else:
                self.checkbutton_voi.set_active(True)
            self.update_plot() # TODO: is this needed?
        
    def open_dos(self,foo):
	filename = self.choose_file("*.dos")
        if filename == None:
            return()
	print "open vdx"
        if os.path.isfile(filename):
            self.window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
            while gtk.events_pending():
                gtk.main_iteration()
            self.dos = DoseCube(filename,"DOSE")
            self.do.append(self.dos) #
            self.cdo += 1
            self.hed = self.dos.header
            self.fname_dos = filename
            self.fname_hed = os.path.splitext(filename)[0]+".hed"
            self.checkbutton_dose.set_active(True)
            self.statusbar1.push(0,"Opened %s" % filename)
            self.window.window.set_cursor(None)
            while gtk.events_pending():
                gtk.main_iteration()
            self.update_plot()

    def open_let(self,foo):
	filename = self.choose_file("*dosemlet.dos")
        if filename == None:
            return()
        if os.path.isfile(filename):
            self.window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
            while gtk.events_pending():
                gtk.main_iteration()
            self.let = DoseCube(filename,"LET")
            self.do.append(self.let) #
            self.cdo += 1
            self.hed = self.dos.header
            self.fname_let = filename
            self.fname_hed = os.path.splitext(filename)[0]+".hed"
            self.checkbutton_dose.set_active(True)
            self.frame_oer.set_sensitive(True)
            self.statusbar1.push(0,"Opened %s" % filename)
            self.window.window.set_cursor(None)
            while gtk.events_pending():
                gtk.main_iteration()            
            self.update_plot()

    def open_dicom(self,foo):
        filenames = self.choose_file("*.dcm", True)
        if filenames == None:
            return()
        # TODO: dos and ctx object should be the same,
        # with a modality string
        print "MAIN: ", filenames
        self.ctx = CtxCube()  # blah.
        self.ctx.import_dicom(filenames)
        self.fname_ctx = filenames[-1] # use last filename
        self.fname_hed = "none"
        self.hed = self.ctx.header
        self.checkbutton_ct.set_active(True)
        self.statusbar1.push(0,"Opened %s" % filenames[0])
        self.window.window.set_cursor(None)
        while gtk.events_pending():
            gtk.main_iteration()
        self.set_center_point(self.ctx) # set cx,cy,cz to center of ctx
        self.update_plot()

    def set_center_point(self, _co):
        """ This function will set the current axis to
        be in the middle of the data cube, which is contained
        inside the cube_object."""
        self.cx = int(_co.header.dimx * 0.5)
        self.cy = int(_co.header.dimy * 0.5)
        self.cz = int(_co.header.dimz * 0.5)
        if self.myaxis == "x":
            self.myslice = self.cx
        if self.myaxis == "y":
            self.myslice = self.cy
        if self.myaxis == "z":
            self.myslice = self.cz

    def voi_toggled(self, _cell, _path, _model):
        # update checkbox in treeview
        _iter = _model.get_iter(_path)
        _val = _model.get_value(_iter,0)  # (iter, column)
        _voi_idx = _model.get_value(_iter,2)  # get voi index
        #print "toggle voi", _val
        #print "voi nr", _voi_idx
        _val = not _val
        _model.set(_iter,0,_val)
        # update voilist
        self.voilist[_voi_idx].visible = _val
        # replot
        self.update_plot()

    def populate_tree(self):
        print "populate tree"
        self.tsvoi_model2 = gtk.TreeStore(bool, str, int, str, str)
        # TODO: only create instances if data are available.
        li_ctx = self.tsvoi_model2.append(None,[False, "CT", 0, "", ""])
        li_vdx = self.tsvoi_model2.append(None,[False, "VOI", 0, "", ""])
        li_dos = self.tsvoi_model2.append(None,[False, "Dose", 0, "", ""])
        li_let = self.tsvoi_model2.append(None,[False, "LET", 0, "", ""])
        li_oer = self.tsvoi_model2.append(None,[False, "OER", 0, "", ""])
        li_rbe = self.tsvoi_model2.append(None,[False, "RBE", 0, "", ""])

        for _do in self.do:
            if _do.type == "CTX":
                li = self.tsvoi_model2.append(li_ctx,
                                              [True, _do.name, 0, "", ""])
            if _do.type == "VDX":
                li = self.tsvoi_model2.append(li_vdx,
                                              [True, _do.name, 0, "", ""])
                # TODO : climb and append voi tree.
                #for i in range(_do.numberofvois):
                #    # TODO: move VOILIST into VDX Class, or find beter solv.
                #    self.tsvoi_model.append(li,[False,
                #                            _do.voi[i].name, i,\
                #                            str(self.voilist[i].dose),
                #                            str(self.voilist[i].priority)])
            if _do.type == "DOSE":
                li = self.tsvoi_model2.append(li_dos,
                                              [True, _do.name, 0, "", ""])
            if _do.type == "LET":
                li = self.tsvoi_model2.append(li_let,
                                              [True, _do.name, 0, "", ""])
            if _do.type == "OER":
                li = self.tsvoi_model2.append(li_oer,
                                              [True, _do.name, 0, "", ""])
            if _do.type == "RBE":
                li = self.tsvoi_model2.append(li_rbe,
                                              [True, _do.name, 0, "", ""])
        cellr_text = gtk.CellRendererText()
        cellr_check = gtk.CellRendererToggle()        
        cellr_check.connect('toggled', self.voi_toggled, self.tsvoi_model2)

        col =  gtk.TreeViewColumn("", cellr_check, active=0)
        col1 = gtk.TreeViewColumn("Name", cellr_text)
        col2 = gtk.TreeViewColumn("Dose", cellr_text)
        col3 = gtk.TreeViewColumn("Priority", cellr_text)

        col.pack_start(cellr_check, False)   # TODO: fix GtkWarning
        col1.pack_start(cellr_text, True)
        col2.pack_start(cellr_text, True)
        col3.pack_start(cellr_text, True)
        
        col.add_attribute(cellr_check, "active", 0) # checkbox
        col1.add_attribute(cellr_text, "text", 1)   # VOI name
        col2.add_attribute(cellr_text, "text", 3)   # Dose
        col3.add_attribute(cellr_text, "text", 4)   # Priority
        
        self.treeview2.set_model(self.tsvoi_model2)
        # only when run for the first time:
        if self.treeview2.get_columns() == [] :
            self.treeview2.append_column(col)
            self.treeview2.append_column(col1)
            self.treeview2.append_column(col2)
            self.treeview2.append_column(col3)        
            self.treeview2.expand_all()



    def populate_voitree(self):        
        print "populate_voitree()"
        # setup voi table, default all are false ( = deactivated).
        # active,name,dose prescribed dose level
        # initialize voilist:
        # [VISIBLE,NAME,DOSE,PRIO,INDEX]
        self.voilist = []
        print "voilist", self.voilist
        for i in range(self.vdx.numberofvois):
            self.voilist.append(GTKVoi(self.vdx.voi[i],0.0,10,i))            
        # ----------- COMBOBOX ------------
        _combo_store=gtk.ListStore(str)
        if self.vdx != False:
            for _voi in self.vdx.voi:
                print "combobox attribute:", _voi.name
                _combo_store.append([_voi.name])
            self.combobox1.set_sensitive(True)
            self.spinbutton2.set_sensitive(True)
            self.spinbutton3.set_sensitive(True)            
        else:
            print" combobox: nothing to be rendered"
            _combo_store.append(["(No VOIs)"])
            self.combobox1.set_sensitive(False)
            self.spinbutton2.set_sensitive(False)
            self.spinbutton3.set_sensitive(False)            
        self.combobox1.set_model(_combo_store)
        self.combobox1.set_active(0)
        _combo_cell = gtk.CellRendererText()
        self.combobox1.pack_start(_combo_cell,True)
        self.combobox1.add_attribute(_combo_cell, 'text', 0)
        # ----------- end of combobox -----------
        self.tsvoi_model = gtk.TreeStore(bool, str, int, str, str)
        ### append(parent,row) where row is a tuple with orderd col values.
        li = self.tsvoi_model.append(None,[True, "VOI", 0, "", ""])
        if self.vdx != False:
            for i in range(self.vdx.numberofvois):
                self.tsvoi_model.append(li,[False,
                                            self.vdx.voi[i].name, i,\
                                            str(self.voilist[i].dose),
                                            str(self.voilist[i].priority)])
        #cell_img = gtk.CellRendererPixbuf()
        cellr_text = gtk.CellRendererText()
        cellr_check = gtk.CellRendererToggle()        
        cellr_check.connect('toggled', self.voi_toggled, self.tsvoi_model)

        col =  gtk.TreeViewColumn("", cellr_check, active=0)
        col1 = gtk.TreeViewColumn("VOI Name", cellr_text)
        col2 = gtk.TreeViewColumn("Dose", cellr_text)
        col3 = gtk.TreeViewColumn("Priority", cellr_text)

        col.pack_start(cellr_check, False)   # TODO: fix GtkWarning
        col1.pack_start(cellr_text, True)
        col2.pack_start(cellr_text, True)
        col3.pack_start(cellr_text, True)
        
        col.add_attribute(cellr_check, "active", 0) # checkbox
        col1.add_attribute(cellr_text, "text", 1)   # VOI name
        col2.add_attribute(cellr_text, "text", 3)   # Dose
        col3.add_attribute(cellr_text, "text", 4)   # Priority
        
        self.treeview1.set_model(self.tsvoi_model)
        self.treeview1.append_column(col)
        self.treeview1.append_column(col1)
        self.treeview1.append_column(col2)
        self.treeview1.append_column(col3)        
        self.treeview1.expand_all()
        # update cursor in treeview to match combobox.
        self.combobox1_change(None)        

    def no_data(self):  
        """ Returns True if no data is available """
        if (self.dos is False) and \
               (self.tdos is False) and \
               (self.vdx is False) and \
               (self.ctx is False) and \
               (self.let is False):
            return(True)
        else:
            return(False)

    def change_dose(self,foo):
        _voi_idx = self.combobox1.get_active()
        self.voilist[_voi_idx].dose = float(self.spinbutton2.get_value())
        _model = self.treeview1.get_model() 
        _iter_str="0:"+str(_voi_idx)  # TODO: is there a better method?
        # TODO: fix NoneType error
        _iter = _model.get_iter_from_string(_iter_str) 
        _model.set(_iter,3,str(self.voilist[_voi_idx].dose))

    def change_prio(self,foo):
        _voi_idx = self.combobox1.get_active()
        self.voilist[_voi_idx].priority = int(self.spinbutton3.get_value())
        _model = self.treeview1.get_model() 
        _iter_str="0:"+str(_voi_idx)  # TODO: is there a better method?
        # TODO: fix NoneType error
        _iter = _model.get_iter_from_string(_iter_str) 
        _model.set(_iter,4,str(self.voilist[_voi_idx].priority))

    def combobox1_change(self,foo):
        _voi_idx = self.combobox1.get_active()
        self.spinbutton2.set_value(self.voilist[_voi_idx].dose)
        self.spinbutton3.set_value(self.voilist[_voi_idx].priority)
        _iter_str="0:"+str(_voi_idx)  # TODO: is there a better method?
        _model = self.treeview1.get_model()
        _iter = _model.get_iter_from_string(_iter_str)
        _path = _model.get_path(_iter)
        self.treeview1.set_cursor(_path)

    def treeview1_select(self, foo):
        _select = self.treeview1.get_cursor()
        # if we are in top level, dont update combobox.
        if len(_select[0])==1:
            return()
        _row = _select[0][1] 
        if _row != None:
            self.combobox1.set_active(_row)
        else:
            self.combobox1.set_active(-1)
        #print "selected VOI:", _row,_select

    def vois2cube(self,vdx,vl):
        """ Returns new data cube based on all vois * dose """

        self.window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
        _sorted_voilist = sorted(vl)        
        # setup target cube:
        target = zeros((self.vdx.header.dimx,
                        self.vdx.header.dimy,
                        self.vdx.header.slice_number),
                       float)

        _mask = zeros((self.vdx.header.dimx,
                       self.vdx.header.dimy,
                       self.vdx.header.slice_number),
                      bool)
        # for each VOI:
        i = 0
        for gtkvoi in _sorted_voilist:
            i += 1
            mystr = " ...calculate VOI %i of %i" %( i , self.vdx.numberofvois)
            self.statusbar1.push(0,mystr)
            while gtk.events_pending():
                gtk.main_iteration()
            if (gtkvoi.dose > 0):
            # lookup current voi (cvoi) from sorted list
            # Really shouldn't be passing arounds vois by index :S
                cvoi = gtkvoi.index
                #print "current voi:", cvoi
                #print "prepare mask of voi number", cvoi, vdx.voi[cvoi].name
                vdx._calc_voi(cvoi)
                _mask = logical_and(vdx.mask[cvoi],logical_not(_mask))
                target += _mask * gtkvoi.dose * 10.0 # multiply with dose
        self.statusbar1.push(0,"Finished building target dose cube.")
        self.window.window.set_cursor(None)
        return(target)
        
    def on_button_resetvoi_clicked(self,foo):
        """ reset all doses to zero """
        _model = self.treeview1.get_model()     
        for i in range(len(self.voilist)):
            _iter_str="0:"+str(i)  # TODO: is tehre a better method?
            _iter = _model.get_iter_from_string(_iter_str)            
            self.voilist[i].dose = 0.0
            self.voilist[i].priority = 10
            _model.set(_iter,3,str(self.voilist[i].dose))
            _model.set(_iter,4,str(self.voilist[i].priority))            
        # update also current spinbuttons
        self.spinbutton2.set_value(0.0)
    
    def on_button_writevoi_clicked(self,foo):
        """ Write the vois to a file. """
        if self.tdos == None:            
            self.on_button_buildcube_clicked(None)
        chooser = gtk.FileChooserDialog(title=None,
                                        action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                        buttons=(gtk.STOCK_CANCEL,
                                                 gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_SAVE,
                                                 gtk.RESPONSE_OK))
        chooser.set_current_name("FOO000001.dos")
	chooser.set_select_multiple(False)
	filter = gtk.FileFilter()
	filter.add_pattern("*.dos")
	chooser.add_filter(filter)
	response = chooser.run()	
	if response == gtk.RESPONSE_OK:
		filename = chooser.get_filename()
		print "Selected", filename
                chooser.destroy()
                #WriteDoseCube(self.tdos, self.vdx.header, filename)
                self.tdos.write(filename)
	elif response == gtk.RESPONSE_CANCEL:
		print "Closed, no files selected"
                chooser.destroy()

    def on_button_buildcube_clicked(self,foo):
        """ Build the target dose cube """
        # tdos is the target dose.
        self.tdos = DoseCube()
        self.tdos.cube = self.vois2cube(self.vdx,self.voilist)
        self.tdos.header = self.vdx.header  # steal header from vdx file.
        self.checkbutton_tdose.set_sensitive(True)

    def on_button_oer_clicked(self,foo):
        _id = 0
        if (self.radiobutton_oer1.get_active()):
            _id = 1
        if (self.radiobutton_oer2.get_active()):
            _id = 2                    
        self.oer = self.let.BuildOER(_id)
        self.checkbutton_oer.set_sensitive(True)
        self.do.append(self.oer) # append oer cube.

    def on_button_test_clicked(self,foo):
        self.populate_tree()

    def plot_text(self):
        #slice infos:
        if self.myaxis == "x":
            print self.cx, self.hed.dimx
            self.cax.text(350, -10, "Slice #: %d/%d"
                         %(self.cx,self.hed.dimx), 
                         fontsize=8, color='white')
            self.cax.text(350, 0, "Slice position: %.2f mm"
                         % (self.hed.bin2pos(self.cx)), 
                         fontsize=8, color='white')
            self.cax.text(350, 320, "Coronal plane",
                         fontsize=8, color='green')
            self.cax.text(0, 150, "V", fontsize=8, color='green') 
            self.cax.text(350, 150, "D", fontsize=8, color='green') 
            self.cax.text(150, 0, "R", fontsize=8, color='green')
            self.cax.text(150, 350, "L", fontsize=8, color='green')

        if self.myaxis == "y":
            self.cax.text(350, -10, "Slice #: %d/%d"
                         %(self.cy,self.hed.dimy), 
                         fontsize=8, color='white')
            self.cax.text(350, 0, "Slice position: %.2f mm"
                         % (self.hed.bin2pos(self.cy)), 
                         fontsize=8, color='white')
            self.cax.text(350, 320, "Sagittal plane",
                         fontsize=8, color='green')
            self.cax.text(0, 150, "V", fontsize=8, color='green') 
            self.cax.text(350, 150, "D", fontsize=8, color='green') 
            self.cax.text(150, 0, "A", fontsize=8, color='green')
            self.cax.text(150, 350, "P", fontsize=8, color='green')


        if self.myaxis == "z":
            self.cax.text(350, -10, "Slice #: %d/%d"
                         %(self.cz,self.hed.slice_number), 
                         fontsize=8, color='white')
            self.cax.text(350, 0, "Slice position: %.2f mm"
                         % (self.hed.bin2slice(self.cz)), 
                         fontsize=8, color='white')
            self.cax.text(350, 320, "Transversal plane",
                         fontsize=8, color='green')
            self.cax.text(0, 150, "L", fontsize=8, color='green') 
            self.cax.text(350, 150, "R", fontsize=8, color='green') 
            self.cax.text(150, 0, "A", fontsize=8, color='green')
            self.cax.text(150, 350, "P", fontsize=8, color='green')

    def foo(self):
        pass

class GTKVoi(object):
    def __init__(self,voi,dose,priority,index,visible=False):    
        self.visible=visible
        self.voi = voi
        self.dose = dose
        self.index = index
        self.priority = priority
    def __cmp__(self, other):
        return cmp(self.priority, other.priority)
    def __str__(self):
        return "VOI: " + self.voi.name + \
            " Visible: " + str(self.visible) + \
            " Dose: " + str(self.dose) + \
            " Priority: " + str(self.priority)        
    def toList(): 
        return [self.visible,self.voi.name,self.dose,self.priority]

def start():
    hwg = PyGtkTRiP()
    gtk.main()

if __name__ == "__main__":
    hwg = PyGtkTRiP()
    gtk.main()




# python
#
# from pytrip import pygtktrip
# hwg = pygtktrip.PyGtkTRiP()
#
