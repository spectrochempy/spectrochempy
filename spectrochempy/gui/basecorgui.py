# -*- coding: utf-8 -*-
#
# ===============================================================================
# spectrochempy.processing.basecorgui
# ===============================================================================
# Copyright (C) 2015-2017
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# This software is governed by the CeCILL-B license under French law
# and abiding by the rules of distribution of free software.
# You can  use, modify and/ or redistribute the software under
# the terms of the CeCILL-B license as circulated by CEA, CNRS and INRIA
# at the following URL "http://www.cecill.info".
# See Licence.txt in the main spectrochempy source directory
#
# author : A.Travert
# ===============================================================================

import tkinter as tk

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib.widgets import SpanSelector

# local import

from ..core.processors.baseline.basecor import basecor
from ..core.api import AxisRange
               
def basecorgui(X, display='all'):
    """graphical interface for dataset baseline correction.
    
    opens one tk window for method parameters, 
    one matplot figure with:
    
    * Subplot for graphical baseline selection, to select range r (global) 
    * Subplot to display the result
    
    if display = 'all': all spextra are shown,
                  n : n equally spaced spectra
    """
    
    class BlGui(tk.Tk):
        ''' The tkinter GUI is defined as a Class with following attributes:
            BlGui.out
            '''
        def __init__(self, X, parent):
            tk.Tk.__init__(self,parent)
            self.parent = parent
            self.initialize(X)
            
        def computebaseline(self, X, r):
            """
            This function is launched when the 'compute' button is pressed:
                - gets the parameters of the baseline correction method,  
                - launch Dataset.basecor() with these parameters 
                - plots the result in subplot 2
            """
                        
            # gets the parameters of the baseline correction method from the listboxes
            meth = self.method.get(self.method.curselection()).lower()   
            interp= self.interpolation.get(self.interpolation.curselection()).lower()
            deg = int(self.order.get(self.order.curselection())) 
            npc = int(self.npc.get()) 
            
 
            # generates baseline corrected dataset 
            self.out = basecor(X, r, meth, interp, deg, npc)
            
            # plots the result in subplot 2            
      
            ax2 = plt.subplot(212) #activate subplot 2        
            plt.cla()
            xaxis = self.out.dims[1].axisset.axes[0]   
            
            plt.plot(xaxis, self.out.data.transpose())
            plt.xlabel(self.out.dims[1].axisset.axes[0].name)
            plt.ylabel(self.out.datalabel)        
            ax2.set_title('Corrected spectra')        
                            
        def closebl(self):
            """
            This function is launched when the 'exit' button is pressed and close all windows
            """
            plt.close(self.plot)
            self.destroy()
                      
        def initialize(self, X):

            self.out = X    # initialization of baseline corrected dataset
            self.r = []     # baseline range
            self.p = []     # corresponding rectangles   
            self.plot = None    # plot of initial and corrected spcs
                          
            self.wm_title("BASELINE - Correction Parameters")
                                       
            l1 = tk.Label(self, text="Correction Method:")
            l1.pack()            
            self.method = tk.Listbox(self, height=2, exportselection=0)
            self.method.pack()         
            for choice in ['sequential', 'multivariate']:
                self.method.insert('end', choice)        
            self.method.select_set(0)    # sets sequential as default
             
            l2 = tk.Label(self, text="Interpolation type:")
            l2.pack()            
            self.interpolation = tk.Listbox(self, height=2, exportselection=0)
            self.interpolation.pack()  
            for choice in ["Polynomial", "Pchip"]:
               self.interpolation.insert('end', choice)        
            self.interpolation.select_set(0)
                
            w3 = tk.Label(self, text="Polynome order:")
            w3.pack()            
            self.order = tk.Listbox(self, height=4, exportselection=0)
            self.order.pack()          
            for choice in ["0", "1", "2", "3"]:
                self.order.insert('end', choice)            
            self.order.select_set(0)    
            
            e4 = tk.Label(self, text="Number of Components:")
            e4.pack()  
            self.npc = tk.Entry()
            self.npc.insert(0, '1')
            self.npc.pack()
            
            b1 = tk.Button(self, text="Correct spectra", command = lambda: self.computebaseline(X, self.r))
            b1.pack()
            
            b2 = tk.Button(self, text="Save & Exit", command = lambda: self.closebl())
            b2.pack()
            
            # Now creates figure with initial & corrected spectra
            
            class XrangeSelector(SpanSelector):
                """
                SpanSelector that is only activated when pan/zoom is off, and left button is pressed
                """
                def ignore(self,event):
                    if 'zoom' in Gcf.get_active().toolbar.mode: return True
                    if event.button == 2 or event.button == 3: return True
            
            def onselect(xmin, xmax):  #add the selected range to the baseline range 
            
                self.r.append((xmin,xmax))
                self.r = axisrange.cleanranges(self.r)                    
                
                #activate figure 1
                plt.subplot(211)
                # update rectangles 
                for patch in [patch for patch in ax1.patches if type(patch)==mpl.patches.Polygon]:
                    patch.remove()
                self.p = []
                for x in self.r:                         #plot the new ones
                  self.p.append(plt.axvspan(x[0], x[1], ymin=0, ymax=1, facecolor='g', alpha=0.3))
                  
            def on_press(event):     #removes a range on right click 
                
                if event.button == 3:
                    for xrange in self.r:
                        if (event.xdata >= xrange[0]) and (event.xdata <= xrange[1]):
                            self.r.remove(xrange)
                            #activate figure 1
                            plt.subplot(211) #activate subplot 1  
                            for patch in [patch for patch in ax1.patches if type(patch) == mpl.patches.Polygon]:
                                patch.remove()
                            if not self.r:
                                plt.figure(1).draw_idle()
                            self.p = []
                            plt.subplot(211)
                            for x in self.r:                         #plot the new ones
                               self.p.append(plt.axvspan(x[0], x[1], ymin=0, ymax=1, facecolor='g', alpha=0.3))                
                
            #default baseline selection lower and upper frac% of axis
            
            frac = 0.05      
            xmin = X.dims[1].axes[0].values[0]
            xmax = X.dims[1].axes[0].values[len(X.dims[1].axes[0])-1]
    
            if (xmin > xmax):
                xmin, xmax = xmax, xmin        
            xmin1 = xmin
            xmax1 = xmin + frac * (xmax - xmin)
            xmin2 = xmax - frac * (xmax - xmin)
            xmax2 = xmax
            self.r.append((xmin1,xmax1))
            self.r.append((xmin2, xmax2))
            self.r.sort()  
            
            # first subplot == baseline selection 
            self.plot =  plt.figure(1)               
            plt.subplot(211)
            xaxis = X.dims[1].axes[0].values                
            plt.plot(xaxis, X.data.transpose())
            if xaxis[0] > xaxis[-1]:
                plt.gca().invert_xaxis()
            plt.ylabel(X.datalabel)
             
            plt.figure(1).canvas.set_window_title('BASELINE - Point selection')
            ax1 = plt.subplot(211)
            ax1.set_title('Baseline select: press left & drag to select / right click to delete')
            
            for x in self.r:     #plots default ranges
                self.p.append(plt.axvspan(x[0], x[1], ymin=0, ymax=1, facecolor='g', alpha=0.3))
                
            XrangeSelector(ax1, onselect, 'horizontal')
            plt.figure(1).canvas.mpl_connect('button_press_event', on_press)
             
            # second subplot == baseline corrected plot (default) 
            ax2 = plt.subplot(212)
            xaxis = X.dims[1].axes[0].values 
            
            plt.plot(xaxis, X.data.transpose())
            if xaxis[0] > xaxis[-1]:
                plt.gca().invert_xaxis()
            plt.xlabel(X.dims[1].axes[0].name)
            plt.ylabel(X.datalabel)
            ax2.set_title('Corrected spectra')
            plt.show()
            self.plot = plt.figure(1)
                                 
    #Launch GUI
    mygui = BlGui(X, None)                      

    # mygui.iconbitmap('icon.ico')
    mygui.mainloop()
    
    return mygui.out
