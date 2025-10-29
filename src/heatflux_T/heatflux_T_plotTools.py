# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:02:45 2021

@author: ygao
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
from os.path import join

def findvminvmax(array, threshold = 0.02):
    '''
    to find the vmin, vmax for the colorbar range of the array.
    @array: could be 1D or 2D (image).
    @threshold: 0.02.
    '''
    vmin = np.nanpercentile(array, threshold * 100)
    vmax = np.nanpercentile(array, (1 - threshold) * 100)
    #vmin, vmax = np.sort(array.flatten())[int(array.size * threshold)], np.sort(array.flatten())[int(array.size * (1 - threshold))]
    return vmin, vmax

def plot_Frame (tt1, tt2, data, title, T = False, savefolder = None, \
                colorbar = None, xlim = None, ylim = None, logscale = False, s = 1, aspect = 100):
    
    tt1, tt2, data = np.hstack(tt1), np.hstack(tt2), np.hstack(data)
    FontSize = 25
    matplotlib.rcParams.update( { 'font.size': FontSize } )   
    plt.rcParams["figure.figsize"] = (19,8)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title)
    #plt.gca().set_aspect('equal')
    ax.set_xlabel('Toroidal angle [deg]')
    ax.set_ylabel('Length [m]')
    ax.set_title(title)      
    if colorbar is None:
        colorbarmin, colorbarmax = findvminvmax(data)
        if logscale:
            scat = ax.scatter(tt1, tt2, c = data.clip(colorbarmin), s=s, cmap = plt.get_cmap( 'jet' ), \
                              norm=matplotlib.colors.LogNorm())
        else:
            scat = ax.scatter(tt1, tt2, c = data, s=s,  cmap = plt.get_cmap( 'jet' ))   

    else:
        colorbarmin, colorbarmax = colorbar
        if logscale:
            scat = ax.scatter(tt1, tt2, c = data.clip(colorbarmin), s=s,  cmap = plt.get_cmap( 'jet' ), \
                              norm=matplotlib.colors.LogNorm(vmin=colorbarmin, vmax=colorbarmax))  
        else:
            scat = ax.scatter(tt1, tt2, c = data, s=s,  cmap = plt.get_cmap( 'jet' ), \
                            vmin = colorbarmin, vmax = colorbarmax)  

    if not T:
        savestr = 'H'
        fig.colorbar(scat, orientation = 'vertical', aspect=aspect, fraction=0.023).set_label(r'Heat Flux [MW/m$^2$]')
        #fig.colorbar(scat, orientation = 'vertical', aspect=50, fraction=0.023).set_label(r'Connection Length [m]')

    else:
        savestr = 'T'
        fig.colorbar(scat, orientation = 'vertical', aspect=aspect, fraction=0.023).set_label(r'Temperature [$^o$C]') 
        
    plt.grid(True)
    if xlim != None:
        plt.xlim(xlim[0], xlim[1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    if savefolder is not None:
        titlesp = title.split('_')
        subfoldername = titlesp[0] + '_' + titlesp[1] + '_' + savestr # PID_H or T
        subfolder = join(savefolder, subfoldername)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        fig.savefig(join(subfolder, title + '.png'))
        plt.close(fig)   
        
def plot_Finger (ft1, ft2, data, title, T = False, savefolder = None, \
                colorbar = None, xlim = None, ylim = None):
    
    ft1, ft2, data = np.hstack(ft1) * 100, np.hstack(ft2) * 100, np.hstack(data)
    FontSize = 25
    matplotlib.rcParams.update( { 'font.size': FontSize } )   
    plt.rcParams["figure.figsize"] = (19,8)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title)
    #plt.gca().set_aspect('equal')
    ax.set_xlabel('Location [cm]')
    ax.set_ylabel('Location [cm]')
    ax.set_title(title)      
    if colorbar is None:
        colorbarmin, colorbarmax = findvminvmax(data)
        scat = ax.scatter(ft1, ft2, c = data, s=1,  cmap = plt.get_cmap( 'jet' ))   
    else:
        colorbarmin, colorbarmax = colorbar
        scat = ax.scatter(ft1, ft2, c = data, s=1,  cmap = plt.get_cmap( 'jet' ), \
                        vmin = colorbarmin, vmax = colorbarmax)  
    if not T:
        savestr = 'H'
        fig.colorbar(scat, orientation = 'vertical', aspect=100, fraction=0.023).set_label(r'Heat Flux [MW/m$^2$]')
    else:
        savestr = 'T'
        fig.colorbar(scat, orientation = 'vertical', aspect=100, fraction=0.023).set_label(r'Temperature [$^o$C]') 
        
    plt.grid(True)
    if xlim != None:
        plt.xlim(xlim[0], xlim[1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    if savefolder is not None:
        titlesp = title.split('_')
        subfoldername = titlesp[0] + '_' + titlesp[1] + '_' + savestr
        subfolder = join(savefolder, subfoldername)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        fig.savefig(join(subfolder, title + '.png'))
        plt.close(fig)  
        
def plot_Profile (S, data, title, T = False, savefolder = None, xlim = None, ylim = None):
    FontSize = 18
    matplotlib.rcParams.update( { 'font.size': FontSize } ) 
    plt.rcParams["figure.figsize"] = (9,6)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title)
    ax.set_xlabel('Location [m]')
    if not T:
        savestr = 'H'
        ax.set_ylabel(r'Heat Flux [MW/m$^2$]')
    else:
        savestr = 'T'
        ax.set_ylabel(r'Temperature [$^o$C]')
    ax.set_title(title)      
    
    if data[0].size == 1:
        # single profile
        ax.plot(S, data, 'o', label = 'averaged')
    else:
        for index, dat in enumerate(data):
            # since the profiles are sorted, we get the lineNO directly from indices
            ax.plot(S[index], dat, label = str(index))
            
    ax.legend()        
    plt.grid(True)
    if xlim != None:
        plt.xlim(xlim[0], xlim[1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    if savefolder is not None:
        titlesp = title.split('_')
        subfoldername = titlesp[0] + '_' + titlesp[1] + '_' + savestr
        subfolder = join(savefolder, subfoldername)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        fig.savefig(join(subfolder, title + '.png'))
        plt.close(fig)   