# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:26:41 2022

@author: Rainey Lyons (rainey.lyons@kau.se)

Functions which calculate the Structure Factor and Correlation Function of morphologies at a fixed time generated via the pde

m_t = D∆m - 2B div((p-m**2)∇J*m)
p_t = D∆p - 2B div(m(1-p)∇J*m) + Evap * (1-p)

with periodic boundary conditions.

This system is designed to represent a mixture of 2 solute particles and 1 solvent particle which, with physically meaningful initial data, will undergo phase separation and
produce morphologies.

Note: This code was made to understand and visualize solutions to the above system. This code was not designed to be efficient in any sense and many improvements can be made. 
If you have any suggestions, please feel free to email me.

For more information on the system above, please see the following references:
    R. Marra and M. Mourragui. Phase segregation dynamics for the Blume–Capel model with Kac interaction. Stochastic Processes and their Applications, 88(1):79–124, 2000.
    
    R. Lyons, S. A. Muntean, E. N. Cirillo, A. Muntean. A Continuum Model for Morphology Formation from Interacting Ternary Mixtures: 
        Simulation Study of the Formation and Growth of Patterns. arXiv preprint arXiv:2212.12447, 2022.
        
    R. Lyons, E. N. Cirillo, A. Muntean. Phase separation and morphology formation in interacting ternary mixtures under evaporation -- 
        Well-posedness and numerical simulation of a non-local evolution system. arXiv preprint arXiv:2303.13981, 2022. 

"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft  import fft2, ifft2, fftshift
import scipy.optimize as opt





def StructureFactor(mat):
    """  
    Parameters
    ----------
    mat :   xlen by ylen np array
            Solution at time t. In practice will be m, p, or (1-p).

     Rx, Ry
    -------
    Computes the strucutre factors from fft2.

    """
    xlen, ylen = np.shape(mat)
    CorFunct =  (1/(xlen*ylen))*np.abs(fftshift(fft2(mat)))**2
    K_x = np.linspace(-np.pi , np.pi, xlen)
    K_y = np.linspace(-np.pi , np.pi, ylen)
    Rx = np.sum(CorFunct)/np.sum(np.abs(K_x)*np.sum(CorFunct,axis = 1))
    Ry = np.sum(CorFunct)/np.sum(np.abs(K_y)*np.sum(CorFunct,axis = 0))
    return Rx,Ry

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def find_first_nearest(array,value=0):
    array = np.asarray(array)-value
    
    for i in range(1,np.size(array)):
        if array[i]>=0:
            idx = i
        else:
            break
        
    return array[idx],idx
            

def CorrelationFunction(mat,cut_off = 0.0):
    """
    Parameters
    ----------
    mat : xlen by ylen np array
          Solution at time t. In practice will be m, p, or (1-p).
         
    cut_off: float
            cut off value for correlation function. Initialized at 0.0.

    Gx, Gy, rootx, rooty
    -------
    Calculates the domain size based on the zeros of the correlation function.

    """
    xlen, ylen = np.shape(mat)
    G_cor = lambda sx, sy: (1/xlen*ylen)*np.sum(mat*np.roll(np.roll(mat,sx,axis=0),sy,axis =1))
    
    Gx = np.array([G_cor(xn,0) for xn in range(xlen)])
    Gy = np.array([G_cor(0,yn) for yn in range(ylen)])
    Gx = (1/Gx[0])*Gx
    Gy = (1/Gy[0])*Gy
    
    # val, BestIndex = find_nearest(np.abs(Gx[0:round(xlen/3)] -cut_off),0)
    val, BestIndex = find_first_nearest(Gx -cut_off,0)
    LinGX = lambda x: np.interp(x,np.linspace(0,xlen,xlen+1)[:-1],Gx-cut_off)
    try:
        rootx = opt.brentq(LinGX,BestIndex-2,BestIndex+2)
    except:
        plt.plot(Gx)
        
    # val, BestIndey = find_nearest(np.abs(Gy[0:round(ylen/3)+1] -cut_off),0)
    val, BestIndey = find_first_nearest(Gy -cut_off,0)
    LinGy = lambda y: np.interp(y,np.linspace(0,ylen,ylen+1)[:-1],Gy-cut_off)
    try:
        rooty = opt.brentq(LinGy,BestIndey-3,BestIndey+3)
    except:
        rooty = 0
        
    return Gx, Gy, rootx, rooty

