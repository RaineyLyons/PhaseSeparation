# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:18:23 2022

@author: Rainey Lyons (rainey.lyons@kau.se)
This code makes use of a finite volume scheme to simulate the system

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
import matplotlib as mpl
import matplotlib.animation as animation
import time
from numpy.fft  import fft2, ifft2


#%% Parameters 

# Run on rectangle of dimensions [-xLength,xLength] X [-yLength,yLength] with NodeCount**2 number of Nodes.
xLength = 10 
yLength = 10
NodeCount = 100

#Final time
tLength = 3

#Model Parameters 
c0 = 0.8 #Initial solvent concentration
D = 1 #Diffusion constant
B = 2*10 #Drift constants
#Evap = 0.0 #Evaporation Constant !!Currently unused


#%% Initialization

alpha = str(Evap).replace('.','')
Con = str(c0).replace('.','')



Delta_x = 2*xLength/NodeCount
Delta_y = 2*yLength/NodeCount
Delta_t = Delta_x*Delta_y/(4*B*D) #Calculated based on very course CFL
gamma_x = (D * Delta_t) / (Delta_x ** 2)
gamma_y = (D * Delta_t) / (Delta_y ** 2)

x = np.linspace(-xLength,xLength,round(2*xLength/Delta_x)+1)
y = np.linspace(-yLength,yLength,round(2*yLength/Delta_y)+1)
t = np.linspace(0,tLength,round(tLength/Delta_t)+1)

# Initialize solution m(k, i, j) and p(k, i, j)
m = np.zeros((round(np.size(t)/1000)+1, np.size(x), np.size(y)))
p = np.zeros((round(np.size(t)/1000)+1, np.size(x), np.size(y)))

xx, yy = np.meshgrid(x,y)

# Initialize interaction potential J
J = np.zeros((np.size(x),np.size(y)))
for i in range(np.size(x)):
    for j in range(np.size(y)):
        r = np.sqrt(x[i]**2 + y[j]**2)
        if r < 1-Delta_x**4:
            J[i,j] = np.exp(-1/(1-r**2))
            #J[i,j] = (x[i]-y[j])**2
        else:
            J[i,j] = 0.0
Int_J = np.trapz(np.trapz(J,y,axis=0),x,axis=0)
J *= 1/Int_J




##For preloaded initial conditions
# m0 = np.load('Simulations/m_init_equalpm_c0eq08.npy')
# phi0 = np.load('Simulations/phi_init_equalpm_c0eq08.npy')
# m[0,:,:] = m0
# p[0,:,:] = phi0


#For 'well mixed' initial conditions
p0 = 0.5*(1-c0)

for i in range(np.size(x)):
    Sigma = np.random.choice([-1.,0.,1.],np.size(y), replace=True , p = [p0,c0,p0])
    m[0,i,:] = Sigma
    p[0,i,:] = Sigma*Sigma



#%% Used funtions
def grad2D(F,dx,dy):
    #Used to calculate the gradient of J
    #Could be adjusted to calculate the gradient in the finite volume scheme.
    Fx = np.zeros(np.shape(F))
    Fy = np.zeros(np.shape(F))
    for i in range(-1,np.size(x)-1):
        for j in range(-1,np.size(y)-1):
            Fx[i,j] = (1/(2*dx))*(F[i+1,j]-F[i-1,j])
            Fy[i,j] = (1/(2*dy))*(F[i,j+1]-F[i,j-1])
    Fx[np.isinf(Fx)] = 0
    Fy[np.isinf(Fy)] = 0
    return [Fx,Fy]

def fftconvolve2d(x, y):
    f2 = ifft2(fft2(x, s =x.shape) * fft2(y, s =x.shape)).real
    f2 = np.roll(f2, (-((y.shape[0] - 1)//2), -((y.shape[1] - 1)//2)), axis=(0, 1))
    return f2

Jgrad =grad2D(J,Delta_x,Delta_y)


def Calculate(m_old,p_old,x,y,t,flag):
    #Calculates the finite volume scheme and saves every 1000 time steps
    m_new = np.zeros(np.shape(m_old))
    p_new = np.zeros(np.shape(p_old))
    
    #initialize convolution and velocities
    conv_Mat1 = Delta_x*Delta_y*fftconvolve2d(Jgrad[0], m_old)
    conv_Mat2 = Delta_x*Delta_y*fftconvolve2d(Jgrad[1], m_old)
        
    v_m = np.array([(p_old-m_old*m_old) * conv_Mat1[:,:] , (p_old-m_old**2) * conv_Mat2[:,:]])
    v_p = np.array([m_old*(1-p_old) * conv_Mat1[:,:],m_old*(1-p_old) * conv_Mat2[:,:]])
    
    
    #Solve m and phi on interior of domain
    Diffusion_m1 = gamma_x * (m_old[2:,1:-1] + m_old[:-2, 1:-1] -2*m_old[1:-1 ,1:-1]) + gamma_y * (m_old[1:-1,2:] + m_old[ 1:-1,:-2] -2*m_old[1:-1 ,1:-1])
    Advection_m1 =  Delta_t /(2*Delta_x) *B*(v_m[0,2:,1:-1] - v_m[0,:-2,1:-1]) + Delta_t /(2*Delta_y)*B*(v_m[1,1:-1,2:]-v_m[1,1:-1,:-2])
    m_new[1:-1,1:-1] = m_old[1:-1,1:-1] + Diffusion_m1 - Advection_m1
    
    Diffusion_p1 = gamma_x * (p_old[2:,1:-1] + p_old[:-2, 1:-1] -2*p_old[1:-1 ,1:-1]) + gamma_y * (p_old[1:-1,2:] + p_old[ 1:-1,:-2] -2*p_old[1:-1 ,1:-1])
    Advection_p1 =  Delta_t /(2*Delta_x) *B*(v_p[0,2:,1:-1] - v_p[0,:-2,1:-1]) + Delta_t /(2*Delta_y)*B*(v_p[1,1:-1,2:]-v_p[1,1:-1,:-2])
    p_new[1:-1,1:-1] = p_old[1:-1,1:-1] + Diffusion_p1 - Advection_p1
    
    
   
    #Periodic boundary conditions
    m_new[1:-1,-1] = gamma_x * (m_old[2:,-1] + m_old[:-2, -1] -2*m_old[1:-1 ,-1]) + gamma_y * (m_old[1:-1,0] + m_old[ 1:-1,-2] -2*m_old[1:-1 ,-1]) - Delta_t /(2*Delta_x) *B*(v_m[0,2:,-1] - v_m[0,:-2,-1]) + Delta_t /(2*Delta_y)*B*(v_m[1,1:-1,0]-v_m[1,1:-1,-2])
    m_new[1:-1,0] = gamma_x * (m_old[2:,0] + m_old[:-2, 0] -2*m_old[1:-1 ,0]) + gamma_y * (m_old[1:-1,1] + m_old[ 1:-1,-1] -2*m_old[1:-1 ,0]) - Delta_t /(2*Delta_x) *B*(v_m[0,2:,0] - v_m[0,:-2,0]) + Delta_t /(2*Delta_y)*B*(v_m[1,1:-1,1]-v_m[1,1:-1,-1])
    m_new[0,1:-1] = gamma_x * (m_old[1,1:-1] + m_old[ -1,1:-1] -2*m_old[0,1:-1]) + gamma_y * (m_old[0,2:] + m_old[0,:-2] -2*m_old[0,1:-1]) - Delta_t /(2*Delta_x) *B*(v_m[0,0,2:] - v_m[0,0,:-2]) + Delta_t /(2*Delta_y)*B*(v_m[1,1,1:-1]-v_m[1,-1,1:-1])
    m_new[-1,1:-1] =gamma_x * (m_old[-1,2:] + m_old[ -1,:-2,] -2*m_old[-1,1:-1]) + gamma_y * (m_old[0,1:-1] + m_old[-2, 1:-1] -2*m_old[-1,1:-1]) - Delta_t /(2*Delta_x) *B*(v_m[0,-1,2:] - v_m[0,-1,:-2]) + Delta_t /(2*Delta_y)*B*(v_m[1,0,1:-1]-v_m[1,-2,1:-1])
    
    p_new[1:-1,-1] = gamma_x * (p_old[2:,-1] + p_old[:-2, -1] -2*p_old[1:-1 ,-1]) + gamma_y * (p_old[1:-1,0] + p_old[ 1:-1,-2] -2*p_old[1:-1 ,-1]) - Delta_t /(2*Delta_x) *B*(v_p[0,2:,-1] - v_p[0,:-2,-1]) + Delta_t /(2*Delta_y)*B*(v_p[1,1:-1,0]-v_p[1,1:-1,-2])
    p_new[1:-1,0] = gamma_x * (p_old[2:,0] + p_old[:-2, 0] -2*p_old[1:-1 ,0]) + gamma_y * (p_old[1:-1,1] + p_old[ 1:-1,-1] -2*p_old[1:-1 ,0]) - Delta_t /(2*Delta_x) *B*(v_p[0,2:,0] - v_p[0,:-2,0]) + Delta_t /(2*Delta_y)*B*(v_p[1,1:-1,1]-v_p[1,1:-1,-1])
    p_new[0,1:-1] = gamma_x * (p_old[1,1:-1] + p_old[ -1,1:-1] -2*p_old[0,1:-1]) + gamma_y * (p_old[0,2:] + p_old[0,:-2] -2*p_old[0,1:-1]) - Delta_t /(2*Delta_x) *B*(v_p[0,0,2:] - v_p[0,0,:-2]) + Delta_t /(2*Delta_y)*B*(v_p[1,1,1:-1]-v_p[1,-1,1:-1])
    p_new[-1,1:-1] =gamma_x * (p_old[-1,2:] + p_old[ -1,:-2,] -2*p_old[-1,1:-1]) + gamma_y * (p_old[0,1:-1] + p_old[-2, 1:-1] -2*p_old[-1,1:-1]) - Delta_t /(2*Delta_x) *B*(v_p[0,-1,2:] - v_p[0,-1,:-2]) + Delta_t /(2*Delta_y)*B*(v_p[1,0,1:-1]-v_p[1,-2,1:-1])

    # for i in range(-1,np.size(x)-1):
    #     for j in range(-1, np.size(y)-1):
    #         Diffusion_m = gamma_x * (m_old[i+1][j] + m_old[i-1][j]  - 2*m_old[i][j]) + gamma_y *(m_old[i][j+1] + m_old[i][j-1] - 2*m_old[i][j])
    #         Diffusion_p = gamma_x * (p_old[i+1][j] + p_old[i-1][j]  - 2*p_old[i][j]) + gamma_y *(p_old[i][j+1] + p_old[i][j-1] - 2*p_old[i][j])
    #         Advection_m = Delta_t /(2*Delta_x) *B*(v_m[0,i+1,j] - v_m[0,i-1,j]) + Delta_t /(2*Delta_y)*B*(v_m[1,i,j+1]-v_m[1,i,j-1])
    #         Advection_p = Delta_t /(2*Delta_x) *B*(v_p[0,i+1,j] - v_p[0,i-1,j]) + Delta_t /(2*Delta_y)*B*(v_p[1,i,j+1]-v_p[1,i,j-1])
    #         Evap_Term = Delta_t*Evap*(1-p_old[i][j])
                
    #         m_new[i, j] =  m_old[i][j] + Diffusion_m - Advection_m
    #         p_new[i,j] = p_old[i,j] + Diffusion_p -Advection_p + Evap_Term
    
    if not flag%1000:
        m[round(flag/1000)] = m_new
        p[round(flag/1000)] = p_new
    

    
    if CalcSolventRatio(p_new) == 0.4:
        np.save(f"m_evap_40_A{alpha}.npy",m_new)
        np.save(f"p_evap_40_A{alpha}.npy",p_new) 

  
    return m_new,p_new

def CalcSolventRatio(phi):
	 Ratio = np.sum(np.abs(1-phi))*(1/512)**2
	 return Ratio


#%%Call
m_prev, p_prev = m[0],p[0]
start_time = time.time()

for k in range(np.size(t)):
    m_prev, p_prev = Calculate(m_prev, p_prev, x, y, t, k)
tf = time.time()

print("--- %s mins ---" % np.round_((tf-start_time)/60,3))    
np.save(f"m_evap{alpha}_{NodeCount}T{tLength}_c0eq{Con}.npy",m)
np.save(f"phi_evap{alpha}_{NodeCount}T{tLength}_c0eq{Con}.npy",p)    



