'''This file is responsible for the final calculation of sputtering yields and thickness of the erosion layer. 
It uses the functions defined in SputteringYieldFunctions after reading the data from the archieveDB'''

import w7xarchive
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate as integrate

import src.SputteringYieldFunctions as calc
#import src.ReadArchieveDB as read

#######################################################################################################################################################################
#initialize common parameter values
e  =  scipy.constants.elementary_charge 
k_B = scipy.constants.Boltzmann
k = k_B/e 

#ion masses in [kg] and ion concentrations (no unit) for [H, D, T, C, O]
ions = ['H', 'D', 'T', 'C', 'O']
m_i = [1, 1, 1, 1, 1] #nonsense values, look them up
f_i = [0.89, 0, 0, 0.4, 0.1]

#lines for [Be, C, Fe, Mo, W] and columns with [H, D, T, He, Self-Sputtering, O] (O only known for C), in [eV]
E_TF = np.array([[256, 282, 308, 720, 2208, 0],
                 [415, 447, 479, 1087, 5688, 9298],
                 [2544, 2590, 2635, 5517, 174122, 0], 
                 [4719, 4768, 4817, 9945, 533127, 0],
                 [9871, 9925, 9978, 20376, 1998893, 0]]) 

#[Be, C, Fe, Mo, W], wo kommen die her, Einheit?! -> muss [eV] stimmt aber glaub ich auch   
E_s = np.array([3.38, 7.42, 4.34, 6.83, 8.68])

#Parameters for chemical erosion of C by H-isotopes, [H, D, T] 
Q_y_chem = np.array([0.035, 0.1, 0.12])                                                                  
C_d_chem = np.array([250, 125, 83])                                                                               
E_thd_chem = [15, 15, 15]   #threshold energy for Y_damage                                                                      
E_ths_chem = [2, 1, 1]      #threshold energy for Y_surf                                                                        
E_th_chem=[31, 27, 29]   

#incident angle in rad
alpha = 2 * np.pi/6

#target density in [1/m^3]
n_target = 2.5 #nonsense value

#Parameters for net erosion specifically for divertor
lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct

#read from data
T_s = np.array([600] * 3)
n_e = np.array([1] * 3)
T_e = np.array([1] * 3)
T_i = T_e
dt = np.array([1] * 3)

if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
    #calculate fluxes for all ion species [H, D, T, C, O] at each single time
    fluxes = []
    for m, f in zip(m_i, f_i):
        fluxes.append(calc.calculateFluxIncidentIon(T_e, T_i, m, n_e, f))
    fluxes = np.array(fluxes)

    #calculate sputtering yields [H, D, T, C, O] at each single time
    Y_i = []
    for flux, ion in zip(fluxes, ions):
        Y_i_single = []
        for i in range(len(flux)): 
            if flux[i] != 0:
                Y_i_single.append(calc.calculateTotalErosionYield(ion, T_i[i], 'C', alpha, T_s[i], flux[i]))
            else:
                Y_i_single.append(0)
        Y_i.append(Y_i_single)
    Y_i = np.array(Y_i)
    print(Y_i)

    #calculate erosion rate caused by all ion species at each single time 
    erosionRate_i = []
    for Y_dt, flux in zip(Y_i.T, fluxes.T):
        erosionRate_i.append(calc.calculateErosionRate(Y_dt, flux, n_target))
    print(erosionRate_i)

    #calculate eroded layer thickness over the whole discharge
    erodedLayerThickness_dt = []
    for Y_dt, flux, dt_step in zip(Y_i.T, fluxes.T, dt):
        erodedLayerThickness_dt.append(calc.calculateDeltaErodedLayer(Y_dt, flux, dt_step, n_target))
    print(erodedLayerThickness_dt)
    
    erodedLayerThickness = sum(erodedLayerThickness_dt)
    print(erodedLayerThickness)