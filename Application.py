'''This file is responsible for the final calculation of sputtering yields and thickness of the erosion layer. 
It uses the functions defined in SputteringYieldFunctions after reading the data from the archieveDB'''

import os
import w7xarchive
import itertools
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate as integrate

import src.SputteringYieldFunctions as calc
import src.ProcessData as process
import src.ReadArchieveDB as read
from src.PositionsLangmuirProbes import OP2_TM2Distances, OP2_TM3Distances, OP2_TM8Distances

#avoids pop up of plot windows
matplotlib.use('agg')

#######################################################################################################################################################################
#######################################################################################################################################################################
#initialize common parameter values
e  =  scipy.constants.elementary_charge 
u = scipy.constants.u   #to convert M in [u] to m in [kg]: M * u = m
k_B = scipy.constants.Boltzmann
k = k_B/e 

#ion masses in [kg] and ion concentrations (no unit) for [H, D, T, C, O]
ions = ['H', 'D', 'T', 'C', 'O']
m_i = np.array([1.00794, 2.01210175, 3.0160495, 12.011, 15.9994]) * u 
f_i = [0.89, 0, 0, 0.04, 0.01]

#lines for [Be, C, Fe, Mo, W] and columns with [H, D, T, He, Self-Sputtering, O] (O only known for C), in [eV] according to Ref. 1
E_TF = np.array([[256, 282, 308, 720, 2208, 0],
                 [415, 447, 479, 1087, 5688, 9298],
                 [2544, 2590, 2635, 5517, 174122, 0], 
                 [4719, 4768, 4817, 9945, 533127, 0],
                 [9871, 9925, 9978, 20376, 1998893, 0]]) 

#[Be, C, Fe, Mo, W], in [eV] according to Ref. 1
E_s = np.array([3.38, 7.42, 4.34, 6.83, 8.68])

#Parameters for chemical erosion of C by H-isotopes, [H, D, T] according to Ref. 1
Q_y_chem = np.array([0.035, 0.1, 0.12])                                                                  
C_d_chem = np.array([250, 125, 83])                                                                               
E_thd_chem = [15, 15, 15]   #threshold energy for Y_damage                                                                      
E_ths_chem = [2, 1, 1]      #threshold energy for Y_surf                                                                        
E_th_chem=[31, 27, 29]   

#target density in [1/m^3]
n_target = 11.3*1e28 #nonsense value?

#Parameters for net erosion specifically for divertor
lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct


#######################################################################################################################################################################
#incident angle in rad
alpha = 2 * np.pi/9

#dischargeID
discharge = '20250429.048'   #20241127.034'

#######################################################################################################################################################################
#######################################################################################################################################################################
#Markus data read in, "intervals" defines how many adjacent values are averaged
#processMarkusData(m_i, f_i, ions, k, n_target, interval = 50)

#######################################################################################################################################################################
if __name__ == '__main__':
    #_lower indicates lower divertor unit, _upper upper divertor unit

    #read langmuir probe positions in [m] from pumping gap
    #index 0 - 5 are langmuir probes on TM2h, 6 - 13 on TM3h, 14 - 17 on TM8h (distance from pumping gap is increasing)
    LP_position = [OP2_TM2Distances]
    LP_position.append(OP2_TM3Distances)
    LP_position.append(OP2_TM8Distances)
    LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list

    #read Langmuir Probe data from xdrive
    #all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
    #ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s]
    ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper = read.readLangmuirProbeDataFromXdrive(discharge)

    #all Langmuir Probes measure at the same times but some will stop earlier than others 
    #-> time_array holds all times that are represented in one discharge at any LP
    time_array = []
    for t_divertorUnit in [t_lower, t_upper]:
        for t_position in t_divertorUnit:
            for t in t_position:
                if t not in time_array:
                    time_array.append(t)
    time_array.sort()
    #print(time_array)
    
    #read IRcam data for both divertor units
    #len(t_upper) represents the number of active LPs per divertor unit (14 for EIM, 4/18 for FTM?)
    #Ts in [K], LP_position in [m] from pumping gap, time_array in [s]
    Ts_lower, Ts_upper = read.readSurfaceTemperatureFramesFromIRcam(discharge, time_array, ['lower', 'upper'], LP_position, len(t_upper))
    
    #missing measurement times and values are replaced by adding arrays of 0 to guarantee same data structure of all arrays concerning t, Te, Ts, ne
    for j in range(len(ne_lower)):
        for i in range(len(time_array) - len(ne_lower[j])):
            ne_lower[j].append(0)
            Te_lower[j].append(0)
            t_lower[j].append(0)
    for j in range(len(ne_upper)):
        for i in range(len(time_array) - len(ne_upper[j])):
            ne_upper[j].append(0)
            Te_upper[j].append(0)
            t_upper[j].append(0)

    Te_lower = np.array(Te_lower)
    Te_upper = np.array(Te_upper)
    ne_lower = np.array(ne_lower)
    ne_upper = np.array(ne_upper)
    t_lower = np.array(t_lower)
    t_upper = np.array(t_upper)
    #print(np.shape(Ts_lower), np.shape(Ts_upper), np.shape(ne_upper), np.shape(ne_lower), np.shape(Te_upper), np.shape(Te_lower))

    #calculate sputtering related physical quantities (sputtering yields, erosion rates, layer thicknesses)
    #all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
    #ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s], Ts in [K], alpha in [rad], LP_position in [m] from pumping gap, m in [kg], k in [eV/K], n_target in [1/m^3]
    #does not return something but writes measurement values and calculated values to 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
    #plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
    process.processOP2Data(discharge, ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, True)

#LF will be replaced by CRLF the next time Git touches it
#######################################################################################################################################################################
#read data from archieveDB

#######################################################################################################################################################################
#######################################################################################################################################################################
#references for look-up values
# Ref.1: D. Naujoks. Plasma-Material Interaction in Controlled Fusion (Vol. 39 in Springer Series on Atomic, Optical, and Plasma Physics). Ed. by G. W. F. Drake, Dr. G. Ecker, and Dr. H. Kleinpoppen. Springer-Verlag Berlin Heidelberg, 2006. 
