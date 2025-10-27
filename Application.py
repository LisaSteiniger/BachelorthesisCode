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
import src.CompareResults as compare
from src.ReadArchieveDB import readMarkusData
from src.ReadArchieveDB import readLangmuirProbeDataFromXdrive

#avoids pop up of plot windows
matplotlib.use('agg')

#######################################################################################################################################################################
def processMarkusData(m_i, f_i, ions, k, n_target, interval = 50): #average Intervall = 50 subsequent values of the value arrays below
    settings_dict_list = [{'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},

                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'}]#,

                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'},
                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'},
                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'}]

    settings_Gao_list = [{'exp': '20180807', 'discharge': '014', 'divertor' : '3lh', 'finger' : '11'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '3lh', 'finger' : '12'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '3lh', 'finger' : '13'},

                        {'exp': '20180807', 'discharge': '014', 'divertor' : '5uh', 'finger' : '11'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '5uh', 'finger' : '12'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '5uh', 'finger' : '13'}]#,

                        #{'exp': '20180814', 'discharge': '006', 'divertor' : '5uh', 'finger' : '11'},
                        #{'exp': '20180814', 'discharge': '006', 'divertor' : '5uh', 'finger' : '12'},
                        #{'exp': '20180814', 'discharge': '006', 'divertor' : '5uh', 'finger' : '13'}]

    for settings_dict, settings_Gao in zip(settings_dict_list, settings_Gao_list):
        ne_values, Te_values, positions, data_Ts, t, S = readMarkusData(settings_dict, interval, settings_Gao) 
        #ne in [1/m^3], Te in [eV], positions in [m], data_Ts in [°C]?, t in [s]?, S in [m]?

        #modify data arrays so that they fit the functions below
        tii = np.hstack(t)           #times for Gao
        ti = (tii/2) * 1004/interval #same number od times for Lukas data
        tiii = ti.astype(int)

        #take the right amount of averages from Lukas data (here every tenth value)
        n_e_values = []
        T_e_values = []
        for ti in tiii:
            if ti > len(ne_values) - 1: 
                ti = len(ne_values) - 1
            n_e_values.append(ne_values[ti])
            T_e_values.append(Te_values[ti])
        n_e_values = np.array(n_e_values)
        T_e_values = np.array(T_e_values)

        #surface temperature array has less measurement positions, extrapolate to further end of target finger by appending the last value as often as neccessary
        T_s_values = []
        if len(data_Ts[0]) < len(n_e_values[0]):
            for Ts in data_Ts:
                Ts = Ts.tolist()
                for i in range (len(n_e_values[0]) - len(data_Ts[0])):
                    Ts.append(Ts[-1]) 
                #print(Ts)
                T_s_values.append(Ts)
        T_s_values = np.array(T_s_values)
        T_i_values = T_e_values

        dt = [0]
        for i, t_ii in enumerate(tii):
            if i < (len(tii) - 1):
                dt.append(tii[1 + i] - t_ii)
        dt = np.array(dt)

        #test for same shape, 9 times (lines) and 15 positions (columns)
        #print(np.shape(n_e_values))
        #print(np.shape(T_e_values))
        #print(np.shape(T_s_values))
        #print(np.shape(dt))
        
        safe = 'compareMarkus/tableCompareMarkus_{exp}.{discharge}_{divertorUnit}_{moduleFinger}.csv'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertorUnit=settings_Gao['divertor'], moduleFinger=settings_Gao['finger'])
        
        calculateErosionRelatedQuantitiesSeveralPositions(T_s_values.T, T_e_values.T, T_i_values.T, n_e_values.T, dt, safe, m_i, f_i, ions, k, n_target)

#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, m_i, f_i, ions, k, n_target):
    if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
        #calculate fluxes for all ion species [H, D, T, C, O] at each single time step (number of time steps = len(dt))
        fluxes = []
        for m, f in zip(m_i, f_i):
            fluxes.append(calc.calculateFluxIncidentIon(T_e/k, T_i/k, m, n_e, f)) #T_e and T_i must be in [K], so conversion from [eV] by dividing through k
        fluxes = np.array(fluxes)   
        #nested array with fluxes[i] is representing fluxes for one ion species at all times and fluxes.T[i] representing fluxes of all ion species for one single time step 

        #calculate sputtering yields [H, D, T, C, O] at each single time
        Y_i = []
        for flux, ion in zip(fluxes, ions):
            Y_i_single = []
            for i in range(len(flux)):
                #iterate over different time steps for one ion species 
                if flux[i] != 0:
                    Y_i_single.append(calc.calculateTotalErosionYield(ion, T_i[i], 'C', alpha, T_s[i], flux[i]))
                else:
                    Y_i_single.append(0)
            Y_i.append(Y_i_single)
        Y_i = np.array(Y_i)
        #print('Total sputtering yields for the ions [H, D, T, C, O]:\n {Y_i}'.format(Y_i=Y_i))
        #nested array with Y_i[i] is representing Y for one ion species at all times and Y_i.T[i] representing Y of all ion species for one single time step 


        #calculate erosion rate caused by all ion species at each single time 
        erosionRate_dt = []
        for Y_dt, flux in zip(Y_i.T, fluxes.T):
            erosionRate_dt.append(calc.calculateErosionRate(Y_dt, flux, n_target))
        #print('Erosion rates for the ions [H, D, T, C, O]:\n {erosionRate_dt}'.format(erosionRate_dt=erosionRate_dt))
        #erosionRate_dt[i] is representing the total erosion rate of all ion species at a single time step

        #calculate eroded layer thickness over the whole discharge
        erodedLayerThickness_dt = []
        for Y_dt, flux, dt_step in zip(Y_i.T, fluxes.T, dt):
            erodedLayerThickness_dt.append(calc.calculateDeltaErodedLayer(Y_dt, flux, dt_step, n_target))
        #print('Total erosion layer thickness for each time step:\n {erodedLayerThickness_dt}'.format(erodedLayerThickness_dt=erodedLayerThickness_dt))
        #erodedLayerThickness_dt[i] is representing the total layer thickness eroded by all ion species together during a single time step
        
        erodedLayerThickness = []
        for i in range(len(erodedLayerThickness_dt)):
            erodedLayerThickness.append(sum(erodedLayerThickness_dt[:i + 1]))
        #print('Total erosion layer thickness over all x time steps:\n {erodedLayerThickness}'.format(erodedLayerThickness=erodedLayerThickness))
        #erodedLayerThickness is representing the total layer thickness eroded by all ion species together over all time steps from 0 to x (x in [0, len(dt)])

        return Y_i[0], Y_i[1], Y_i[2], Y_i[3], Y_i[4], erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness
    
    else:
        return 'lengths of input arrays is not matching'
    
#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesSeveralPositions(T_s_values, T_e_values, T_i_values, n_e_values, dt, safe, m_i, f_i, ions, k, n_target):
    #_values arrays are 2 dimensional with lines representig the time and columns representing positions, e.g _values[0] being the array _values at position 1 over all times
    position_counter = 0 #for how many positions there are usable measurement values (= equal number of T_s, T_e, n_e, dt values)?

    #arrays will be 2 dimensional with e.g Y_H[0] being the array of erosion yields at position 1 over all times
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], [] 
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []

    for T_s, T_e, T_i, n_e in zip(T_s_values, T_e_values, T_i_values, n_e_values):
        if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
            position_counter += 1

            Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, m_i, f_i, ions, k, n_target)
            Y_H.append(Y_0)
            Y_D.append(Y_1)
            Y_T.append(Y_2)
            Y_C.append(Y_3)
            Y_O.append(Y_4)
            erosionRate_dt_position.append(erosionRate_dt)
            erodedLayerThickness_dt_position.append(erodedLayerThickness_dt)
            erodedLayerThickness_position.append(erodedLayerThickness)

    time =[list(range(1, len(Y_H[0]) + 1))] * position_counter 
    #print(time)  

    position = []
    for i in range(1, position_counter + 1):
        position.append([i] * (len(time[0])))
    np.hstack(position)
    #print(position)

    #print(len(np.hstack(position)), len(np.hstack(Y_H)), len(np.hstack(Y_D)), len(np.hstack(Y_T)), len(np.hstack(Y_C)), len(np.hstack(Y_O)), len(np.hstack(erosionRate_dt_position)), len(np.hstack(erodedLayerThickness_dt_position)), len(np.hstack(erodedLayerThickness_position)))
    tableOverview = {'LangmuirProbe':np.hstack(position),
                        #'Position':
                        'time': np.hstack(time),
                        'Y_H':np.hstack(Y_H), 
                        'Y_D':np.hstack(Y_D),  
                        'Y_T':np.hstack(Y_T),  
                        'Y_C':np.hstack(Y_C), 
                        'Y_O':np.hstack(Y_O), 
                        'erosionRate':np.hstack(erosionRate_dt_position),
                        'erodedLayerThickness':np.hstack(erodedLayerThickness_dt_position),
                        'totalErodedLayerThickness':np.hstack(erodedLayerThickness_position)}
    tableOverview = pd.DataFrame(tableOverview)
    
    #if results should be compared to Markus Kandlers results
    compare.compareMarkus(safe, tableOverview)
    
    tableOverview.to_csv(safe, sep=';')

#######################################################################################################################################################################
#plot sputtering yields, erosion rate, total eroded layer thickness, and electron temperature, surface temperature, density over time for one discharge and one langmuir probe
def plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, dt, safe):
    fig = plt.figure()#figsize=figsize)
    filter = np.array([i != 0 for i in Y_0])
    
    plt.plot(dt[filter], np.array(n_e)[filter], label='$n_e$')
    plt.plot(dt[filter], T_e[filter], label='$T_e$')
    plt.plot(dt[filter], T_s[filter], label='$T_s$')
    plt.plot(dt[filter], np.array(Y_0)[filter], label='Y of hydrogen')
    #plt.plot(dt, Y_1, label='Y of deuterium')
    #plt.plot(dt, Y_2, label='Y of tritium')
    plt.plot(dt[filter], np.array(Y_3)[filter], label='Y of carbon')
    plt.plot(dt[filter], np.array(Y_4)[filter], label='Y of oxygen')
    plt.plot(dt[filter], np.array(erosionRate_dt)[filter] * 1e9, label=' $\Delta_{ero}/t$')
    plt.plot(dt[filter], np.array(erodedLayerThickness)[filter] * 1e9, label=' $\Delta_{ero}$')

    plt.yscale('log')
    plt.xlabel('Time t from start of the discharge (s)')
    plt.ylabel('Plasma density $n_e$ (x 1e19 1/m$^3$)\nElectron temperature $T_e$ (eV)\nSurface temperature $T_s$ (K)\nSputtering yields Y\nErosion rate $\Delta_{ero}/t$ (nm/s)\nTotal eroded layer thickness  $\Delta_{ero}$ (nm)')
    plt.legend()

    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()

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

#target density in [1/m^3]
n_target = 11.3*1e28 #nonsense value?

#Parameters for net erosion specifically for divertor
lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct


#######################################################################################################################################################################
#incident angle in rad
alpha = 2 * np.pi/9

#######################################################################################################################################################################
#Markus data read in
#processMarkusData(m_i, f_i, ions, k, n_target, interval = 50)

#######################################################################################################################################################################
#read data from archieveDB

#######################################################################################################################################################################
#read Langmuir Probe data from xdrive
discharge = '20241127.034'
ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper = readLangmuirProbeDataFromXdrive(discharge)

#for Ts values are missing, so first Te is used as Ts value in K
Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], []
erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []
LP, time = [], []
ne, Te, Ts = [], [], []

divertorUnit = 'lower'

for ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, t_divertorUnit in zip([ne_lower, ne_upper], [Te_lower, Te_upper], [Te_lower, Te_upper], [Te_lower, Te_upper], [t_lower, t_upper]):
    Y_0_divertorUnit, Y_1_divertorUnit, Y_2_divertorUnit, Y_3_divertorUnit, Y_4_divertorUnit = [], [], [], [], []
    erosionRate_dt_divertorUnit, erodedLayerThickness_dt_divertorUnit, erodedLayerThickness_divertorUnit = [], [], []
    position = 1
    for n_e, T_e, T_i, T_s, dt in zip(ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, t_divertorUnit):
        #I need the time differences between two adjacent times to give to the function below
        timesteps = [dt[0]]
        for j in range(1, len(dt)):
            timesteps.append(dt[j] - dt[j - 1])
        timesteps = np.array(timesteps)

        Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, m_i, f_i, ions, k, n_target)
        
        #plot n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness over time
        #safe = 'plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
        #plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, dt, safe)
 
        #nested with e.g. Y_0[0] is sputtering yield for H for first position over all times
        Y_0_divertorUnit.append(Y_0)    
        Y_1_divertorUnit.append(Y_1)    
        Y_2_divertorUnit.append(Y_2)    
        Y_3_divertorUnit.append(Y_3)    
        Y_4_divertorUnit.append(Y_4)  
        erosionRate_dt_divertorUnit.append(erosionRate_dt)
        erodedLayerThickness_dt_divertorUnit.append(erodedLayerThickness_dt)
        erodedLayerThickness_divertorUnit.append(erodedLayerThickness)
        
        ne.append(n_e)
        Te.append(T_e)
        Ts.append(T_s)
        
        time.append(dt)
        LP.append(['{divertorUnit}{position}'.format(divertorUnit=divertorUnit, position=position)]*len(Y_0))
        position += 1

    #nested with e.g. Y_H[0] is sputtering yield for H on lower divertor for all positions there over all times, Y_H[0][0] for first position over all times    
    Y_H.append(Y_0_divertorUnit)
    Y_D.append(Y_1_divertorUnit)
    Y_T.append(Y_2_divertorUnit)
    Y_C.append(Y_3_divertorUnit)
    Y_O.append(Y_4_divertorUnit)
    erosionRate_dt_position.append(erosionRate_dt_divertorUnit)
    erodedLayerThickness_dt_position.append(erodedLayerThickness_dt_divertorUnit)
    erodedLayerThickness_position.append(erodedLayerThickness_divertorUnit)

    divertorUnit = 'upper'

tableOverview = {'LangmuirProbe':list(itertools.chain.from_iterable(LP)), #chain.from_iterable flattens nested lists by one dimension
                #'Position':
                'time':list(itertools.chain.from_iterable(time)),
                'ne':list(itertools.chain.from_iterable(ne)),
                'Te':list(itertools.chain.from_iterable(Te)),
                'Ts':list(itertools.chain.from_iterable(Ts)),
                'Y_H':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(Y_H)))), 
                'Y_D':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(Y_D)))), 
                'Y_T':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(Y_T)))), 
                'Y_C':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(Y_C)))), 
                'Y_O':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(Y_O)))), 
                'erosionRate':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(erosionRate_dt_position)))),
                'erodedLayerThickness':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(erodedLayerThickness_dt_position)))),
                'totalErodedLayerThickness':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(erodedLayerThickness_position))))}

#print(len(tableOverview['LangmuirProbe']), len(tableOverview['time']), len(tableOverview['Y_O']), len(tableOverview['erosionRate']), len(tableOverview['erodedLayerThickness']),len(tableOverview['totalErodedLayerThickness']))

tableOverview = pd.DataFrame(tableOverview) #missing values in the table are nan values
safe = 'calculationTables/results_{discharge}.csv'.format(discharge=discharge)
tableOverview.to_csv(safe, sep=';')


#######################################################################################################################################################################
