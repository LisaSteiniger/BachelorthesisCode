'''This file contains the functions neccessary for processing data from w7xArchieveDB/xdrive/downloaded files... and calculating erosion related quantities'''

import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.SputteringYieldFunctions as calc
import src.ReadArchieveDB as read
import src.PlotData as plot

#######################################################################################################################################################################
def compareMarkus(safe, tableOverview):
    ''' compares results calculated using this porgramm code to result files returned from M. Kandlers program 
        (create them seperately and beforehand by running "Python skripts by Markus Kandler" "FinaleRechnungen.py")'''
    dataMarkus = pd.read_csv('{safe}_Markus.csv'.format(safe=safe[:-4]), sep=';')
        
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['{key}_Markus'.format(key=key)] = dataMarkus[key]
    
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['Delta_{key}'.format(key=key)] = np.round((tableOverview[key] - dataMarkus[key])/tableOverview[key], 10)

    return tableOverview

#######################################################################################################################################################################
def processMarkusData(m_i, f_i, ions, k, n_target, interval = 50): 
    ''' reads data for exemplary discharges from OP1.2b from downloaded files, processes it and writes it to an overview .csv file together with calculated values for erosion related quamtities
        average interval = 50 -> 50 subsequent values of the value arrays below are averaged'''
    #to identify path to downloaded langmuir probe files with electron density and temperature
    settings_dict_list = [{'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},

                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'}]#,

                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'},
                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'},
                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'}]
    
    #to identify path to downloaded IRcam files with divertor surface temperatures
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
        ne_values, Te_values, positions, data_Ts, t, S = read.readMarkusData(settings_dict, interval, settings_Gao) 
        #ne in [1/m^3], Te in [eV], positions in [m], data_Ts in [°C], t in [s], S in [m]

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
        
        #calculates erosion related quantities and writes to .csv file saved as "safe" together with used measurement data
        safe = 'results/compareMarkus/tableCompareMarkus_{exp}.{discharge}_{divertorUnit}_{moduleFinger}.csv'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertorUnit=settings_Gao['divertor'], moduleFinger=settings_Gao['finger'])
        calculateErosionRelatedQuantitiesSeveralPositions(T_s_values.T, T_e_values.T, T_i_values.T, n_e_values.T, dt, safe, m_i, f_i, ions, k, n_target, compareResults=True)

#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesSeveralPositions(T_s_values, T_e_values, T_i_values, n_e_values, dt, alpha, safe, m_i, f_i, ions, k, n_target, compareResults=False):
    ''' returns suttering yields for hydrogen, deuterium, tritium, carbon and oxygen on carbon targets, the combined erosion rates and layer thicknesses
        _values arrays are 2 dimensional with columns representig the time and lines representing positions, e.g _values[0] being the array _values at position 1 over all times
        ne in [1/m^3], Te and Ti in [eV], dt are time steps (duration) in [s], Ts in [K], alpha in [rad], f being concentrtions of ions, m in [kg], k in [eV/K], n_target in [1/m^3]'''
    position_counter = 0 #for how many positions there are usable measurement values (= equal number of T_s, T_e, n_e, dt values)?

    #arrays will be 2 dimensional with e.g Y_H[0] being the array of erosion yields at position 1 over all times
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], [] 
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []
    depositionRate_dt_position, depositedLayerThickness_dt_position, depositedLayerThickness_position = [], [], []

    
    for T_s, T_e, T_i, n_e in zip(T_s_values, T_e_values, T_i_values, n_e_values): #calculate erosio related quantities for each position
        if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
            position_counter += 1

            #returns arrays over all time steps at this location
            return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, alpha, m_i, f_i, ions, k, n_target)
            if type(return_erosion) == str:
                continue
            Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = return_erosion
            Y_H.append(Y_0)
            Y_D.append(Y_1)
            Y_T.append(Y_2)
            Y_C.append(Y_3)
            Y_O.append(Y_4)
            erosionRate_dt_position.append(erosionRate_dt)
            erodedLayerThickness_dt_position.append(erodedLayerThickness_dt)
            erodedLayerThickness_position.append(erodedLayerThickness)
            depositionRate_dt_position.append(depositionRate_dt)
            depositedLayerThickness_dt_position.append(depositedLayerThickness_dt)
            depositedLayerThickness_position.append(depositedLayerThickness)

    #times are not actually times in [s] but numbered time steps 1 to x
    time = []
    time = [list(range(1, len(Y_H[0]) + 1))] * position_counter 
    #print(time)  

    #positions are no actually distances from pumping gap in [m] but numbered langmuir probes
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
                        'totalErodedLayerThickness':np.hstack(erodedLayerThickness_position),
                        'depositionRate':np.hstack(depositionRate_dt_position),
                        'depositedLayerThickness':np.hstack(depositedLayerThickness_dt_position),
                        'totalDepositedLayerThickness':np.hstack(depositedLayerThickness_position)}
    tableOverview = pd.DataFrame(tableOverview)
    
    #if results should be compared to Markus Kandlers results
    if compareResults==True:
        tableOverview = compareMarkus(safe, tableOverview)
    
    #does not return anything specifically but prints results to .csv file
    tableOverview.to_csv(safe, sep=';')

#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, alpha, m_i, f_i, ions, k, n_target):
    ''' returns sputtering yields for hydrogen, deuterium, tritium, carbon and oxygen on carbon targets, the combined erosion rates and layer thicknesses for all time steps at one position
        _values arrays are 1 dimensional the time steps at one position, e.g _values[0] being the value at position 1 at time step 1
        ne in [1/m^3], Te and Ti in [eV], dt are time steps (duration) in [s], Ts in [K], alpha in [rad], m in [kg], f being concentrtions of ions, k in [eV/K], n_target in [1/m^3]'''

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

        #calculate eroded layer thickness for each time step
        erodedLayerThickness_dt = []
        for Y_dt, flux, dt_step in zip(Y_i.T, fluxes.T, dt):
            erodedLayerThickness_dt.append(calc.calculateDeltaErodedLayer(Y_dt, flux, dt_step, n_target))
        #print('Total erosion layer thickness for each time step:\n {erodedLayerThickness_dt}'.format(erodedLayerThickness_dt=erodedLayerThickness_dt))
        #erodedLayerThickness_dt[i] is representing the total layer thickness eroded by all ion species together during a single time step
        
        #calculate accumulated eroded layer thickness during discharge up to  each time step
        erodedLayerThickness = []
        for i in range(len(erodedLayerThickness_dt)):
            erodedLayerThickness.append(sum(erodedLayerThickness_dt[:i + 1]))
        #print('Total erosion layer thickness over all x time steps:\n {erodedLayerThickness}'.format(erodedLayerThickness=erodedLayerThickness))
        #erodedLayerThickness is representing the total layer thickness eroded by all ion species together over all time steps from 0 to x (x in [0, len(dt)])

        #calculate deposition rate for each time step
        depositionRate_dt = []
        for flux in fluxes[3]:
            depositionRate_dt.append(calc.calculateDepositionRate(flux, n_target))
        #depositionRate_dt[i] is representing the deposition rate of carbon deposited during a single time step
    
        #calculate deposited layer thickness for each time step
        depositedLayerThickness_dt = []
        for flux, timestep in zip(fluxes[3], dt):
            depositedLayerThickness_dt.append(calc.calculateDepositionLayerThickness(flux, timestep, n_target))
        #depositedLayerThickness_dt[i] is representing the layer thickness of carbon deposited during a single time step
    
        #calculate accumulated eroded layer thickness during discharge up to  each time step
        depositedLayerThickness = []
        for i in range(len(depositedLayerThickness_dt)):
            depositedLayerThickness.append(sum(depositedLayerThickness_dt[:i + 1]))
        #print('Total erosion layer thickness over all x time steps:\n {erodedLayerThickness}'.format(erodedLayerThickness=erodedLayerThickness))
        #erodedLayerThickness is representing the total layer thickness eroded by all ion species together over all time steps from 0 to x (x in [0, len(dt)])


        return [Y_i[0], Y_i[1], Y_i[2], Y_i[3], Y_i[4], erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness]
    
    else:
        return 'lengths of input arrays is not matching'

#######################################################################################################################################################################
def processOP2Dataold(discharge, ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, plotting=False):
    ''' calculate sputtering related physical quantities (sputtering yields, erosion rates, layer thicknesses)
        all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s], Ts in [K], alpha in [rad], LP_position in [m] from pumping gap, ion masses m_i in [kg], f_i are ion concentrations, k in [eV/K], n_target in [1/m^3]
        does not return something but writes measurement values and calculated values to 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
        plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
        plotting=True refers to the plotting of erosion yields, erosion rate, etc over time'''
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], []
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []
    LP, LP_distance, time = [], [], [] #LP is the number of the langmuir probe and which divertor unit it belongs to, LP_distance is the distance in [m] from the pumping gap
    ne, Te, Ts = [], [], []

    divertorUnit = 'lower'

    #treat each divertor unit separately
    for ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, t_divertorUnit in zip([ne_lower, ne_upper], [Te_lower, Te_upper], [Te_lower, Te_upper], [Ts_lower, Ts_upper], [t_lower, t_upper]):
        
        Y_0_divertorUnit, Y_1_divertorUnit, Y_2_divertorUnit, Y_3_divertorUnit, Y_4_divertorUnit = [], [], [], [], []
        erosionRate_dt_divertorUnit, erodedLayerThickness_dt_divertorUnit, erodedLayerThickness_divertorUnit = [], [], []
        position = 1

        #treat each langmuir probe position separately
        for n_e, T_e, T_i, T_s, dt in zip(ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, t_divertorUnit):
            #I need the time differences between two adjacent times to give to the function below
            if len(dt) != 0:
                timesteps = [dt[0]]
            else:
                timesteps = []

            for j in range(1, len(dt)):
                timesteps.append(dt[j] - dt[j - 1])
            timesteps = np.array(timesteps)

            #calculate sputtering yields for each ion species ([H, D, T, C, O]), erosion rates, layer thicknesses
            return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, m_i, f_i, ions, k, n_target)
            if type(return_erosion) == str:
                print(return_erosion)
                exit()
            else:
                Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = return_erosion
            #plot n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness over time
            if plotting==True:    
                safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
                plot.plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, dt, safe)
    
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
            LP_distance.append([LP_position[position - 1]]*len(Y_0))
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
                    'Position':list(itertools.chain.from_iterable(LP_distance)),
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

    #does not return anything but saves measured values and calculated quantities in .csv file
    tableOverview = pd.DataFrame(tableOverview) #missing values in the table are nan values
    safe = 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
    tableOverview.to_csv(safe, sep=';')
        
#######################################################################################################################################################################

def processOP2Data(discharge, ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, index_lower, index_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, plotting=False):
    ''' calculate sputtering related physical quantities (sputtering yields, erosion rates, layer thicknesses)
        all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s], Ts in [K], alpha in [rad], LP_position in [m] from pumping gap, ion masses m_i in [kg], f_i are ion concentrations, k in [eV/K], n_target in [1/m^3]
        does not return something but writes measurement values and calculated values to 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
        plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
        plotting=True refers to the plotting of erosion yields, erosion rate, etc over time'''
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], []
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []
    depositionRate_dt_position, depositedLayerThickness_dt_position, depositedLayerThickness_position = [], [], []
    LP, LP_distance, time = [], [], [] #LP is the number of the langmuir probe and which divertor unit it belongs to, LP_distance is the distance in [m] from the pumping gap
    ne, Te, Ts = [], [], []

    divertorUnit = 'lower'

    #treat each divertor unit separately
    for ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, LP_indices, t_divertorUnit in zip([ne_lower, ne_upper], [Te_lower, Te_upper], [Te_lower, Te_upper], [Ts_lower, Ts_upper], [index_lower, index_upper], [t_lower, t_upper]):
        
        Y_0_divertorUnit, Y_1_divertorUnit, Y_2_divertorUnit, Y_3_divertorUnit, Y_4_divertorUnit = [], [], [], [], []
        erosionRate_dt_divertorUnit, erodedLayerThickness_dt_divertorUnit, erodedLayerThickness_divertorUnit = [], [], []
        depositionRate_dt_divertorUnit, depositedLayerThickness_dt_divertorUnit, depositedLayerThickness_divertorUnit = [], [], []
        #position = 1

        #treat each langmuir probe position separately
        for n_e, T_e, T_i, T_s, LP_index, dt in zip(ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, LP_indices, t_divertorUnit):
            #I need the time differences between two adjacent times to give to the function below
            if len(dt) != 0:
                timesteps = [dt[0]]
            else:
                timesteps = []

            for j in range(1, len(dt)):
                timesteps.append(dt[j] - dt[j - 1])
            timesteps = np.array(timesteps)

            #calculate sputtering yields for each ion species ([H, D, T, C, O]), erosion rates, layer thicknesses
            return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, m_i, f_i, ions, k, n_target)
            if type(return_erosion) == str:
                print(return_erosion)
                exit()
            else:
                Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = return_erosion
            #plot n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness over time
            if plotting==True:    
                safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=LP_index)
                plot.plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness, dt, safe)
    
            #nested with e.g. Y_0[0] is sputtering yield for H for first position over all times
            Y_0_divertorUnit.append(Y_0)    
            Y_1_divertorUnit.append(Y_1)    
            Y_2_divertorUnit.append(Y_2)    
            Y_3_divertorUnit.append(Y_3)    
            Y_4_divertorUnit.append(Y_4)

            erosionRate_dt_divertorUnit.append(erosionRate_dt)
            erodedLayerThickness_dt_divertorUnit.append(erodedLayerThickness_dt)
            erodedLayerThickness_divertorUnit.append(erodedLayerThickness)
              
            depositionRate_dt_divertorUnit.append(depositionRate_dt)
            depositedLayerThickness_dt_divertorUnit.append(depositedLayerThickness_dt)
            depositedLayerThickness_divertorUnit.append(depositedLayerThickness)
            
            ne.append(n_e)
            Te.append(T_e)
            Ts.append(T_s)
            
            time.append(dt)
            LP.append(['{divertorUnit}{position}'.format(divertorUnit=divertorUnit, position=LP_index)]*len(Y_0))
            LP_distance.append([LP_position[LP_index]]*len(Y_0))
            #position += 1

        #nested with e.g. Y_H[0] is sputtering yield for H on lower divertor for all positions there over all times, Y_H[0][0] for first position over all times    
        Y_H.append(Y_0_divertorUnit)
        Y_D.append(Y_1_divertorUnit)
        Y_T.append(Y_2_divertorUnit)
        Y_C.append(Y_3_divertorUnit)
        Y_O.append(Y_4_divertorUnit)
        
        erosionRate_dt_position.append(erosionRate_dt_divertorUnit)
        erodedLayerThickness_dt_position.append(erodedLayerThickness_dt_divertorUnit)
        erodedLayerThickness_position.append(erodedLayerThickness_divertorUnit)

        depositionRate_dt_position.append(depositionRate_dt_divertorUnit)
        depositedLayerThickness_dt_position.append(depositedLayerThickness_dt_divertorUnit)
        depositedLayerThickness_position.append(depositedLayerThickness_divertorUnit)

        divertorUnit = 'upper'

    tableOverview = {'LangmuirProbe':list(itertools.chain.from_iterable(LP)), #chain.from_iterable flattens nested lists by one dimension
                    'Position':list(itertools.chain.from_iterable(LP_distance)),
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
                    'totalErodedLayerThickness':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(erodedLayerThickness_position)))),
                    'depositionRate':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(depositionRate_dt_position)))),
                    'depositedLayerThickness':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(depositedLayerThickness_dt_position)))),
                    'totalDepositedLayerThickness':list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(depositedLayerThickness_position))))}

    #print(len(tableOverview['LangmuirProbe']), len(tableOverview['time']), len(tableOverview['Y_O']), len(tableOverview['erosionRate']), len(tableOverview['erodedLayerThickness']),len(tableOverview['totalErodedLayerThickness']))

    #does not return anything but saves measured values and calculated quantities in .csv file
    tableOverview = pd.DataFrame(tableOverview) #missing values in the table are nan values
    safe = 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
    tableOverview.to_csv(safe, sep=';')
        
#######################################################################################################################################################################
#######################################################################################################################################################################
def findIndexLP(overviewTable):
    ''' finds first and last index of each langmuir probe in overviewTable and returns 2D array with [[LP1, firstIndex1, lastindex1], [LP2, firstindex2, last index2], ...]'''
    #langmuir probes which were active during discharge corresponding to overviewTable
    LPs = list(np.unique(overviewTable['LangmuirProbe']))

    indexLPfirst = []
    indexLPlast = []

    for LP in LPs:
        indexLPfirst.append(list(overviewTable['LangmuirProbe']).index(LP)) #.index finds index of first list element that is equal to LP 
 
    for indexFirst in indexLPfirst[1:]:
        indexLPlast.append(indexFirst - 1)  #first index of next LP minus 1 is last index of LP in question
    indexLPlast.append(len(overviewTable['LangmuirProbe']) - 1) #for last LP last index is last index of overviewTable['LangmuirProbe']
  
    LPindices = list(np.array([LPs, indexLPfirst]).T)
    LPindices = sorted(LPindices, key=lambda x: int(x[1]))

    LPs, indexLPfirst = np.array(LPindices).T
    return np.array([LPs, indexLPfirst, sorted(indexLPlast)]).T #ATTENTION: ALL ELEMENTS ARE STRINGS

#######################################################################################################################################################################
def intrapolateMissingValues(discharge, overviewTable, LPindices, alpha, m_i, f_i, ions, k, n_target, plotting=False):
    ''' intrapolates missing values in electron density ne, electron temperature Te, and surface temperature Ts linearly
        calculates sputtering yields, erosion rates, and layer thicknesses
        returns updated overviewTable and overwrites old file with missing values
        adds columns "origin_ne", "origin_Te", "origin_Ts" with measured values having a "M" and intraploated an "I"
        ATTENTION: ALL ELEMENTS OF LPindices ARE STRINGS'''
    
    #create same timestamps for all LPs 
    # -> avoid [0.87, 2.87, 4.87, 0, 0] for lower0 vs [0.87, 2.87, 4.87, 6.87, 8.87] for lower1, instead apply [0.87, 2.87, 4.87, 6.87, 8.87] to all
    # -> guarantee that timesteps will be fitting and no negative timesteps occur and result in decreasing total layer thickness
    
    times = list(np.unique(overviewTable['time']))
    if 0 in times:
        times.remove(0)
    times.sort()
    times = [times] * len(LPindices)
    times = np.hstack(np.array(times))

    for quantity_column, replacer in zip(['ne', 'Te', 'Ts'], [np.nan, np.nan, 320]):
        quantity_list = []
        quantity_origin = []
        for index, quantity in enumerate(overviewTable[quantity_column]):
            if quantity == 0 or np.isnan(quantity): #intrapolation required
                quantity_origin.append('I')

                if str(index) in LPindices.T[1]: #if first value is missing
                    print(str(index) + ' in LPfirst')

                    if str(index) in LPindices.T[2]: #if first value is also last value and missing
                        quantity_list.append(replacer)
                    
                    else: #if first value is not also last value
                        for i in range(index + 1, int(LPindices[-1][2]) + 1):
                            if overviewTable[quantity_column][i] != 0 and not np.isnan(overviewTable[quantity_column][i]):  #find existing value before hitting last index, append inbetween of last existing value and that value (according to number of missing values)
                                quantity_list.append(overviewTable[quantity_column][i])
                                break

                            elif str(i) in LPindices.T[2]: #last value is reached without finding an existing value, append nan
                                quantity_list.append(replacer)
                                break

                elif str(index) in LPindices.T[2]: #if last value is missing
                    quantity_list.append(quantity_list[-1])
                    print(str(index) + ' in LPlast')

                
                else: #if intermediate value is missing
                    for counter, i in enumerate(range(index + 1, int(LPindices[-1][2]) + 1)):
                        if overviewTable[quantity_column][i] != 0 and not np.isnan(overviewTable[quantity_column][i]):  #find existing value before hitting last index, append inbetween of last existing value and that value (according to number of missing values)
                            m = (overviewTable[quantity_column][i] - quantity_list[-1])/(overviewTable['time'][i] - overviewTable['time'][index - 1])
                            quantity_list.append(quantity_list[-1] + m * (overviewTable['time'][index] - overviewTable['time'][index - 1]))
                            #quantity_list.append((quantity_list[index - 1] + overviewTable[quantity_column][i])/(counter + 2))
                            break

                        elif str(i) in LPindices.T[2]: #last value is reached without finding an existing value, append last existing value
                            quantity_list.append(quantity_list[-1])
                            break
                            
            else:   #no intrapolation, just take existing value
                quantity_list.append(quantity)
                quantity_origin.append('M')

        if quantity_column == 'ne':
            ne_list = np.array(np.array_split(quantity_list, len(LPindices)))
            origin_ne = quantity_origin
        elif quantity_column == 'Te':
            Te_list = np.array(np.array_split(quantity_list, len(LPindices)))
            origin_Te = quantity_origin
        else:
            Ts_list = np.array(np.array_split(quantity_list, len(LPindices)))
            origin_Ts = quantity_origin

    #get time steps
    timesteps = [times[0]] #[overviewTable['time'][0]]
    for j in range(1, len(times[:int(LPindices[0][2]) + 1])):#len(overviewTable['time'][:int(LPindices[0][2]) + 1])):
        timesteps.append(times[j] - times[j - 1])#overviewTable['time'][j] - overviewTable['time'][j - 1])
    timesteps = np.array(timesteps)

    #now calculate sputtering yields
    #Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = calculateErosionRelatedQuantitiesOnePosition(Te_list, Te_list, Ts_list, ne_list, timesteps, alpha, m_i, f_i, ions, k, n_target)
    
    Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = [], [], [], [], [], [], [], [], []

    for T_e, T_i, T_s, n_e, LPindex in zip(Te_list, Te_list, Ts_list, ne_list, LPindices):
        return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, m_i, f_i, ions, k, n_target)
        if type(return_erosion) == str:
            print(return_erosion)
            exit()
        else:
            Y_0.append(return_erosion[0])
            Y_3.append(return_erosion[3])
            Y_4.append(return_erosion[4])
            erosionRate_dt.append(return_erosion[5])
            erodedLayerThickness_dt.append(return_erosion[6])
            erodedLayerThickness.append(return_erosion[7])
            depositionRate_dt.append(return_erosion[8])
            depositedLayerThickness_dt.append(return_erosion[9])
            depositedLayerThickness.append(return_erosion[10])
            #plotting not tested yet
            if plotting==True:   
                safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=LPindex[0][:5], position=LPindex[0][5:])
                #print(times)
                #print(len(n_e), len(T_e), len(T_s), len(return_erosion[0]), len(times[:int(LPindices[0][2]) + 1]))
                plot.plotOverview(n_e, T_e, T_s, return_erosion[0], return_erosion[3], return_erosion[4], 
                                  return_erosion[5], return_erosion[7], return_erosion[8], return_erosion[10], times[:int(LPindices[0][2]) + 1], safe)
    

    #overwrite overviewTable file
    overviewTable['time'] = times
    overviewTable['ne'] = np.hstack(ne_list)
    overviewTable['origin_ne'] = origin_ne
    overviewTable['Te'] = np.hstack(Te_list)
    overviewTable['origin_Te'] = origin_Te
    overviewTable['Ts'] = np.hstack(Ts_list)
    overviewTable['origin_Ts'] = origin_Ts
    overviewTable['Y_H'] = np.hstack(Y_0)
    overviewTable['Y_C'] = np.hstack(Y_3)
    overviewTable['Y_O'] = np.hstack(Y_4)
    overviewTable['erosionRate'] = np.hstack(erosionRate_dt)
    overviewTable['erodedLayerThickness'] = np.hstack(erodedLayerThickness_dt)
    overviewTable['totalErodedLayerThickness'] = np.hstack(erodedLayerThickness)
    overviewTable['depositionRate'] = np.hstack(depositionRate_dt)
    overviewTable['depositedLayerThickness'] = np.hstack(depositedLayerThickness_dt)
    overviewTable['totalDepositedLayerThickness'] = np.hstack(depositedLayerThickness)

    #does not return anything but saves measured values and calculated quantities in .csv file
    overviewTable = pd.DataFrame(overviewTable) #missing values in the table are nan values
    safe = 'results/calculationTablesNew/results_{discharge}.csv'.format(discharge=discharge)
    overviewTable.to_csv(safe, sep=';')
    
    return overviewTable

#######################################################################################################################################################################
def calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, m_i, f_i, ions, k, n_target, intrapolated=False, plotting=False):
    ''' calculates total erosion layer thickness for all LPs of a discharge by adding erosion layer thickness up to last LP measurement to erosion layer thickness from last LP measurement to end of discharge
        duration is duration of discharge corresponding to overviewTable that is by default not intrapolated (intrapolated=False) 
        returns 2D array of structure [[LP1, erosionLayer1], [LP2, erosionLayer2], ...]'''
    #get first and last index of each LP in overviewTable, LPindices is 2D array with [[LP1, firstIndex1, lastindex1], [LP2, firstindex2, last index2], ...]
    LPindices = findIndexLP(overviewTable) 

    #intrapolate overviewTable in required
    if intrapolated == False:
        overviewTable = intrapolateMissingValues(discharge, overviewTable, LPindices, alpha, m_i, f_i, ions, k, n_target, plotting)

    #will hold the total erosion layer thickness at each LP position after discharge
    erosion = []

    #find last valid measurement (time < duration)
    if max(overviewTable['time']) > duration:
        lastTimeIndex = [j > duration for j in overviewTable['time']].index(True) - 1
        
        if lastTimeIndex > int(LPindices[0][2]): 
            for i in range(len(LPindices)):
                lastTimeIndex -= (1 + int(LPindices[0][2]))
                if lastTimeIndex < 0:
                    lastTimeIndex += (1 + int(LPindices[0][2]))
                    break
        if lastTimeIndex == - 1:
            lastTimeIndex = 0
            lastTime = 0
        else:
            lastTime = overviewTable['time'][lastTimeIndex]

            
    else:
        lastTimeIndex = int(LPindices[0][2])
        lastTime = overviewTable['time'][lastTimeIndex]

    for counter, LP in enumerate(LPindices):
        lastTimeIndexCounter = counter * (1 + int(LPindices[0][2])) + lastTimeIndex
        if lastTime == 0:
            erosionDuringLP = 0
            depositionDuringLP = 0
        else:
            erosionDuringLP = overviewTable['totalErodedLayerThickness'][lastTimeIndexCounter] #total eroded layer thickness for time interval until last LP measurement
            depositionDuringLP = overviewTable['totalDepositedLayerThickness'][lastTimeIndexCounter] #total eroded layer thickness for time interval until last LP measurement
        erosionAfterLP = overviewTable['erosionRate'][lastTimeIndexCounter] * (duration - lastTime) #overviewTable['time'][lastTimeIndex]) #total eroded layer thickness for time interval after last LP measurement, multiply last erosion rate with time step till end of discharge
        depositionAfterLP = overviewTable['depositionRate'][lastTimeIndexCounter] * (duration - lastTime) #overviewTable['time'][lastTimeIndex]) #total eroded layer thickness for time interval after last LP measurement, multiply last erosion rate with time step till end of discharge
        erosion.append([LP[0], erosionDuringLP + erosionAfterLP, depositionDuringLP + depositionAfterLP])
    
    return erosion

#######################################################################################################################################################################
def calculateTotalErodedLayerThicknessSeveralDischarges(config, discharges, durations, overviewTables, alpha, m_i, f_i, ions, k, n_target, intrapolated=False, plotting=False):
    ''' calculates total erosion layer thickness for all LPs of a discharge by adding erosion layer thickness up to last LP measurement to erosion layer thickness from last LP measurement to end of discharge
        duration is duration of discharge corresponding to overviewTable that is by default not intrapolated (intrapolated=False) 
        returns 2D array of structure [[LP1, erosionLayer1], [LP2, erosionLayer2], ...]'''
    LP_lower0, LP_lower1, LP_lower2, LP_lower3, LP_lower4, LP_lower5, LP_lower6, LP_lower7, LP_lower8, LP_lower9, LP_lower10, LP_lower11, LP_lower12, LP_lower13, LP_lower14, LP_lower15, LP_lower16, LP_lower17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] 
    LP_upper0, LP_upper1, LP_upper2, LP_upper3, LP_upper4, LP_upper5, LP_upper6, LP_upper7, LP_upper8, LP_upper9, LP_upper10, LP_upper11, LP_upper12, LP_upper13, LP_upper14, LP_upper15, LP_upper16, LP_upper17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    dischargeList, durationList = [], []
    for discharge, duration, overviewTable in zip(discharges, durations, overviewTables):
        lower = [False] * 18
        upper = [False] * 18
 
        if not os.path.isfile(overviewTable):
            continue
        
        print(discharge)
        dischargeList.append(discharge)
        durationList.append(duration)

        overviewTable = pd.read_csv(overviewTable, sep=';')
        erosion = calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, m_i, f_i, ions, k, n_target, intrapolated, plotting)
        for erosionLP in erosion:
            if 'lower0' == erosionLP[0]:
                LP_lower0.append(erosionLP[1:])
                lower[0] = True
            elif 'lower1' == erosionLP[0]:
                LP_lower1.append(erosionLP[1:])
                lower[1] = True
            elif 'lower2' == erosionLP[0]:
                LP_lower2.append(erosionLP[1:])
                lower[2] = True
            elif 'lower3' == erosionLP[0]:
                LP_lower3.append(erosionLP[1:])
                lower[3] = True
            elif 'lower4' == erosionLP[0]:
                LP_lower4.append(erosionLP[1:])
                lower[4] = True
            elif 'lower5' == erosionLP[0]:
                LP_lower5.append(erosionLP[1:])
                lower[5] = True
            elif 'lower6' == erosionLP[0]:
                LP_lower6.append(erosionLP[1:])
                lower[6] = True
            elif 'lower7' == erosionLP[0]:
                LP_lower7.append(erosionLP[1:])
                lower[7] = True
            elif 'lower8' == erosionLP[0]:
                LP_lower8.append(erosionLP[1:])
                lower[8] = True
            elif 'lower9' == erosionLP[0]:
                LP_lower9.append(erosionLP[1:])
                lower[9] = True
            elif 'lower10' == erosionLP[0]:
                LP_lower10.append(erosionLP[1:])
                lower[10] = True
            elif 'lower11' == erosionLP[0]:
                LP_lower11.append(erosionLP[1:])
                lower[11] = True
            elif 'lower12' == erosionLP[0]:
                LP_lower12.append(erosionLP[1:])
                lower[12] = True
            elif 'lower13' == erosionLP[0]:
                LP_lower13.append(erosionLP[1:])
                lower[13] = True
            elif 'lower14' == erosionLP[0]:
                LP_lower14.append(erosionLP[1:])
                lower[14] = True
            elif 'lower15' == erosionLP[0]:
                LP_lower15.append(erosionLP[1:])
                lower[15] = True
            elif 'lower16' == erosionLP[0]:
                LP_lower16.append(erosionLP[1:])
                lower[16] = True
            elif 'lower17' == erosionLP[0]:
                LP_lower17.append(erosionLP[1:])
                lower[17] = True
            
            elif 'upper0' == erosionLP[0]:
                LP_upper0.append(erosionLP[1:])
                upper[0] = True
            elif 'upper1' == erosionLP[0]:
                LP_upper1.append(erosionLP[1:])
                upper[1] = True
            elif 'upper2' == erosionLP[0]:
                LP_upper2.append(erosionLP[1:])
                upper[2] = True
            elif 'upper3' == erosionLP[0]:
                LP_upper3.append(erosionLP[1:])
                upper[3] = True
            elif 'upper4' == erosionLP[0]:
                LP_upper4.append(erosionLP[1:])
                upper[4] = True
            elif 'upper5' == erosionLP[0]:
                LP_upper5.append(erosionLP[1:])
                upper[5] = True
            elif 'upper6' == erosionLP[0]:
                LP_upper6.append(erosionLP[1:])
                upper[6] = True
            elif 'upper7' == erosionLP[0]:
                LP_upper7.append(erosionLP[1:])
                upper[7] = True
            elif 'upper8' == erosionLP[0]:
                LP_upper8.append(erosionLP[1:])
                upper[8] = True
            elif 'upper9' == erosionLP[0]:
                LP_upper9.append(erosionLP[1:])
                upper[9] = True
            elif 'upper10' == erosionLP[0]:
                LP_upper10.append(erosionLP[1:])
                upper[10] = True
            elif 'upper11' == erosionLP[0]:
                LP_upper11.append(erosionLP[1:])
                upper[11] = True
            elif 'upper12' == erosionLP[0]:
                LP_upper12.append(erosionLP[1:])
                upper[12] = True
            elif 'upper13' == erosionLP[0]:
                LP_upper13.append(erosionLP[1:])
                upper[13] = True
            elif 'upper14' == erosionLP[0]:
                LP_upper14.append(erosionLP[1:])
                upper[14] = True
            elif 'upper15' == erosionLP[0]:
                LP_upper15.append(erosionLP[1:])
                upper[15] = True
            elif 'upper16' == erosionLP[0]:
                LP_upper16.append(erosionLP[1:])
                upper[16] = True
            elif 'upper17' == erosionLP[0]:
                LP_upper17.append(erosionLP[1:])
                upper[17] = True

        if not lower[0]:
            LP_lower0.append([np.nan] * 2)
        if not lower[1]:
            LP_lower1.append([np.nan] * 2)
        if not lower[2]:
            LP_lower2.append([np.nan] * 2)
        if not lower[3]:
            LP_lower3.append([np.nan] * 2)
        if not lower[4]:
            LP_lower4.append([np.nan] * 2)
        if not lower[5]:
            LP_lower5.append([np.nan] * 2)
        if not lower[6]:
            LP_lower6.append([np.nan] * 2)
        if not lower[7]:
            LP_lower7.append([np.nan] * 2)
        if not lower[8]:
            LP_lower8.append([np.nan] * 2)
        if not lower[9]:
            LP_lower9.append([np.nan] * 2)
        if not lower[10]:
            LP_lower10.append([np.nan] * 2)
        if not lower[11]:
            LP_lower11.append([np.nan] * 2)
        if not lower[12]:
            LP_lower12.append([np.nan] * 2)
        if not lower[13]:
            LP_lower13.append([np.nan] * 2)
        if not lower[14]:
            LP_lower14.append([np.nan] * 2) 
        if not lower[15]:
            LP_lower15.append([np.nan] * 2) 
        if not lower[16]:
            LP_lower16.append([np.nan] * 2) 
        if not lower[17]:
            LP_lower17.append([np.nan] * 2) 
        
        if not upper[0]:
            LP_upper0.append([np.nan] * 2) 
        if not upper[1]:
            LP_upper1.append([np.nan] * 2)   
        if not upper[2]:
            LP_upper2.append([np.nan] * 2)
        if not upper[3]:
            LP_upper3.append([np.nan] * 2)   
        if not upper[4]:
            LP_upper4.append([np.nan] * 2)   
        if not upper[5]:
            LP_upper5.append([np.nan] * 2)
        if not upper[6]:
            LP_upper6.append([np.nan] * 2)
        if not upper[7]:
            LP_upper7.append([np.nan] * 2)   
        if not upper[8]:
            LP_upper8.append([np.nan] * 2)   
        if not upper[9]:
            LP_upper9.append([np.nan] * 2)
        if not upper[10]:
            LP_upper10.append([np.nan] * 2)
        if not upper[11]:
            LP_upper11.append([np.nan] * 2)
        if not upper[12]:
            LP_upper12.append([np.nan] * 2)
        if not upper[13]:
            LP_upper13.append([np.nan] * 2)
        if not upper[14]:
            LP_upper14.append([np.nan] * 2)
        if not upper[15]:
            LP_upper15.append([np.nan] * 2)
        if not upper[16]:
            LP_upper16.append([np.nan] * 2)
        if not upper[17]:
            LP_upper17.append([np.nan] * 2)
    print(len(durationList), len(dischargeList), len(LP_lower0))
    
    erosionTable = pd.DataFrame({})
    erosionTable['discharge'] = dischargeList
    erosionTable['duration'] = durationList
    for LP, LP_name in zip([LP_lower0, LP_lower1, LP_lower2, LP_lower3, LP_lower4, LP_lower5, LP_lower6, LP_lower7, LP_lower8, LP_lower9, LP_lower10, LP_lower11, LP_lower12, LP_lower13, LP_lower14, LP_lower15, LP_lower16, LP_lower17,
                            LP_upper0, LP_upper1, LP_upper2, LP_upper3, LP_upper4, LP_upper5, LP_upper6, LP_upper7, LP_upper8, LP_upper9, LP_upper10, LP_upper11, LP_upper12, LP_upper13, LP_upper14, LP_upper15, LP_upper16, LP_upper17],
                            ['lower0', 'lower1', 'lower2', 'lower3', 'lower4', 'lower5', 'lower6', 'lower7', 'lower8', 'lower9', 'lower10', 'lower11', 'lower12', 'lower13', 'lower14', 'lower15', 'lower16', 'lower17', 
                             'upper0', 'upper1', 'upper2', 'upper3', 'upper4', 'upper5', 'upper6', 'upper7', 'upper8', 'upper9', 'upper10', 'upper11', 'upper12', 'upper13', 'upper14', 'upper15', 'upper16', 'upper17']):
        if LP != []:
            erosionTable[LP_name + '_erosion'] = np.array(LP).T[0]
            erosionTable[LP_name + '_deposition'] = np.array(LP).T[1]
        else:
            erosionTable[LP_name + '_erosion'] = 0
            erosionTable[LP_name + '_deposition'] = 0
    erosionTable.to_csv('results/erosionMeasuredConfig/totalErosionAtPosition_{config}.csv'.format(config=config), sep=';')
    return erosionTable

#######################################################################################################################################################################        
def calculateTotalErodedLayerThicknessWholeCampaignPerConfig(config, erosionTable='results/erosionMeasuredConfig/totalErosionAtPosition_', dischargeOverview = 'results/configurations/dischargeList_OP223'):
    if not os.path.isfile(erosionTable + '{config}.csv'.format(config=config)) or not os.path.isfile(dischargeOverview + '_{config}.csv'.format(config=config)):
        return 'file missing for ' + config
    
    erosionTable = pd.read_csv(erosionTable + '{config}.csv'.format(config=config), sep=';')
    dischargeOverview = pd.read_csv(dischargeOverview + '_{config}.csv'.format(config=config), sep=';')

    LP, erosion_knownData, deposition_knownData, time_knownErosion, time_knownDeposition, erosion_total, deposition_total = [], [], [], [], [], [], []
    totalTime = np.nansum(np.array(dischargeOverview['duration']))
    for key in erosionTable.keys():
        if 'lower' in key or 'upper' in key:
            if 'erosion' in key:
                LP.append(key.split('_')[0])
                erosion_knownData.append(np.nansum(erosionTable[key]))
                nan = np.array([np.isnan(i) for i in erosionTable[key]])
                if not list(nan):
                    time_knownErosion.append(0)
                else:
                    time_knownErosion.append(np.nansum(np.array(erosionTable['duration'])[~nan]))
                erosion_total.append(erosion_knownData[-1] * totalTime/time_knownErosion[-1])
            else:
                deposition_knownData.append(np.nansum(erosionTable[key]))
                nan = np.array([np.isnan(i) for i in erosionTable[key]])
                if not list(nan):
                    time_knownDeposition.append(0)
                else:
                    time_knownDeposition.append(np.nansum(np.array(erosionTable['duration'])[~nan]))
                deposition_total.append(deposition_knownData[-1] * totalTime/time_knownDeposition[-1])
    #print(len(LP), len(time_knownErosion), len(time_knownDeposition), len(erosion_knownData), len(deposition_knownData), len(erosion_total), len(deposition_total))
    erosion = pd.DataFrame({'LP': LP, 
                            'duration_knownErosion (s)': time_knownErosion, 
                            'erosion_known (m)': erosion_knownData, 
                            'duration_knownDeposition (s)': time_knownDeposition, 
                            'deposition_known (m)': deposition_knownData, 
                            'duration_total (s)': [totalTime] * len(LP), 
                            'erosion_total (m)': erosion_total,
                            'deposition_total (m)': deposition_total,
                            'netErosion_total (m)': np.array(erosion_total) - np.array(deposition_total)})
    erosion.to_csv('results/erosionExtrapolatedConfig/totalErosionAtPositionWholeCampaign_{config}.csv'.format(config=config), sep=';')
    return 'succesfully calculated total erosion in ' + config

#######################################################################################################################################################################        
def calculateTotalErodedLayerThicknessWholeCampaign(configurationList, LP_position, totalErosionFiles='results/erosionExtrapolatedConfig/totalErosionAtPositionWholeCampaign_'):
    erosion_position = np.array([0.] * 36)
    deposition_position = np.array([0.] * 36)
    config_missing = []
    no_data = pd.DataFrame({'LP': ['lower0', 'lower1', 'lower2', 'lower3', 'lower4', 'lower5', 'lower6', 'lower7', 'lower8', 'lower9', 'lower10', 'lower11', 'lower12', 'lower13', 'lower14', 'lower15', 'lower16', 'lower17', 
                             'upper0', 'upper1', 'upper2', 'upper3', 'upper4', 'upper5', 'upper6', 'upper7', 'upper8', 'upper9', 'upper10', 'upper11', 'upper12', 'upper13', 'upper14', 'upper15', 'upper16', 'upper17']})
                            #['lower0', 'lower0', 'lower1', 'lower1', 'lower2', 'lower2', 'lower3', 'lower3', 'lower4', 'lower4', 'lower5', 'lower5', 
                             #      'lower6', 'lower6', 'lower7', 'lower7', 'lower8', 'lower8', 'lower9', 'lower9', 'lower10', 'lower10', 'lower11', 'lower11', 
                             #      'lower12', 'lower12', 'lower13', 'lower13', 'lower14', 'lower14', 'lower15', 'lower15', 'lower16', 'lower16', 'lower17', 'lower17', 
                             #      'upper0', 'upper0', 'upper1', 'upper1', 'upper2', 'upper2', 'upper3', 'upper3', 'upper4', 'upper4', 'upper5', 'upper5', 
                             #      'upper6', 'upper6', 'upper7', 'upper7', 'upper8', 'upper8', 'upper9', 'upper9', 'upper10', 'upper10', 'upper11', 'upper11', 
                             #      'upper12', 'upper12', 'upper13', 'upper13', 'upper14', 'upper14', 'upper15', 'upper15', 'upper16', 'upper16', 'upper17', 'upper17']})
    for counter, config in enumerate(configurationList):
        if not os.path.isfile(totalErosionFiles + '{config}.csv'.format(config=config)):
            print('file missing for ' + config)
            config_missing.append(config)
            continue
        
        else:
            erosion = pd.read_csv(totalErosionFiles + '{config}.csv'.format(config=config), sep=';')
            erosion_position = np.hstack(np.nansum(np.dstack((np.array(erosion['erosion_total (m)']), erosion_position)), 2))
            deposition_position = np.hstack(np.nansum(np.dstack((np.array(erosion['deposition_total (m)']), deposition_position)), 2))
            no_data[config + '_erosion'] = np.array(erosion['erosion_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            no_data[config + '_deposition'] = np.array(erosion['deposition_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            no_data[config + '_netErosion'] = np.array(erosion['netErosion_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
    #print(erosion_position)
    plt.plot(LP_position[:14], erosion_position[:14], 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[:14], deposition_position[:14], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[:14], erosion_position[:14] - deposition_position[:14],'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[:14], erosion_position[18:32], 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[:14], deposition_position[18:32], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[:14], erosion_position[18:32] - deposition_position[18:32], 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('total layer thickness (m)')
    plt.savefig('results/erosionFullCampaign/totalErosionWholeCampaignAllPositionsLowIota.png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(LP_position[14:], erosion_position[14:18], 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[14:], deposition_position[14:18], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[14:], erosion_position[14:18] - deposition_position[14:18], 'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[14:], erosion_position[32:], 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[14:], deposition_position[32:], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[14:], erosion_position[32:] - deposition_position[32:], 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('total layer thickness (m)')
    plt.savefig('results/erosionFullCampaign/totalErosionWholeCampaignAllPositionsHighIota.png', bbox_inches='tight')
    plt.show()
    plt.close()
    no_data.to_csv('results/erosionFullCampaign/totalErosionWholeCampaignAllPositions.csv', sep=';')