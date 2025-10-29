import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.SputteringYieldFunctions as calc
import src.ReadArchieveDB as read

#######################################################################################################################################################################
def compareMarkus(safe, tableOverview):
    #compares results calculated by this program with results calculated from Markus Kandlers Program (modified version)
    dataMarkus = pd.read_csv('{safe}_Markus.csv'.format(safe=safe[:-4]), sep=';')
        
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['{key}_Markus'.format(key=key)] = dataMarkus[key]
    
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['Delta_{key}'.format(key=key)] = np.round((tableOverview[key] - dataMarkus[key])/tableOverview[key], 10)

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
        ne_values, Te_values, positions, data_Ts, t, S = read.readMarkusData(settings_dict, interval, settings_Gao) 
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
        
        safe = 'results/compareMarkus/tableCompareMarkus_{exp}.{discharge}_{divertorUnit}_{moduleFinger}.csv'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertorUnit=settings_Gao['divertor'], moduleFinger=settings_Gao['finger'])
        
        calculateErosionRelatedQuantitiesSeveralPositions(T_s_values.T, T_e_values.T, T_i_values.T, n_e_values.T, dt, safe, m_i, f_i, ions, k, n_target, compareResults=True)

#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesSeveralPositions(T_s_values, T_e_values, T_i_values, n_e_values, dt, alpha, safe, m_i, f_i, ions, k, n_target, compareResults=False):
    #_values arrays are 2 dimensional with columns representig the time and lines representing positions, e.g _values[0] being the array _values at position 1 over all times
    position_counter = 0 #for how many positions there are usable measurement values (= equal number of T_s, T_e, n_e, dt values)?

    #arrays will be 2 dimensional with e.g Y_H[0] being the array of erosion yields at position 1 over all times
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], [] 
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []

    
    for T_s, T_e, T_i, n_e in zip(T_s_values, T_e_values, T_i_values, n_e_values):
        if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
            position_counter += 1

            Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, alpha, m_i, f_i, ions, k, n_target)
            Y_H.append(Y_0)
            Y_D.append(Y_1)
            Y_T.append(Y_2)
            Y_C.append(Y_3)
            Y_O.append(Y_4)
            erosionRate_dt_position.append(erosionRate_dt)
            erodedLayerThickness_dt_position.append(erodedLayerThickness_dt)
            erodedLayerThickness_position.append(erodedLayerThickness)

    time = []
    time = [list(range(1, len(Y_H[0]) + 1))] * position_counter 
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
    if compareResults==True:
        compareMarkus(safe, tableOverview)
    
    tableOverview.to_csv(safe, sep=';')

#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, alpha, m_i, f_i, ions, k, n_target):
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
def processOP2Data(discharge, ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, plotting=False):
    #plotting refers to the plotting of erosion yields, erosion rate, etc over time 
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], []
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []
    LP, LP_distance, time = [], [], []
    ne, Te, Ts = [], [], []

    divertorUnit = 'lower'

    for ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, t_divertorUnit in zip([ne_lower, ne_upper], [Te_lower, Te_upper], [Te_lower, Te_upper], [Ts_lower, Ts_upper], [t_lower, t_upper]):
        
        Y_0_divertorUnit, Y_1_divertorUnit, Y_2_divertorUnit, Y_3_divertorUnit, Y_4_divertorUnit = [], [], [], [], []
        erosionRate_dt_divertorUnit, erodedLayerThickness_dt_divertorUnit, erodedLayerThickness_divertorUnit = [], [], []
        position = 1

        for n_e, T_e, T_i, T_s, dt in zip(ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, t_divertorUnit):
            #I need the time differences between two adjacent times to give to the function below
            if len(dt) != 0:
                timesteps = [dt[0]]
            else:
                timesteps = []

            for j in range(1, len(dt)):
                timesteps.append(dt[j] - dt[j - 1])
            timesteps = np.array(timesteps)

            Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, m_i, f_i, ions, k, n_target)
            
            #plot n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness over time
            if plotting==True:    
                safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
                plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, dt, safe)
    
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

    tableOverview = pd.DataFrame(tableOverview) #missing values in the table are nan values
    safe = 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
    tableOverview.to_csv(safe, sep=';')
        
#######################################################################################################################################################################
def plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, dt, safe):
    ''' plot sputtering yields, erosion rate, total eroded layer thickness, and electron temperature, surface temperature, density over time for one discharge and one langmuir probe
        all input parameters besides "safe" should be provided as 1D lists/arrays representing measurement/calculation values at one position/langmuir probe over the same number of timesteps
        "safe" is a string determining where to save the plot and under which name, e.g. 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)'''
    #filter out values for which not all input parameters were measured successfully (they were set to 0, thus Y_0 becomes 0)
    filter = np.array([i != 0 and np.isnan(i)==False for i in Y_0])
    
    #if all values are filtered out, no plot is neccessary
    if sum(filter)==0:
        return 'fail'
    
    fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(10, 12), sharex=True)
    
    #first subplot for measurement values
    ax[0].plot(np.array(dt)[filter], np.array(n_e)[filter], '-x', label='$n_e$')
    ax[0].plot(np.array(dt)[filter], np.array(T_e)[filter], '-x', label='$T_e$')
    ax[0].plot(np.array(dt)[filter], np.array(T_s)[filter], '-x', label='$T_s$')
    
    #second subplot for calculated sputtering yields
    ax[1].plot(np.array(dt)[filter], np.array(Y_0)[filter], '-x', label='Y of hydrogen')
    #ax[1].plot(np.array(dt)[filter], np.array(Y_1)[filter], '-x', label='Y of deuterium')
    #ax[1].plot(np.array(dt)[filter], np.array(Y_2)[filter], '-x', label='Y of tritium')
    ax[1].plot(np.array(dt)[filter], np.array(Y_3)[filter], '-x', label='Y of carbon')
    ax[1].plot(np.array(dt)[filter], np.array(Y_4)[filter], '-x', label='Y of oxygen')
    
    #third subplot for resulting erosion rates and layer thicknesses
    ax[2].plot(np.array(dt)[filter], np.array(erosionRate_dt)[filter] * 1e9, '-x', label=' $\Delta_{ero}/t$')
    ax[2].plot(np.array(dt)[filter], np.array(erodedLayerThickness)[filter] * 1e9, '-x', label=' $\Delta_{ero}$')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    ax[0].set_xlabel('Time t from start of the discharge (s)')
    ax[1].set_xlabel('Time t from start of the discharge (s)')
    ax[2].set_xlabel('Time t from start of the discharge (s)')

    ax[0].set_ylabel('Plasma density $n_e$ (x 1e19 1/m$^3$)\nElectron temperature $T_e$ (eV)\nSurface temperature $T_s$ (K)')
    ax[1].set_ylabel('Sputtering yields Y')
    ax[2].set_ylabel('\nErosion rate $\Delta_{ero}/t$ (nm/s)\nTotal eroded layer thickness  $\Delta_{ero}$ (nm)')
    
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    ax[2].grid(True)

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()
