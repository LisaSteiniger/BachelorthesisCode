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
def calculateErosionRelatedQuantitiesOnePosition(T_e: list[int|float], 
                                                 T_i: list[int|float], 
                                                 T_s: list[int|float], 
                                                 n_e: list[int|float], 
                                                 dt: list[int|float], 
                                                 alpha: int|float, 
                                                 zeta: int|float,
                                                 m_i: list[int|float], 
                                                 f_i: list[int|float], 
                                                 ions:list[str], k: int|float, 
                                                 n_target: int|float) -> str|list[list[int|float]]:
    ''' This function calculates sputtering yields for different ion species, the combined erosion rates and layer thicknesses for all time steps at one position
        -> for hydrogen, deuterium, tritium, carbon and oxygen ions on carbon targets
        "T_*" and "n_e" arrays are 1 dimensional representing the value of the quantity for each time step at the position in question
        Electron density "ne" is given in [1/m^3], electron and ion temperature "Te" and "Ti" in [eV]
        "dt" are the durations of each time step in [s], "Ts" is the surface temperature of the target in [K]
        The incident angle "alpha" of the ions is given in [rad]
        "zeta" is the incident angle of the magnetic field lines on the target measured form the surface towards the surface normal
        The ion masses "m_i" in [kg], the concentrations "f" of the ions, and the ion names "ions" are array like objects of same length
        The Boltzmann constant "k" is provided in [eV/K]
        The atomic target density "n_target" should be given in [1/m^3]
        
        Returns either a string if calculation was impossible due to not matching lengths of "dt", "T_*" and "n_e"
        Or a list of lists holding the values of some quantities at each time step
        -> the sputtering yields of all ion species, the erosion and deposition rates and layer thicknesses (for each time step and the whole time passed since start of discharge)
        -> yields are [unitless], rates are in [m/s], thicknesses in [m]'''

    if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
        #calculate fluxes for all ion species [H, D, T, C, O] at each single time step (number of time steps = len(dt))
        fluxes = []
        for m, f in zip(m_i, f_i):
            fluxes.append(calc.calculateFluxIncidentIon(zeta, T_e/k, T_i/k, m, n_e, f)) #T_e and T_i must be in [K], so conversion from [eV] by dividing through k
        fluxes = np.array(fluxes)   
        #nested array with fluxes[i] is representing fluxes for one ion species at all times and fluxes.T[i] representing fluxes of all ion species for one single time step 

        #calculate sputtering yields [H, D, T, C, O] at each single time
        Y_i = []
        for flux, ion in zip(fluxes, ions):
            Y_i_single = []
            for i in range(len(flux)):
                #iterate over different time steps for one ion species 
                if flux[i] != 0:
                    Y_i_single.append(calc.calculateTotalErosionYield(ion, T_i[i], 'C', alpha, T_s[i], flux[i], n_target))
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
def processOP2Data(discharge: str, 
                   ne_lower: list[list[int|float]],
                   ne_upper: list[list[int|float]], 
                   Te_lower: list[list[int|float]], 
                   Te_upper: list[list[int|float]], 
                   Ts_lower: list[list[int|float]], 
                   Ts_upper: list[list[int|float]], 
                   t_lower: list[list[int|float]], 
                   t_upper: list[list[int|float]], 
                   index_lower: list[int],
                   index_upper: list[int],
                   alpha: int|float, 
                   LP_position: list[int|float], 
                   LP_zeta: list[int|float], 
                   m_i: list[int|float], 
                   f_i: list[int|float], 
                   ions: list[str], 
                   k: int|float, 
                   n_target: int|float, 
                   plotting: bool =False) -> None:
    ''' This function calculates sputtering related physical quantities (sputtering yields, erosion/deposition rates, layer thicknesses) for various time steps of a discharge "discharge" at different positions
        "*_upper" and "*_lower" determine which divertor unit is considered
        Electron density "ne_*" in [1/m^3], electron temperature "Te_*" in [eV] and assumption that "Te_*" = ion temperature "Ti_*", surface temperature of the target "Ts_*" in [K], times "t_*" in [s]
        -> all arrays "ne_*", "Te_*", "Ts_*" are of ndim=2 with each line representing measurements over time at one Langmuir probe positions (=each column representing measurements at all positions at one time)
        -> times are given in [s] from trigger t1 of the discharge
        Indices of active Langmuir Probes are given by "index_*", e.g "index_lower" = [0, 1, 2, 3, 5, 6, 7] means that LP0 - LP4 and LP5 - LP7 measured something (numbering as in "LP_position") 
        Incident angle of the ions  "alpha" is given in [rad]
        The distance of each langmuir probe from the pumping gap is given in "LP_position" in [m], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The ion masses "m_i" are in [kg], the ion concentrations "f_i", and ion names "ions" should have the same length as "m_i"
        The Boltzmann constant "k" must be in [eV/K]
        The atomic target density "n_target" should be provided in [1/m^3]
        The parameter "plotting" determines, if measurement data, sputtering yields and erosion/deposition rates/layer thicknesses are plotted
        -> if True, plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
        
        This function does not return something but writes measurement values and calculated values to 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)'''
    
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

        #treat each langmuir probe position separately
        for n_e, T_e, T_i, T_s, LP_index, dt in zip(ne_divertorUnit, Te_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, LP_indices, t_divertorUnit):
            #I need the time differences between two adjacent times to give to the function below
            #changed1#####################
            if len(dt) != 0:
                if len(dt) > 1:
                    timesteps = [dt[0] + (dt[1] - dt[0])/2]#timesteps = [dt[0]]
                else:
                    timesteps = [dt[0]]
            else:
                timesteps = []

            for j in range(1, len(dt) -1):#range(1, len(dt)):
                timesteps.append((dt[j] - dt[j - 1])/2 + (dt[j + 1] - dt[j])/2)#timesteps.append(dt[j] - dt[j - 1])
            timesteps.append((dt[-1] - dt[-2])/2)
            timesteps = np.array(timesteps)
            #changed1#####################

            #calculate sputtering yields for each ion species ([H, D, T, C, O]), erosion rates, layer thicknesses
            return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, LP_zeta[LP_index], m_i, f_i, ions, k, n_target)
            if type(return_erosion) == str: #if input arrays do not have the same length, function will fail and return error string
                print(return_erosion)
                exit()
            else:
                Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = return_erosion
            
            #plot n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness over time
            if plotting:    
                safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=LP_index)
                #changed1############### remove timesteps
                plot.plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness, dt, timesteps, safe)
                #changed1###############
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
def findIndexLP(overviewTable: pd.DataFrame) -> list[list[str]]:
    ''' This function finds first and last index of each langmuir probe in "overviewTable" 
        -> "overviewTable" is Dataframe with keys 'LangmuirProbe', 'time', ne, ... as in 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge) or 'results/calculationTablesNew/results_{discharge}.csv'.format(discharge=discharge)
        Returns 2D array with [[LP1, firstIndex1, lastindex1], [LP2, firstindex2, last index2], ...] 
        -> ALL ELEMENTS IN THAT ARRAY ARE STRINGS'''
    #langmuir probes which were active during discharge corresponding to overviewTable
    LPs = list(np.unique(overviewTable['LangmuirProbe']))

    indexLPfirst = []
    indexLPlast = []

    for LP in LPs:
        indexLPfirst.append(list(overviewTable['LangmuirProbe']).index(LP)) #.index finds index of first list element that is equal to LP 
 
    LPindices = list(np.array([LPs, indexLPfirst]).T)
    LPindices = sorted(LPindices, key=lambda x: int(x[1]))

    LPs, indexLPfirst = np.array(LPindices).T

    for indexFirst in indexLPfirst[1:]:
        indexLPlast.append(int(indexFirst) - 1)  #first index of next LP minus 1 is last index of LP in question
    indexLPlast.append(len(overviewTable['LangmuirProbe']) - 1) #for last LP last index is last index of overviewTable['LangmuirProbe']
  
    return np.array([LPs, indexLPfirst, sorted(indexLPlast)]).T #ATTENTION: ALL ELEMENTS ARE STRINGS

#######################################################################################################################################################################
def intrapolateMissingValues(discharge: str, 
                             overviewTable: pd.DataFrame, 
                             LPindices: list[list[str]], 
                             alpha: int|float, 
                             LP_zeta: list[int|float],
                             m_i: list[int|float], 
                             f_i: list[int|float], 
                             ions: list[str], 
                             k: int|float, 
                             n_target: int|float,
                             defaultValues: list[int|float] =[np.nan, np.nan, 320], 
                             plotting: bool =False) -> pd.DataFrame:
    ''' This function intrapolates missing values in electron density ne, electron temperature Te, and surface temperature Ts linearly for a given plasma discharge with ID "discharge" with data stored in "overviewTable"
        -> it then calculates sputtering yields, erosion/deposition rates, and layer thicknesses
        -> it adds columns "origin_ne", "origin_Te", "origin_Ts" with measured values having a "M" and intraploated an "I" to "overviewTable"
        "overviewTable" is initially Dataframe with keys 'LangmuirProbe', 'time', ne, ... as in 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
        "LPindices" holds the name of each Langmuir Probe represented in "discharge", the index of its first and last occurance 
        -> [[LP1, firstIndex1, lastindex1], [LP2, firstindex2, last index2], ...] 
        -> ATTENTION: ALL ELEMENTS OF LPindices ARE STRINGS
        Incident angle of the ions  "alpha" is given in [rad]
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The ion masses "m_i" are in [kg], the ion concentrations "f_i", and ion names "ions" should have the same length as "m_i"
        The Boltzmann constant "k" must be in [eV/K]
        The atomic target density "n_target" should be provided in [1/m^3]
        The default values given for [ne, Te, Ts] in "defaultValues" determine which value is inserted if no data exists at a certain for the whole discharge (no intrapolation possible)
        -> give them in [1/m^3, eV, K] (as Ts does not have too much of an influence, setting a constant default value makes sense and does not affect the results reliability too much)
        The parameter "plotting" determines, if measurement data, sputtering yields and erosion/deposition rates/layer thicknesses are plotted
        -> if True, plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
        
        Returns updated overviewTable and writes new .csv file with intrapolated values to 'results/calculationTablesNew/results_{discharge}.csv'.format(discharge=discharge)'''
    
    #create same timestamps for all LPs 
    # -> avoid [0.87, 2.87, 4.87, 0, 0] for lower0 vs [0.87, 2.87, 4.87, 6.87, 8.87] for lower1, instead apply [0.87, 2.87, 4.87, 6.87, 8.87] to all
    # -> guarantee that timesteps will be fitting and no negative timesteps occur and result in decreasing total layer thickness
    
    times = list(np.unique(overviewTable['time']))
    if 0 in times:
        times.remove(0)
    times.sort()
    times = [times] * len(LPindices)
    times = np.hstack(np.array(times)) #flattens times to get an one dimensional array

    for quantity_column, replacer in zip(['ne', 'Te', 'Ts'], defaultValues): #replacer is the value that is inserted if no data is present for the whole discharge
        quantity_list = []
        quantity_origin = []
        for index, quantity in enumerate(overviewTable[quantity_column]):
            if quantity == 0 or np.isnan(quantity): #intrapolation required
                quantity_origin.append('I')

                if str(index) in LPindices.T[1]: #if first value is missing

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

                else: #if intermediate value is missing
                    for i in range(index + 1, int(LPindices[-1][2]) + 1):
                        if overviewTable[quantity_column][i] != 0 and not np.isnan(overviewTable[quantity_column][i]):  #find existing value before hitting last index, append inbetween of last existing value and that value (according to number of missing values)
                            m = (overviewTable[quantity_column][i] - quantity_list[-1])/(overviewTable['time'][i] - overviewTable['time'][index - 1])
                            quantity_list.append(quantity_list[-1] + m * (overviewTable['time'][index] - overviewTable['time'][index - 1]))
                            break

                        elif str(i) in LPindices.T[2]: #last value is reached without finding an existing value, append last existing value
                            quantity_list.append(quantity_list[-1])
                            break
                            
            else:   #no intrapolation, just take existing value
                quantity_list.append(quantity)
                quantity_origin.append('M')

        if quantity_column == 'ne':
            ne_list = np.array(np.array_split(np.array(quantity_list), len(LPindices))) #inserted np.array() around quantity_list, in case there are any problems, that could be why
            origin_ne = quantity_origin
        elif quantity_column == 'Te':
            Te_list = np.array(np.array_split(np.array(quantity_list), len(LPindices)))
            origin_Te = quantity_origin
        else:
            Ts_list = np.array(np.array_split(np.array(quantity_list), len(LPindices)))
            origin_Ts = quantity_origin
    #changed1####################
    if len(times) != 0:
        if len(times) > 1:
            timesteps = [times[0] + (times[1] - times[0])/2]#timesteps = [dt[0]]
            for j in range(1, len(times[:int(LPindices[0][2]) + 1]) - 1):#range(1, len(dt)):
                timesteps.append((times[j] - times[j - 1])/2 + (times[j + 1] - times[j])/2)#timesteps.append(dt[j] - dt[j - 1])
            timesteps.append((times[-1] - times[-2])/2)
        else:
            timesteps = [times[0]]
    else:
        timesteps = []

    timesteps = np.array(timesteps)

    #get time steps
    #timesteps = [times[0]] 
    #for j in range(1, len(times[:int(LPindices[0][2]) + 1])):
    #    timesteps.append(times[j] - times[j - 1])
    #timesteps = np.array(timesteps)
    #changed1####################


    #now calculate sputtering yields
    Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = [], [], [], [], [], [], [], [], []

    for T_e, T_i, T_s, n_e, LPindex in zip(Te_list, Te_list, Ts_list, ne_list, LPindices):
        return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, LP_zeta[LPindex[0][5:]], m_i, f_i, ions, k, n_target)
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
            
            #plotting 
            if plotting==True:   
                safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=LPindex[0][:5], position=LPindex[0][5:])
                #changed1############# remove timesteps
                plot.plotOverview(n_e, T_e, T_s, return_erosion[0], return_erosion[3], return_erosion[4], 
                                  return_erosion[5], return_erosion[7], return_erosion[8], return_erosion[10], times[:int(LPindices[0][2]) + 1], timesteps[:int(LPindices[0][2]) + 1], safe)
                #changed1#############

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

    #saves measured values and calculated quantities in .csv file
    overviewTable = pd.DataFrame(overviewTable) #missing values in the table are nan values
    safe = 'results/calculationTablesNew/results_{discharge}.csv'.format(discharge=discharge)
    overviewTable.to_csv(safe, sep=';')
    
    return overviewTable

#######################################################################################################################################################################
def calculateTotalErodedLayerThicknessOneDischarge(discharge: str, 
                                                   duration: int|float,
                                                   overviewTable: pd.DataFrame,  
                                                   alpha: int|float, 
                                                   LP_zeta: list[int|float],
                                                   m_i: list[int|float], 
                                                   f_i: list[int|float], 
                                                   ions: list[str], 
                                                   k: int|float, 
                                                   n_target: int|float,
                                                   defaultValues: list[int|float] =[np.nan, np.nan, 320], 
                                                   intrapolated: bool =False,
                                                   plotting: bool =False) -> list[list[str, float, float]]:
    ''' This function calculates total erosion/deposition layer thickness for all LPs of a discharge given by its ID "discharge" with duration "duration" in [s] 
        -> adding erosion layer thickness up to last LP measurement to erosion layer thickness from last LP measurement to end of discharge
        "overviewTable" is initially Dataframe with keys 'LangmuirProbe', 'time', ne, ... as in 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
        -> overviewTable is by default not intrapolated ("intrapolated"=False), so intrapolation happens internally
        Incident angle of the ions  "alpha" is given in [rad]
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The ion masses "m_i" are in [kg], the ion concentrations "f_i", and ion names "ions" should have the same length as "m_i"
        The Boltzmann constant "k" must be in [eV/K]
        The atomic target density "n_target" should be provided in [1/m^3]
        The default values given for [ne, Te, Ts] in "defaultValues" determine which value is inserted if no data exists at a certain for the whole discharge (no intrapolation possible)
        -> give them in [1/m^3, eV, K] (as Ts does not have too much of an influence, setting a constant default value makes sense and does not affect the results reliability too much)
        The parameter "plotting" determines, if measurement data, sputtering yields and erosion/deposition rates/layer thicknesses are plotted
        -> if True, plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position) 
         
        Returns 2D list of structure [[LP1, erosionLayer1, depositionLayer1], [LP2, erosionLayer2, depositionLayer2], ...]'''
    #get first and last index of each LP in overviewTable, LPindices is 2D array with [[LP1, firstIndex1, lastindex1], [LP2, firstindex2, last index2], ...] -> ALL ELEMENTS ARE STRINGS
    LPindices = findIndexLP(overviewTable) 

    #intrapolate overviewTable in required
    if intrapolated == False:
        overviewTable = intrapolateMissingValues(discharge, overviewTable, LPindices, alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, plotting)
    
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
        if lastTimeIndex == -1:
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
            #changed1#############
            if lastTimeIndex > 0:
                erosionDuringLP = overviewTable['totalErodedLayerThickness'][lastTimeIndexCounter - 1] + overviewTable['erosionRate'][lastTimeIndexCounter] * (lastTime - overviewTable['time'][lastTimeIndex - 1])/2 #total eroded layer thickness for time interval until last LP measurement
                depositionDuringLP = overviewTable['totalDepositedLayerThickness'][lastTimeIndexCounter - 1] + overviewTable['depositionRate'][lastTimeIndexCounter] * (lastTime - overviewTable['time'][lastTimeIndex - 1])/2#total eroded layer thickness for time interval until last LP measurement
            else:
                erosionDuringLP = overviewTable['erosionRate'][lastTimeIndexCounter] * (lastTime) #total eroded layer thickness for time interval until last LP measurement
                depositionDuringLP = overviewTable['depositionRate'][lastTimeIndexCounter] * (lastTime)#total eroded layer thickness for time interval until last LP measurement
                
        erosionAfterLP = overviewTable['erosionRate'][lastTimeIndexCounter] * (duration - lastTime) #overviewTable['time'][lastTimeIndex]) #total eroded layer thickness for time interval after last LP measurement, multiply last erosion rate with time step till end of discharge
        depositionAfterLP = overviewTable['depositionRate'][lastTimeIndexCounter] * (duration - lastTime) #overviewTable['time'][lastTimeIndex]) #total eroded layer thickness for time interval after last LP measurement, multiply last erosion rate with time step till end of discharge
            #changed1#############
        
        erosion.append([LP[0], erosionDuringLP + erosionAfterLP, depositionDuringLP + depositionAfterLP])
    
    return erosion

#######################################################################################################################################################################
def calculateTotalErodedLayerThicknessSeveralDischarges(config: str, 
                                                        discharges: list[str|float], 
                                                        durations: list[int|float], 
                                                        overviewTables: list[str], 
                                                        alpha: int|float, 
                                                        LP_zeta: list[int|float],
                                                        m_i: list[int|float], 
                                                        f_i: list[int|float], 
                                                        ions: list[str], 
                                                        k: int|float, 
                                                        n_target: int|float, 
                                                        defaultValues: list[int|float] =[np.nan, np.nan, 320], 
                                                        intrapolated: bool =False, 
                                                        plotting: bool =False) -> pd.DataFrame:
    ''' This function calculates total erosion/deposition layer thickness for all LPs of all discharges of configuration "config" given by their IDs "discharges" with their durations "durations" in [s]
        -> adding erosion layer thickness up to last LP measurement to erosion layer thickness from last LP measurement to end of discharge and sum them up for all discharges of the configuration
        "overviewTables" is list of links following this structure 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
        -> leads to .csv files holding a Dataframe with keys 'LangmuirProbe', 'time', 'ne', ...
        -> overviewTable is by default not intrapolated ("intrapolated"=False), so intrapolation happens internally
        Incident angle of the ions  "alpha" is given in [rad]
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The ion masses "m_i" are in [kg], the ion concentrations "f_i", and ion names "ions" should have the same length as "m_i"
        The Boltzmann constant "k" must be in [eV/K]
        The atomic target density "n_target" should be provided in [1/m^3]
        The default values given for [ne, Te, Ts] in "defaultValues" determine which value is inserted if no data exists at a certain for the whole discharge (no intrapolation possible)
        -> give them in [1/m^3, eV, K] (as Ts does not have too much of an influence, setting a constant default value makes sense and does not affect the results reliability too much)
        The parameter "plotting" determines, if measurement data, sputtering yields and erosion/deposition rates/layer thicknesses are plotted
        -> if True, plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position) 
         
        Returns pd.DataFrame that is also saved as .csv file under 'results/erosionMeasuredConfig/totalErosionAtPosition_{config}.csv'.format(config=config)
        -> lists total erosion and deposition layer thicknesses for each discharge (with any data available) at each position
        -> holds keys like 'discharge', 'duration', 'lower0_erosion', 'lower0_deposition' '''
    LP_lower0, LP_lower1, LP_lower2, LP_lower3, LP_lower4, LP_lower5, LP_lower6, LP_lower7, LP_lower8, LP_lower9, LP_lower10, LP_lower11, LP_lower12, LP_lower13, LP_lower14, LP_lower15, LP_lower16, LP_lower17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] 
    LP_upper0, LP_upper1, LP_upper2, LP_upper3, LP_upper4, LP_upper5, LP_upper6, LP_upper7, LP_upper8, LP_upper9, LP_upper10, LP_upper11, LP_upper12, LP_upper13, LP_upper14, LP_upper15, LP_upper16, LP_upper17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    dischargeList, durationList = [], []
    for discharge, duration, overviewTable in zip(discharges, durations, overviewTables):
        discharge = str(discharge)
        lower = [False] * 18
        upper = [False] * 18
 
        if not os.path.isfile(overviewTable):
            continue
        
        print('processing ' + config + ' ' + discharge)
        dischargeList.append(discharge)
        durationList.append(duration)

        if not intrapolated:
            overviewTable = pd.read_csv(overviewTable, sep=';')
            erosion = calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, intrapolated, plotting)
        else:
            if not os.path.isfile('results/calculationTablesNew/' + overviewTable.split('/')[-1]):
                overviewTable = pd.read_csv(overviewTable, sep=';')
                erosion = calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, intrapolated=False, plotting=plotting)
            else:
                overviewTable = pd.read_csv('results/calculationTablesNew/' + overviewTable.split('/')[-1], sep = ';')
                erosion = calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, intrapolated, plotting)
        
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
def calculateTotalErodedLayerThicknessWholeCampaignPerConfig(config: str, 
                                                             campaign: str ='',
                                                             erosionTable: str ='results/erosionMeasuredConfig/totalErosionAtPosition_', 
                                                             dischargeOverview: str= 'results/configurations/dischargeList_') -> str:
    ''' This function calculates total erosion/deposition layer thickness for all LPs that occurred during discharges in configuration "config" 
        -> one value for erosion/deposition layer thickness of each LP
        -> known total erosion/deposition layer thicknesses are extrapolated to whole plasma time t_total by multiplying known values with t_total/t_known 
        "campaign" defines the campaign being looked at ('OP22', 'OP23', or '' for both)
        "erosionTable" is the structure of the path where a .csv file is saved
        -> that file contains a DataFrame with the total erosion/deposition layer thicknesses of all recorded discharges of that configuration (keys like 'discharge', 'duration' 'lower0_erosion', 'lower0_deposition')
        -> such a file can be created by running "calculateTotalErodedLayerThicknessSeveralDischarges"
        "dischargeOverview" is the path structure to a .csv file containing a DataFrame that lists all discharges in that configration no matter if data is available (keys 'dischargeID', 'duration',...)
        -> such a file can be created by running "ReadArchieveDB.readAllShotNumbersFromLogbook

        Returns string that everything has been calculated successfully
        -> saves the calculation results in .csv file under 'results/erosionExtrapolatedConfig/totalErosionAtPositionWholeCampaign_{config}.csv'.format(config=config)'''
    if campaign == '':
        campaign = 'OP223'
    if not os.path.isfile(erosionTable + '{config}.csv'.format(config=config)) or not os.path.isfile(dischargeOverview + campaign + '_{config}.csv'.format(config=config)):
        return 'file missing for ' + config
    
    erosionTable = pd.read_csv(erosionTable + '{config}.csv'.format(config=config), sep=';')
    dischargeOverview = pd.read_csv(dischargeOverview + campaign + '_{config}.csv'.format(config=config), sep=';')

    LP_erosion, LP_deposition, erosion_knownData, deposition_knownData, time_knownErosion, time_knownDeposition, erosion_total, deposition_total = [], [], [], [], [], [], [], []
    totalTime = np.nansum(np.array(dischargeOverview['duration']))
    for key in erosionTable.keys():
        if 'lower' in key or 'upper' in key:
            if 'erosion' in key:
                LP_erosion.append(key.split('_')[0])
                erosion_knownData.append(np.nansum(erosionTable[key]))
                nan = np.array([np.isnan(i) for i in erosionTable[key]])
                if not list(nan):
                    time_knownErosion.append(0)
                    erosion_total.append(0)
                else:
                    time_knownErosion.append(np.nansum(np.array(erosionTable['duration'])[~nan]))
                    if time_knownErosion[-1] == 0:
                        erosion_total.append(0)
                    else:
                        erosion_total.append(erosion_knownData[-1] * totalTime/time_knownErosion[-1])
            else:
                LP_deposition.append(key.split('_')[0])
                deposition_knownData.append(np.nansum(erosionTable[key]))
                nan = np.array([np.isnan(i) for i in erosionTable[key]])
                if not list(nan):
                    time_knownDeposition.append(0)
                    deposition_total.append(0)
                else:
                    time_knownDeposition.append(np.nansum(np.array(erosionTable['duration'])[~nan]))
                    if time_knownDeposition == 0:
                        deposition_total.append(0)
                    else:
                        deposition_total.append(deposition_knownData[-1] * totalTime/time_knownDeposition[-1])
    #print(len(LP), len(time_knownErosion), len(time_knownDeposition), len(erosion_knownData), len(deposition_knownData), len(erosion_total), len(deposition_total))
    if LP_erosion==LP_deposition:
        erosion = pd.DataFrame({'LP': LP_erosion,
                                'duration_knownErosion (s)': time_knownErosion, 
                                'erosion_known (m)': erosion_knownData, 
                                'duration_knownDeposition (s)': time_knownDeposition, 
                                'deposition_known (m)': deposition_knownData, 
                                'duration_total (s)': [totalTime] * len(LP_erosion), 
                                'erosion_total (m)': erosion_total,
                                'deposition_total (m)': deposition_total,
                                'netErosion_total (m)': np.array(erosion_total) - np.array(deposition_total)})
    else:
        #that is actually bullshit, if the two lists are not identical, they must be sorted before calculating anything with them
        erosion = pd.DataFrame({'LP': LP_erosion,
                                'LP_deposition': LP_deposition, 
                                'duration_knownErosion (s)': time_knownErosion, 
                                'erosion_known (m)': erosion_knownData, 
                                'duration_knownDeposition (s)': time_knownDeposition, 
                                'deposition_known (m)': deposition_knownData, 
                                'duration_total (s)': [totalTime] * len(LP_erosion), 
                                'erosion_total (m)': erosion_total,
                                'deposition_total (m)': deposition_total,
                                'netErosion_total (m)': np.array(erosion_total) - np.array(deposition_total)})
    erosion.to_csv('results/erosionExtrapolatedConfig/totalErosionAtPositionWholeCampaign_{config}.csv'.format(config=config), sep=';')
    return 'succesfully calculated total erosion in ' + config

#######################################################################################################################################################################        
def calculateTotalErodedLayerThicknessWholeCampaign(configurationList: list[str], 
                                                    LP_position: list[int|float], 
                                                    campaign: str ='',
                                                    T_default: str ='',
                                                    totalErosionFiles: str ='results/erosionExtrapolatedConfig/totalErosionAtPositionWholeCampaign_',
                                                    configurationOverview: str ='inputFiles/Overview4.csv') -> None:
    ''' This function calculates total erosion/deposition layer thickness for all LPs that occurred during the whole campaign in all configurations listed in "configurationList"
        -> if for a configuration no data exists at a LP position, that can not be taken into account)
        -> one value for erosion/deposition layer thickness of each LP by adding up the values of each configuration 
        The distance of each langmuir probe from the pumping gap is given in "LP_position" in [m], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        "campaign" is introducing which campaigns are investigated 
        -> '' means all available campaigns, otherwise type 'OP22' or 'OP23'
        "T_default" defines treatment of missing T_s values (if '' this means T_s is set to 320K)
        "totalErosionFiles" is the structure of the path where a .csv file is saved
        -> that file contains a DataFrame with the total erosion/deposition layer thicknesses of a configuration (keys like 'LP', 'erosion_total' 'netErosion_total', 'deposition_total')
        -> such a file can be created by running "calculateTotalErodedLayerThicknessWholeCampaignPerConfig"
        "configurationOverview" is an overview file containing information about the iota of each configuration ever released 

        Returns nothing but saves the calculation results in .csv file under 'results/erosionFullCampaign/totalErosionWholeCampaignAllPositions.csv' '''
    config_missing = [] #configurations, which are released but never used (no discharges)

    #for not extraplated values -> missing configurations are missing##############
    erosion_position = np.array([0.] * 36)
    deposition_position = np.array([0.] * 36)
    dataOverview = pd.DataFrame({'LP': ['lower0', 'lower1', 'lower2', 'lower3', 'lower4', 'lower5', 'lower6', 'lower7', 'lower8', 'lower9', 'lower10', 'lower11', 'lower12', 'lower13', 'lower14', 'lower15', 'lower16', 'lower17', 
                             'upper0', 'upper1', 'upper2', 'upper3', 'upper4', 'upper5', 'upper6', 'upper7', 'upper8', 'upper9', 'upper10', 'upper11', 'upper12', 'upper13', 'upper14', 'upper15', 'upper16', 'upper17']})
    
    for config in configurationList:
        if not os.path.isfile(totalErosionFiles + '{config}.csv'.format(config=config)):
            print('file missing for ' + config)
            config_missing.append(config)
            continue
        
        else:
            erosion = pd.read_csv(totalErosionFiles + '{config}.csv'.format(config=config), sep=';')
            if list(erosion['LP']) == list(dataOverview['LP']):
                print('everything is all right')
            else:
                print('we have got a problem') #in that case handling of the below has to be changed
            erosion_position = np.hstack(np.nansum(np.dstack((np.array(erosion['erosion_total (m)']), erosion_position)), 2))
            deposition_position = np.hstack(np.nansum(np.dstack((np.array(erosion['deposition_total (m)']), deposition_position)), 2))
            dataOverview[config + '_erosion'] = np.array(erosion['erosion_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            dataOverview[config + '_deposition'] = np.array(erosion['deposition_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            dataOverview[config + '_netErosion'] = np.array(erosion['netErosion_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            dataOverview[config + '_duration'] = np.array(erosion['duration_total (s)'])

    #plot low iota, not extrapolated
    plt.plot(LP_position[:14], 0 - erosion_position[:14], 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[:14], deposition_position[:14], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[:14], 0 - erosion_position[:14] + deposition_position[:14],'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[:14], 0 - erosion_position[18:32], 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[:14], deposition_position[18:32], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[:14], 0 - erosion_position[18:32] + deposition_position[18:32], 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('total layer thickness (m)')
    plt.savefig('results/erosionFullCampaign/{campaign}{Ts}totalErosionWholeCampaignAllPositionsLowIota.png'.format(campaign=campaign, Ts='_'+T_default+'_'), bbox_inches='tight')
    plt.show()
    plt.close()

    #plot high iota, not extrapolated
    plt.plot(LP_position[14:], 0 - erosion_position[14:18], 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[14:], deposition_position[14:18], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[14:], 0 - erosion_position[14:18] + deposition_position[14:18], 'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[14:], 0 - erosion_position[32:], 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[14:], deposition_position[32:], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[14:], 0 - erosion_position[32:] + deposition_position[32:], 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('total layer thickness (m)')
    plt.savefig('results/erosionFullCampaign/{campaign}{Ts}totalErosionWholeCampaignAllPositionsHighIota.png'.format(campaign=campaign, Ts='_'+T_default+'_'), bbox_inches='tight')
    plt.show()
    plt.close()

    #end of: for not extraplated values -> missing configurations are missing##############
    
    #for extraplated values -> extrapolate values for missing configurations with same AAA in AAA+-XXXX ###############
    configOverview = pd.read_csv(configurationOverview, sep=';')
    config_short = []
    config_iota = []
    iota_problem = []
    for config in configurationList:
        if config not in config_missing:
            if config[:3] + config[6] not in config_short:
                config_short.append(config[:3] + config[6])
                i = list(configOverview['configuration']).index(config)
                config_iota.append(configOverview['iota'][i])
            else:
                i = config_short.index(config[:3] + config[6])
                if config_iota[i] != configOverview['iota'][list(configOverview['configuration']).index(config)]:
                    iota_problem.append(config_short[i])
    print(iota_problem)
    dataOverview2 = pd.DataFrame({'LP': dataOverview['LP']})

    for config_3 in config_short:
        duration_total_config = 0
        duration_erosion_known_config = []
        duration_deposition_known_config = []
        erosion_known_config = []
        erosion_total_config = []
        deposition_known_config = []
        deposition_total_config = []
        config_column = []

        for LP in range(len(dataOverview['LP'])):
            erosion = 0
            deposition = 0
            duration_deposition = 0
            duration_erosion = 0

            for config in configurationList:
                if config[:3] + config[6] != config_3:
                    continue
                if config in config_missing:
                    continue
                
                if list(dataOverview[config + '_erosion'])[LP] != 0 and not np.isnan(list(dataOverview[config + '_erosion'])[LP]):
                    erosion += list(dataOverview[config + '_erosion'])[LP]
                    duration_erosion += list(dataOverview[config + '_duration'])[LP]
                if list(dataOverview[config + '_deposition'])[LP] != 0 and not np.isnan(list(dataOverview[config + '_deposition'])[LP]):
                    deposition += list(dataOverview[config + '_deposition'])[LP]
                    duration_deposition += list(dataOverview[config + '_duration'])[LP]

                if LP == 0:
                    duration_total_config += list(dataOverview[config + '_duration'])[0]
                    config_column.append(config)

            erosion_known_config.append(erosion)
            deposition_known_config.append(deposition)
            duration_erosion_known_config.append(duration_erosion)     
            duration_deposition_known_config.append(duration_deposition)  
            if duration_erosion != 0:
                erosion_total_config.append(erosion * duration_total_config/duration_erosion)   
            else: 
                erosion_total_config.append(0)
            if duration_deposition != 0:
                deposition_total_config.append(deposition * duration_total_config/duration_deposition)   
            else:
                deposition_total_config.append(0)


        #copy dataframe dataOverview but write extrapolated values to it
        for config in config_column:
            erosion, deposition, netErosion = [], [], []

            for LP in range(len(dataOverview['LP'])):
                if list(dataOverview[config + '_erosion'])[LP] == 0 or np.isnan(list(dataOverview[config + '_erosion'])[LP]):
                    erosion.append(erosion_total_config[LP] * list(dataOverview[config + '_duration'])[LP]/duration_total_config)
                else:
                    erosion.append(dataOverview[config + '_erosion'][LP])

                if list(dataOverview[config + '_deposition'])[LP] == 0 or np.isnan(list(dataOverview[config + '_deposition'])[LP]):
                    deposition.append(deposition_total_config[LP] * list(dataOverview[config + '_duration'])[LP]/duration_total_config)
                else:
                    deposition.append(list(dataOverview[config + '_deposition'])[LP])

                netErosion.append(erosion[-1] - deposition[-1])

            dataOverview2[config + '_erosion'] = erosion
            dataOverview2[config + '_deposition'] = deposition
            dataOverview2[config + '_netErosion'] = netErosion
            dataOverview2[config + '_duration'] = dataOverview[config + '_duration']
    
    #end of: for extraplated values -> extrapolate values for missing configurations with same AAA in AAA+-XXXX ###############

    #for extraplated values -> extrapolate values for missing configurations generally##################
    erosion_total, deposition_total = np.array([0.] * 36), np.array([0.] * 36)
    for iota in ['low', 'standard', 'high']:
        duration_total = 0
        erosion_total_known = []
        deposition_total_known = []
        duration_erosion_known = []
        duration_deposition_known = []
        erosion_total_iota = []
        deposition_total_iota = []

        for LP in range(len(dataOverview2['LP'])):
            erosion = 0
            deposition = 0
            duration_deposition = 0
            duration_erosion = 0

            for config in configurationList:
                if config in config_missing:
                    continue
                if configOverview['iota'][list(configOverview['configuration']).index(config)] != iota:
                    continue
                if list(dataOverview2[config + '_erosion'])[LP] != 0 and not np.isnan(list(dataOverview2[config + '_erosion'])[LP]):
                    erosion += list(dataOverview2[config + '_erosion'])[LP]
                    duration_erosion += list(dataOverview2[config + '_duration'])[LP]
                if list(dataOverview2[config + '_deposition'])[LP] != 0 and not np.isnan(list(dataOverview2[config + '_deposition'])[LP]):
                    deposition += list(dataOverview2[config + '_deposition'])[LP]
                    duration_deposition += list(dataOverview2[config + '_duration'])[LP]
                if LP == 0:
                    duration_total += list(dataOverview2[config + '_duration'])[0]

            erosion_total_known.append(erosion)
            deposition_total_known.append(deposition)
            duration_erosion_known.append(duration_erosion)     
            duration_deposition_known.append(duration_deposition)  
            if duration_erosion != 0:
                erosion_total_iota.append(erosion * duration_total/duration_erosion)   
            else: 
                erosion_total_iota.append(0)
            if duration_deposition != 0:
                deposition_total_iota.append(deposition * duration_total/duration_deposition)   
            else:
                deposition_total_iota.append(0)

        #copy dataframe dataOverview but write extrapolated values to it
        dataOverview3 = pd.DataFrame({'LP': dataOverview['LP']})
        for config in configurationList:
            if config in config_missing:
                    continue
            erosion, deposition, netErosion = [], [], []

            for LP in range(len(dataOverview2['LP'])):
                if list(dataOverview2[config + '_erosion'])[LP] == 0 or np.isnan(list(dataOverview2[config + '_erosion'])[LP]):
                    erosion.append(erosion_total_iota[LP] * list(dataOverview2[config + '_duration'])[LP]/duration_total)
                else:
                    erosion.append(dataOverview2[config + '_erosion'][LP])

                if list(dataOverview2[config + '_deposition'])[LP] == 0 or np.isnan(list(dataOverview2[config + '_deposition'])[LP]):
                    deposition.append(deposition_total_iota[LP] * list(dataOverview2[config + '_duration'])[LP]/duration_total)
                else:
                    deposition.append(list(dataOverview2[config + '_deposition'])[LP])

                netErosion.append(erosion[-1] - deposition[-1])

            dataOverview3[config + '_erosion'] = erosion
            dataOverview3[config + '_deposition'] = deposition
            dataOverview3[config + '_netErosion'] = netErosion
            dataOverview3[config + '_duration'] = dataOverview2[config + '_duration']
        erosion_total = np.hstack(np.nansum(np.dstack((np.array(erosion_total_iota), erosion_total)), 2))
        deposition_total = np.hstack(np.nansum(np.dstack((np.array(deposition_total_iota), deposition_total)), 2))

    #erosion_total = np.array(erosion_total)
    #deposition_total = np.array(deposition_total)

    #plot low iota, extrapolated
    plt.plot(LP_position[:14], 0 - erosion_total[:14], 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[:14], deposition_total[:14], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[:14], 0 - erosion_total[:14] + deposition_total[:14],'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[:14], 0 - erosion_total[18:32], 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[:14], deposition_total[18:32], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[:14], 0 - erosion_total[18:32] + deposition_total[18:32], 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('total layer thickness (m)')
    plt.savefig('results/erosionFullCampaign/{campaign}{Ts}totalErosionWholeCampaignAllPositionsLowIotaExtrapolated.png'.format(campaign=campaign, Ts='_'+T_default+'_'), bbox_inches='tight')
    plt.show()
    plt.close()

    #plot high iota, extrapolated
    plt.plot(LP_position[14:], 0 - erosion_total[14:18], 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[14:], deposition_total[14:18], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[14:], 0 - erosion_total[14:18] + deposition_total[14:18], 'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[14:], 0 - erosion_total[32:], 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[14:], deposition_total[32:], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[14:], 0 - erosion_total[32:] + deposition_total[32:], 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('total layer thickness (m)')
    plt.savefig('results/erosionFullCampaign/{campaign}{Ts}totalErosionWholeCampaignAllPositionsHighIotaExtrapolated.png'.format(campaign=campaign, Ts='_'+T_default+'_'), bbox_inches='tight')
    plt.show()
    plt.close()

    #end of: for extraplated values -> extrapolate values for missing configurations generally##################

    dataOverview.to_csv('results/erosionFullCampaign/{campaign}{Ts}totalErosionWholeCampaignAllPositions.csv'.format(campaign=campaign, Ts='_'+T_default+'_'), sep=';')
    dataOverview3.to_csv('results/erosionFullCampaign/{campaign}{Ts}totalErosionWholeCampaignAllPositionsExtrapolated.csv'.format(campaign=campaign, Ts='_'+T_default+'_'), sep=';')

##################################################################################################################################################################
def calculateAverageQuantityPerConfiguration(quantity: str,
                                             config: str, 
                                             LP_position: list[int|float], 
                                             campaign: str ='',
                                             dischargeList: str ='results/configurations/dischargeList_') -> str|None:
    if campaign == '':
        campaign = 'OP223'
    if not os.path.isfile(dischargeList + campaign + '_{config}.csv'.format(config=config)):
        return 'file missing for ' + config
    
    dischargeOverview = pd.read_csv(dischargeList + campaign + '_{config}.csv'.format(config=config), sep=';')
    averageConfiguration = np.array([0.]*36)
    timeConfiguration = np.array([0.]*36)
    for discharge, duration, overviewTable in zip(dischargeOverview['dischargeID'], dischargeOverview['duration'], dischargeOverview['overviewTable']):
        discharge = str(discharge)
 
        if os.path.isfile('results/calculationTablesNew/' + overviewTable.split('/')[-1]):
            overviewTable = pd.read_csv('results/calculationTablesNew/' + overviewTable.split('/')[-1], sep = ';')
        #elif os.path.isfile(overviewTable):
        #    overviewTable = pd.read_csv(overviewTable, sep=';')
        else:
            continue
        times = list(np.unique(overviewTable['time']))
        LPlist, indices = list(np.unique(overviewTable['LangmuirProbe'], return_index=True))
        LPs = np.array(overviewTable['LangmuirProbe'])[np.sort(indices)]
        if 0 in times:
            times.remove(0)
        times.sort()
        #times = [times] * len(LPs)
        #times = np.hstack(np.array(times)) #flattens times to get an one dimensional array
    
        if len(times) != 0:
            if max(times) > duration:
                lastTimeIndex = [j > duration for j in times].index(True) - 1            
            else:
                lastTimeIndex = len(times) - 1

            if len(times[:lastTimeIndex + 1]) > 1:
                timesteps = [times[0] + (times[1] - times[0])/2]#timesteps = [dt[0]]
                for j in range(1, len(times[:lastTimeIndex + 1]) - 1):#range(1, len(dt)):
                    timesteps.append((times[j] - times[j - 1])/2 + (times[j + 1] - times[j])/2)#timesteps.append(dt[j] - dt[j - 1])
                if lastTimeIndex != len(times) - 1:
                    if times[lastTimeIndex] + (times[lastTimeIndex + 1] - times[lastTimeIndex])/2 > duration:
                        timesteps.append((times[lastTimeIndex] - times[lastTimeIndex - 1])/2 + duration - times[lastTimeIndex])
                    else:
                        timesteps.append((times[lastTimeIndex] - times[lastTimeIndex - 1])/2 + (times[lastTimeIndex + 1] - times[lastTimeIndex])/2)
                        timesteps.append(duration - times[lastTimeIndex] - (times[lastTimeIndex + 1] - times[lastTimeIndex])/2)
                else:
                    timesteps.append((times[lastTimeIndex] - times[lastTimeIndex - 1])/2 + duration - times[lastTimeIndex])
            elif len(times[:lastTimeIndex + 1]) == 0:
                lastTimeIndex == 0
                timesteps = [duration]
            else:
                if times[0] + (times[1] - times[0])/2 > duration:
                    timesteps = [duration]
                else:
                    timesteps = [times[0] + (times[1] - times[0])/2, duration - (times[0] + (times[1] - times[0])/2)]
        else:
            timesteps = []

        timesteps = np.array(timesteps)

        LPs = list(LPs)
        averageListDischarge = []
        timeDischarge = []
        for DUindex in ['lower', 'upper']:
            for LPindex in range(18):
                if DUindex + str(LPindex) in LPs:
                    counter = LPs.index(DUindex + str(LPindex))
                    averageDischarge = 0
                    duration_measurement = 0
                    for counter2, dt in enumerate(timesteps):
                        indexCounter = counter * len(timesteps) + counter2
                        if list(overviewTable[quantity])[indexCounter] != 0 and not np.isnan(list(overviewTable[quantity])[indexCounter]):
                            if quantity == 'Te':
                                if list(overviewTable[quantity])[indexCounter] != 320:
                                    averageDischarge += list(overviewTable[quantity])[indexCounter] * dt
                                    duration_measurement += dt
                                else:
                                    continue
                            else:
                                averageDischarge += list(overviewTable[quantity])[indexCounter] * dt
                                duration_measurement += dt
                        else:
                            continue
                    if duration_measurement != 0:
                        averageListDischarge.append(averageDischarge/duration_measurement)
                        timeDischarge.append(duration_measurement)
                    else:
                        averageListDischarge.append(0)
                        timeDischarge.append(0)
                else:
                    averageListDischarge.append(0)
                    timeDischarge.append(0)
        averageConfiguration = np.hstack(np.nansum(np.dstack((np.array(averageConfiguration), (np.array(averageListDischarge) * np.array(timeDischarge)))), 2))
        timeConfiguration = np.hstack(np.nansum(np.dstack((np.array(timeConfiguration), np.array(timeDischarge))), 2))
    averageConfiguration = averageConfiguration/timeConfiguration

    if not os.path.exists('results/averageQuantities/{quantity}/{config}'.format(quantity=quantity, config=config)):
        os.makedirs('results/averageQuantities/{quantity}/{config}'.format(quantity=quantity, config=config)) 

    plt.plot(LP_position[14:], averageConfiguration[14:18], 'b', label='lower divertor unit')
    plt.plot(LP_position[14:], averageConfiguration[32:], 'm', label='upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('average ' + quantity)
    plt.savefig('results/averageQuantities/{quantity}/{config}/{quantity}{campaign}{config}AverageAllPositionsHighIota.png'.format(quantity=quantity, campaign=campaign, config=config), bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(LP_position[:14], averageConfiguration[:14], 'b', label='lower divertor unit')
    plt.plot(LP_position[:14], averageConfiguration[18:32], 'm', label='upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('average ' + quantity)
    plt.savefig('results/averageQuantities/{quantity}/{config}/{quantity}{campaign}{config}AverageAllPositionsLowIota.png'.format(quantity=quantity, campaign=campaign, config=config), bbox_inches='tight')
    plt.show()
    plt.close()

    return averageConfiguration, timeConfiguration