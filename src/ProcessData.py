'''This file contains the functions neccessary for processing data from w7xArchieveDB/xdrive/downloaded files... and calculating erosion related quantities'''

import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

import src.SputteringYieldFunctions as calc
#import src.ReadArchieveDB as read
import src.PlotData as plot
from src.settings import k

#######################################################################################################################################################################
def subresults(m_i: list[int|float], f_i: list[int|float], ions: list[str], ne: int|float =1e+19, Te: int|float =15, Ts: int|float =320, alpha: int|float =np.deg2rad(40), zeta: int|float =np.deg2rad(2), n_C: int|float =9.5*1e+28, t: int|float =15) -> None:
    ''' This function prints out the subresults of the calculation prozess for one parameter set ne (electron density), Te (electron temperature), Ts (surface temperature), alpha (ion incidence angle), zeta (mag. filed line incident angle), n_C (atomic target density), t (duration of discharge)
        Subresults include flux densities, physical and chemical sputtering yields, erosion and deposition layer thicknesses and net eroded layer thickness of all ions
        For all ions, their masses "m_i" in (kg), their concentrations "f_i" and names/chemical symbols "ions" must be given
        ne and n_C in (m^-3), Te in (eV), Ts in (K), alpha and zeta in (rad), t in (s)'''
    total = 0
    for mi, fi, ion in zip(m_i, f_i, ions):
        flux = calc.calculateFluxIncidentIon(zeta, Te/k, Te/k, mi, ne, fi)
        print(f'flux density (1/m^2 s) for {ion}: ', flux)

        Ys = calc.calculateTotalErosionYield(ion, Te, 'C', alpha, Ts, flux, n_C, True)
        print(f'Physical sputtering yield for {ion}: ', Ys[0])
        print(f'Chemical erosion yield for {ion}: ', Ys[1])

        erosion = t * flux * (Ys[0] + Ys[1])/n_C
        print(f'Eroded layer thickness (m) for {ion}: ', erosion)
        
        if ion == 'C':
            deposition = t * flux/n_C
        else:
            deposition = 0
        print(f'Deposited layer thickness (m) for {ion}: ', deposition)

        print(f'Net eroded layer thickness (m) for {ion}: ', deposition - erosion)

        total += deposition - erosion
        
        print()

    print(f'Total net eroded layer thickness (m) for all ions: ', total)

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
                   sne_lower: list[list[int|float]],
                   sne_upper: list[list[int|float]], 
                   sTe_lower: list[list[int|float]], 
                   sTe_upper: list[list[int|float]], 
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
    ne, Te, Ts, sne, sTe = [], [], [], [], []

    divertorUnit = 'lower'

    #treat each divertor unit separately
    for ne_divertorUnit, Te_divertorUnit, sne_divertorUnit, sTe_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, LP_indices, t_divertorUnit in zip([ne_lower, ne_upper], [Te_lower, Te_upper], [sne_lower, sne_upper], [sTe_lower, sTe_upper], [Te_lower, Te_upper], [Ts_lower, Ts_upper], [index_lower, index_upper], [t_lower, t_upper]):
        
        Y_0_divertorUnit, Y_1_divertorUnit, Y_2_divertorUnit, Y_3_divertorUnit, Y_4_divertorUnit = [], [], [], [], []
        erosionRate_dt_divertorUnit, erodedLayerThickness_dt_divertorUnit, erodedLayerThickness_divertorUnit = [], [], []
        depositionRate_dt_divertorUnit, depositedLayerThickness_dt_divertorUnit, depositedLayerThickness_divertorUnit = [], [], []

        #treat each langmuir probe position separately
        for n_e, T_e, sn_e, sT_e, T_i, T_s, LP_index, dt in zip(ne_divertorUnit, Te_divertorUnit, sne_divertorUnit, sTe_divertorUnit, Ti_divertorUnit, Ts_divertorUnit, LP_indices, t_divertorUnit):
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
            sne.append(sn_e)
            Te.append(T_e)
            sTe.append(sT_e)
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
                    'sne':list(itertools.chain.from_iterable(sne)),
                    'Te':list(itertools.chain.from_iterable(Te)),
                    'sTe':list(itertools.chain.from_iterable(sTe)),
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

    for quantity_column, replacer in zip(['ne', 'Te', 'Ts', 'sne', 'sTe'], defaultValues): #replacer is the value that is inserted if no data is present for the whole discharge
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
        elif quantity_column == 'Ts':
            Ts_list = np.array(np.array_split(np.array(quantity_list), len(LPindices)))
            origin_Ts = quantity_origin
        elif quantity_column == 'sne':
            sne_list = np.array(np.array_split(np.array(quantity_list), len(LPindices))) #inserted np.array() around quantity_list, in case there are any problems, that could be why
            origin_sne = quantity_origin
        elif quantity_column == 'sTe':
            sTe_list = np.array(np.array_split(np.array(quantity_list), len(LPindices)))
            origin_sTe = quantity_origin
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
        return_erosion = calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, LP_zeta[int(LPindex[0][5:])], m_i, f_i, ions, k, n_target)
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
    overviewTable['sne'] = np.hstack(sne_list)
    overviewTable['origin_sne'] = origin_sne
    overviewTable['sTe'] = np.hstack(sTe_list)
    overviewTable['origin_sTe'] = origin_sTe
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
                                                        plotting: bool =False,
                                                        excluded: list[str] =[]) -> pd.DataFrame:
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
        "excluded" is a list of discharges that should not be taken into account

        Returns pd.DataFrame that is also saved as .csv file under 'results/erosionMeasuredConfig/totalErosionAtPosition_{config}.csv'.format(config=config)
        -> lists total erosion and deposition layer thicknesses for each discharge (with any data available) at each position
        -> holds keys like 'discharge', 'duration', 'lower0_erosion', 'lower0_deposition' '''
    LP_lower0, LP_lower1, LP_lower2, LP_lower3, LP_lower4, LP_lower5, LP_lower6, LP_lower7, LP_lower8, LP_lower9, LP_lower10, LP_lower11, LP_lower12, LP_lower13, LP_lower14, LP_lower15, LP_lower16, LP_lower17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] 
    LP_upper0, LP_upper1, LP_upper2, LP_upper3, LP_upper4, LP_upper5, LP_upper6, LP_upper7, LP_upper8, LP_upper9, LP_upper10, LP_upper11, LP_upper12, LP_upper13, LP_upper14, LP_upper15, LP_upper16, LP_upper17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    dischargeList, durationList = [], []
    for discharge, duration, overviewTable in zip(discharges, durations, overviewTables):
        discharge = str(discharge)

        if discharge[-2] == '.':
            discharge = discharge + '00'
        elif discharge[-3] == '.':
            discharge = discharge + '0'

        if discharge in excluded:
            print(discharge)
            continue
        
        lower = [False] * 18
        upper = [False] * 18
 
        if not os.path.isfile(overviewTable):
            continue
        
        print('processing ' + config + ' ' + discharge)
        dischargeList.append(discharge)
        durationList.append(duration)

        if not intrapolated or discharge.endswith('0'):
            overviewTable = pd.read_csv(overviewTable, sep=';')
            erosion = calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, False, plotting)
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
def calculateTotalErodedLayerThicknessWholeCampaign(n_target: int|float,
                                                    m_i: list[int|float], f_i: list[int|float], alpha: int|float, zeta: list[int|float],
                                                    configurationList: list[str], 
                                                    configurationChosen: str,
                                                    LP_position: list[int|float], 
                                                    campaign: str ='',
                                                    T_default: str ='',
                                                    errors: bool = False,
                                                    totalErosionFiles: str ='results/erosionExtrapolatedConfig/totalErosionAtPositionWholeCampaign_',
                                                    configurationOverview: str ='inputFiles/Overview4.csv') -> int|float:
    ''' This function calculates total erosion/deposition layer thickness for all LPs that occurred during the whole campaign in all/some configurations listed in "configurationList"
        -> if for a configuration no data exists at a LP position, that can not be taken into account
        -> one value for erosion/deposition layer thickness of each LP by adding up the values of each configuration 
        -> "configurationChosen" = 'all' means, that all configurations are considered, 'EIM' means only standard configuration is considered, ...
        The distance of each langmuir probe from the pumping gap is given in "LP_position" in [m], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        "campaign" is introducing which campaigns are investigated 
        -> '' means all available campaigns, otherwise type 'OP22' or 'OP23'
        "T_default" defines treatment of missing T_s values (if '' this means T_s is set to 320K)
        "totalErosionFiles" is the structure of the path where a .csv file is saved
        -> that file contains a DataFrame with the total erosion/deposition layer thicknesses of a configuration (keys like 'LP', 'erosion_total' 'netErosion_total', 'deposition_total')
        -> such a file can be created by running "calculateTotalErodedLayerThicknessWholeCampaignPerConfig"
        "configurationOverview" is an overview file containing information about the iota of each configuration ever released 
        "n_target" is the atomic density of the divertor target
        Returns mass of eroded material in g and saves the calculation results in .csv file under 'results/erosionFullCampaign/totalErosionWholeCampaignAllPositions.csv' '''
    config_missing = [] #configurations, which are released but never used (no discharges)

    #for not extraplated values -> missing configurations are missing##############
    erosion_position = np.array([0.] * 36)
    deposition_position = np.array([0.] * 36)
    duration_ero = np.array([0.] * 36)
    duration_depo = np.array([0.] * 36)

    dataOverview = pd.DataFrame({'LP': ['lower0', 'lower1', 'lower2', 'lower3', 'lower4', 'lower5', 'lower6', 'lower7', 'lower8', 'lower9', 'lower10', 'lower11', 'lower12', 'lower13', 'lower14', 'lower15', 'lower16', 'lower17', 
                             'upper0', 'upper1', 'upper2', 'upper3', 'upper4', 'upper5', 'upper6', 'upper7', 'upper8', 'upper9', 'upper10', 'upper11', 'upper12', 'upper13', 'upper14', 'upper15', 'upper16', 'upper17']})
    
    for config in configurationList:
        if not os.path.isfile(totalErosionFiles + '{config}.csv'.format(config=config)):
            print('file missing for ' + config)
            config_missing.append(config)
            continue

        if configurationChosen != 'all' and not config.startswith(configurationChosen):
            print(config + ' is not matching configuration ' + configurationChosen)
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

            ero, depo = [], []
            for LP in range(len(erosion_position)):
                if erosion['erosion_total (m)'][LP] != 0:
                    ero.append(np.array(erosion['duration_total (s)'])[LP])
                else:
                    ero.append(0)
                if erosion['deposition_total (m)'][LP] != 0:
                    depo.append(np.array(erosion['duration_total (s)'])[LP])
                else:
                    depo.append(0)

            duration_ero = np.hstack(np.nansum(np.dstack((np.array(ero), duration_ero)), 2))
            duration_depo = np.hstack(np.nansum(np.dstack((np.array(depo), duration_depo)), 2))

            dataOverview[config + '_erosion'] = np.array(erosion['erosion_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            dataOverview[config + '_deposition'] = np.array(erosion['deposition_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            dataOverview[config + '_netErosion'] = np.array(erosion['netErosion_total (m)'])#np.array([np.isnan(i) for i in erosion['erosion_total (m)']])
            dataOverview[config + '_duration'] = np.array(erosion['duration_total (s)'])

    for j, x in enumerate(duration_ero):
        if np.isnan(x) or x == 0:
            duration_ero[j] = np.inf

    for j, x in enumerate(duration_depo):
        if np.isnan(x) or x == 0:
            duration_depo[j] = np.inf

    erosion_rate = erosion_position/duration_ero    
    deposition_rate = deposition_position/duration_depo    
    #print(erosion_rate)    
    #print(deposition_rate)   

    erosionStd = getErrorBarsForErosion(m_i, f_i, alpha, zeta, campaign, configurationChosen, duration_ero, n_target)
    erosion_rateStd = np.array(erosionStd)/duration_ero
    depositionStd = getErrorBarsForDeposition(m_i, f_i, zeta, campaign, configurationChosen, duration_depo, n_target)
    deposition_rateStd = np.array(depositionStd)/duration_depo
    if not errors:
        erosionStd = np.array(erosionStd) * 0
        erosion_rateStd = np.array(erosionStd) * 0
        depositionStd = np.array(erosionStd) * 0
        deposition_rateStd = np.array(erosionStd) * 0
    #plot low iota, not extrapolated
    plot.plotTotalErodedLayerThickness(LP_position, erosion_position, deposition_position, erosionStd, depositionStd, '', configurationChosen, campaign, T_default)
    plot.plotTotalErodedLayerThickness(LP_position, erosion_rate, deposition_rate, erosion_rateStd, deposition_rateStd, '', configurationChosen, campaign, T_default, False, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_position, deposition_position, 'low', configurationChosen, campaign, T_default)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_rate, deposition_rate, 'low', configurationChosen, campaign, T_default, False, True)

    #plot high iota, not extrapolated
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_position, deposition_position, 'high', configurationChosen, campaign, T_default)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_rate, deposition_rate, 'high', configurationChosen, campaign, T_default, False, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_position, deposition_position, 'high', configurationChosen, campaign, T_default)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_rate, deposition_rate, 'high', configurationChosen, campaign, T_default, False, True)

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
        erosion_total_config = []
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
    duration_total = 0

    
    #copy dataframe dataOverview but write extrapolated values to it
    dataOverview3 = pd.DataFrame({'LP': dataOverview['LP']})    
    for iota in ['low', 'standard', 'high']:
        duration_total_iota = 0
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
                    duration_total_iota += list(dataOverview2[config + '_duration'])[0]

            if duration_erosion != 0:
                erosion_total_iota.append(erosion * duration_total_iota/duration_erosion)   
            else: 
                erosion_total_iota.append(0)
            if duration_deposition != 0:
                deposition_total_iota.append(deposition * duration_total_iota/duration_deposition)   
            else:
                deposition_total_iota.append(0)

        for config in configurationList:
            if config in config_missing:
                continue
            if configOverview['iota'][list(configOverview['configuration']).index(config)] != iota:
                continue
            erosion, deposition, netErosion = [], [], []

            for LP in range(len(dataOverview2['LP'])):
                if duration_total_iota != 0:
                    if list(dataOverview2[config + '_erosion'])[LP] == 0 or np.isnan(list(dataOverview2[config + '_erosion'])[LP]):
                        erosion.append(erosion_total_iota[LP] * list(dataOverview2[config + '_duration'])[LP]/duration_total_iota)
                    else:
                        erosion.append(dataOverview2[config + '_erosion'][LP])

                    if list(dataOverview2[config + '_deposition'])[LP] == 0 or np.isnan(list(dataOverview2[config + '_deposition'])[LP]):
                        deposition.append(deposition_total_iota[LP] * list(dataOverview2[config + '_duration'])[LP]/duration_total_iota)
                    else:
                        deposition.append(list(dataOverview2[config + '_deposition'])[LP])
                else:
                    erosion.append(0)
                    deposition.append(0)
                netErosion.append(erosion[-1] - deposition[-1])

            dataOverview3[config + '_erosion'] = erosion
            dataOverview3[config + '_deposition'] = deposition
            dataOverview3[config + '_netErosion'] = netErosion
            dataOverview3[config + '_duration'] = dataOverview2[config + '_duration']
        duration_total += duration_total_iota
        erosion_total = np.hstack(np.nansum(np.dstack((np.array(erosion_total_iota), erosion_total)), 2))
        deposition_total = np.hstack(np.nansum(np.dstack((np.array(deposition_total_iota), deposition_total)), 2))

    if duration_total != 0:
        erosion_rate_total = erosion_total/duration_total
        deposition_rate_total = deposition_total/duration_total
    else:
        erosion_rate_total = np.zeros_like(erosion_rate_total)
        deposition_rate_total = np.zeros_like(erosion_rate_total)
    
    erosion_totalStd = getErrorBarsForErosion(m_i, f_i, alpha, zeta, campaign, configurationChosen, duration_total, n_target)
    erosion_rate_totalStd = np.array(erosion_totalStd)/duration_total
    deposition_totalStd = getErrorBarsForDeposition(m_i, f_i, zeta, campaign, configurationChosen, duration_total, n_target)
    deposition_rate_totalStd = np.array(deposition_totalStd)/duration_total
    if not errors:
        erosion_totalStd = np.array(erosionStd) * 0
        erosion_rate_totalStd = np.array(erosionStd) * 0
        deposition_totalStd = np.array(erosionStd) * 0
        deposition_rate_totalStd = np.array(erosionStd) * 0
    
    #plot low iota, extrapolated
    plot.plotTotalErodedLayerThickness(LP_position, erosion_total, deposition_total, erosion_totalStd, deposition_totalStd, '', configurationChosen, campaign, T_default, True)
    plot.plotTotalErodedLayerThickness(LP_position, erosion_rate_total, deposition_rate_total, erosion_rate_totalStd, deposition_rate_totalStd, '', configurationChosen, campaign, T_default, True, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_total, deposition_total, 'low', configurationChosen, campaign, T_default, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_rate_total, deposition_rate_total, 'low', configurationChosen, campaign, T_default, True, True)
    
    #plot high iota, extrapolated
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_total, deposition_total, '', configurationChosen, campaign, T_default, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_rate_total, deposition_rate_total, 'high', configurationChosen, campaign, T_default, True, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_total, deposition_total, 'high', configurationChosen, campaign, T_default, True)
    #plot.plotTotalErodedLayerThickness(LP_position, erosion_rate_total, deposition_rate_total, 'high', configurationChosen, campaign, T_default, True, True)
    
    mass = approximationErodedMaterialMassWholeCampaign(LP_position, erosion_total, deposition_total, n_target)

    #end of: for extraplated values -> extrapolate values for missing configurations generally##################
    if configurationChosen == 'all':
        configurationChosen = 'WholeCampaign'
    dataOverview.to_csv('results/erosionFullCampaign/{campaign}_{Ts}_totalErosion{configuration}AllPositions.csv'.format(campaign=campaign, Ts=T_default, configuration=configurationChosen), sep=';')
    dataOverview3.to_csv('results/erosionFullCampaign/{campaign}_{Ts}_totalErosion{configuration}AllPositionsExtrapolated.csv'.format(campaign=campaign, Ts=T_default, configuration=configurationChosen), sep=';')

    return mass

##################################################################################################################################################################
def calculateTotalErodedLayerThicknessFromOverviewFile(m_i: list[int|float], f_iList: list[list[int|float]], alpha: int|float, zeta: list[int|float], n_target: int|float, campaign: str, config_short: str, extrapolated: bool, LPposition: list[int|float], LPexcluded: list[int] =[], T_default: str ='320K', overviewFile: str ='_totalErosionWholeCampaignAllPositions') -> list[list[float]]:
    mass = []
    erosionList, erosion_rateList, depositionList, deposition_rateList = [], [], [], []
    erosionStdList, erosion_rateStdList, depositionStdList, deposition_rateStdList = [], [], [], []
    for f_i in f_iList:
        conc = str(f_i[0]).split('.')[1] + '_' + str(f_i[3]).split('.')[1] + '_' + str(f_i[4]).split('.')[1]
        if extrapolated:
            overviewFilePath = f'results{campaign}_{conc}/erosionFullCampaign/{campaign}_{T_default}{overviewFile}Extrapolated.csv'
        else:
            overviewFilePath = f'results{campaign}_{conc}/erosionFullCampaign/{campaign}_{T_default}{overviewFile}.csv'

        if not os.path.isfile(overviewFilePath):
            print(f'No overview file present under "{overviewFile}"')
            continue

        overviewTable = pd.read_csv(overviewFilePath, sep=';')
        keys = np.array(overviewTable.keys())

        if config_short == 'all':
            filterKeysErosion = np.array(['erosion' in key for key in keys])
            filterKeysDeposition = np.array(['deposition' in key for key in keys])
            filterKeysDuration = np.array(['duration' in key for key in keys])
        else:
            filterKeysErosion = np.array([config_short in key and 'erosion' in key for key in keys])
            filterKeysDeposition = np.array([config_short in key and 'deposition' in key for key in keys])
            filterKeysDuration = np.array([config_short in key and 'duration' in key for key in keys])

        keysErosion = keys[filterKeysErosion]
        keysDeposition = keys[filterKeysDeposition]
        keysDuration = keys[filterKeysDuration]

        erosion = np.array([0.]*36)
        deposition = np.array([0.]*36)
        duration = np.array([0.]*36)

        for keyEro, keyDepo, keyDur in zip(keysErosion, keysDeposition, keysDuration):
            erosion = np.hstack(np.nansum(np.dstack((np.array(overviewTable[keyEro]), erosion)), 2))
            deposition = np.hstack(np.nansum(np.dstack((np.array(overviewTable[keyDepo]), deposition)), 2))
            for i in range(len(overviewTable[keyEro])):
                if overviewTable[keyEro][i] != 0 and not np.isnan(overviewTable[keyEro][i]):
                    duration[i] = duration[i] + overviewTable[keyDur][i]

        erosionStd = getErrorBarsForErosion(m_i, f_i, alpha, zeta, campaign, config_short, duration, n_target)
        depositionStd = getErrorBarsForDeposition(m_i, f_i, zeta, campaign, config_short, duration, n_target)

        for i in LPexcluded:
            LPposition[i] = np.nan
            erosion[i] = np.nan
            deposition[i] = np.nan
            erosionStd[i] = np.nan
            deposition[i] = np.nan
        
        erosion_rate = erosion/np.array(duration)
        deposition_rate = deposition/np.array(duration)
        erosion_rateStd = np.array(erosionStd)/np.array(duration)
        deposition_rateStd = np.array(depositionStd)/np.array(duration)

        #erosionStd = np.array(erosionStd) * 0
        #erosion_rateStd = np.array(erosionStd) * 0
        #depositionStd = np.array(erosionStd) * 0
        #deposition_rateStd = np.array(erosionStd) * 0

        plot.plotTotalErodedLayerThickness(LPposition, erosion, deposition, erosionStd, depositionStd, '', config_short, campaign, T_default, extrapolated, safe='results{campaign}_{conc}/erosionFullCampaign/{campaign}_{Ts}_totalErosion{rates}_{config}{iota}{extrapolated}.png'.format(campaign=campaign, conc=conc, Ts=T_default, rates='Layers', iota='', config=config_short, extrapolated=extrapolated))
        plot.plotTotalErodedLayerThickness(LPposition, erosion_rate, deposition_rate, erosion_rateStd, deposition_rateStd, '', config_short, campaign, T_default, True, True, safe='results{campaign}_{conc}/erosionFullCampaign/{campaign}_{Ts}_totalErosion{rates}_{config}{iota}{extrapolated}.png'.format(campaign=campaign, conc=conc, Ts=T_default, rates='Rates', iota='', config=config_short, extrapolated=extrapolated))
    
        mass.append(approximationErodedMaterialMassWholeCampaign(LPposition, erosion, deposition, n_target))
        
        erosionList.append(erosion)
        depositionList.append(deposition)
        erosionStdList.append(erosionStd)
        depositionStdList.append(depositionStd)
        erosion_rateList.append(erosion_rate)
        deposition_rateList.append(deposition_rate)
        erosion_rateStdList.append(erosion_rateStd)
        deposition_rateStdList.append(deposition_rateStd)

    plot.plotComparisonErodedLayerThickness(LPposition, erosionList, depositionList, erosionStdList, depositionStdList, 1, config_short, campaign, T_default, extrapolated)
    plot.plotComparisonErodedLayerThickness(LPposition, erosion_rateList, deposition_rateList, erosion_rateStdList, deposition_rateStdList, 1, config_short, campaign, T_default, extrapolated, True)

    return mass

##################################################################################################################################################################
def approximationErodedMaterialMassWholeCampaign(LP_position: list[int|float], erosion: list[int|float], deposition: list[int|float], n_target: int|float, M_target: int|float =12.011) -> float:
    ''' returns the mass of net eroded/deposited (</> 0) target material in g'''
    LP_position = list(itertools.chain.from_iterable([LP_position, LP_position]))
    indices = [0, 14, 18, 32, 36]

    erosionNAN = np.array([not np.isnan(x) for x in erosion])
    erosion = np.array(erosion)[erosionNAN]
    deposition = np.array(deposition)[erosionNAN]
    excludedIndex = np.array(range(36))[~erosionNAN]
    LP_position = np.array(LP_position)[erosionNAN]

    for index in excludedIndex:
        if index < 14:
            indices[1] -= 1
        elif index < 18:
            indices[2] -= 1
        elif index < 32:
            indices[3] -= 1
        elif index < 36:
            indices[4] -= 1

    volume, volumeEro, volumeDep = 0, 0, 0
    for i in [0, 2]:#for i in range(len(indices) - 1):
        crossSection, crossSectionEro, crossSectionDep = 0, 0, 0
        for j in range(indices[i], indices[i + 1] - 1):
            delta_x = LP_position[j + 1] - LP_position[j]
            erosion_av = (erosion[j + 1] + erosion[j])/2
            deposition_av = (deposition[j + 1] + deposition[j])/2
            crossSectionEro += delta_x * erosion_av 
            crossSectionDep += delta_x * deposition_av
            crossSection += delta_x * (- erosion_av + deposition_av) 
        volumeEro += crossSectionEro * 0.5 * 1/(0.1)#0.1m is width of strike line, 1m^2 is the area occupied by the strikeline
        volumeDep += crossSectionDep * 0.5 * 1/(0.1)#0.1m is width of strike line, 1m^2 is the area occupied by the strikeline
        volume += crossSection * 0.5 * 1/(0.1)#0.1 is width of strike line, 1 is the area occupied by the strikeline
    return [n_target * M_target * volume/scipy.constants.N_A, n_target * M_target * volumeEro/scipy.constants.N_A, n_target * M_target * volumeDep/scipy.constants.N_A] #equivalent to rho * V

##################################################################################################################################################################
def calculateAverageQuantityPerConfiguration(quantity: str,
                                             config: str, 
                                             LP_position: list[int|float], 
                                             campaign: str ='',
                                             dischargeList: str ='results/configurations/dischargeList_', 
                                             excluded :list[str] = [],
                                             errorscalculated: bool =False) -> str|None:
    ''' This function calculates the average value for one of the quantities ne, Te, or Ts (="quantity") for one configuration "config"
        -> at all Langmuir Probe positions given in "LP_position"
        -> for all discharges in "config" and in "campaign" given in "dischargeList" but not in "excluded"
        "errorscalculated" means that errors are read from existing file instead of being calculated'''
    if campaign == '':
        campaign = 'OP223'

    if not os.path.isfile(dischargeList + campaign + '_{config}.csv'.format(config=config)):
        return 'file missing for ' + config
    
    dischargeOverview = pd.read_csv(dischargeList + campaign + '_{config}.csv'.format(config=config), sep=';')
    averageConfiguration = np.array([0.]*36)
    averageConfigurationStd = np.array([[0.]]*36)
    averageConfigurationStdData = np.array([[0.]]*36)
    timeConfiguration = np.array([0.]*36)
    x = np.nansum(np.array(dischargeOverview['duration']))
    timeConfigurationTotal = np.array([x]*36)
    for discharge, duration, overviewTable in zip(dischargeOverview['dischargeID'], dischargeOverview['duration'], dischargeOverview['overviewTable']):
        discharge = str(discharge)

        if discharge[-2] == '.':
            discharge = discharge + '00'
        elif discharge[-3] == '.':
            discharge = discharge + '0'

        if discharge in excluded:
            continue
 
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
        averageListDischargeStdData = []
        averageListDischargeStd = []
        timeDischarge = []
        for DUindex in ['lower', 'upper']:
            for LPindex in range(18):
                if DUindex + str(LPindex) in LPs:
                    counter = LPs.index(DUindex + str(LPindex))
                    averageDischarge = 0
                    averageDischargeStdData = 0
                    averageDischargeStd = []
                    duration_measurement = 0
                    for counter2, dt in enumerate(timesteps):
                        indexCounter = counter * len(timesteps) + counter2
                        if list(overviewTable[quantity])[indexCounter] != 0 and not np.isnan(list(overviewTable[quantity])[indexCounter]):
                            if quantity == 'Te':
                                if list(overviewTable[quantity])[indexCounter] != 320:
                                    averageDischarge += list(overviewTable[quantity])[indexCounter] * dt
                                    averageDischargeStdData += 0
                                    duration_measurement += dt
                                    if np.isnan(dt):
                                        dt = 0
                                    for correction in range(int(dt*10)):
                                        averageDischargeStd.append(list(overviewTable[quantity])[indexCounter])
                                else:
                                    continue
                            else:
                                averageDischarge += list(overviewTable[quantity])[indexCounter] * dt
                                averageDischargeStdData += list(overviewTable['s'+quantity])[indexCounter] * dt
                                duration_measurement += dt
                                if np.isnan(dt):
                                    dt = 0 
                                for correction in range(int(dt*10)):
                                    averageDischargeStd.append(list(overviewTable[quantity])[indexCounter])
                        else:
                            continue
                    if duration_measurement != 0:
                        averageListDischarge.append(averageDischarge/duration_measurement)
                        averageListDischargeStdData.append(averageDischargeStdData/duration_measurement)
#1                        averageListDischargeStd.append(np.std(np.array(averageDischargeStd)/duration_measurement))
                        averageListDischargeStd.append(averageDischargeStd)
                        timeDischarge.append(duration_measurement)
                    else:
                        averageListDischarge.append(0)
                        averageListDischargeStdData.append(0)
#1                        averageListDischargeStd.append(0)
                        averageListDischargeStd.append([np.nan])
                        timeDischarge.append(0)
                else:
                    averageListDischarge.append(0)
                    averageListDischargeStdData.append(0)
#1                    averageListDischargeStd.append(0)
                    averageListDischargeStd.append([np.nan])
                    timeDischarge.append(0)
        averageConfiguration = np.hstack(np.nansum(np.dstack((np.array(averageConfiguration), (np.array(averageListDischarge) * np.array(timeDischarge)))), 2))
        averageConfigurationStdData = np.hstack(np.nansum(np.dstack((np.array(averageConfigurationStdData), (np.array(averageListDischargeStdData) * np.array(timeDischarge)))), 2))
#1        averageConfigurationStd = np.hstack(np.nansum(np.dstack((np.array(averageConfigurationStd), (np.array(averageListDischargeStd)**2 * np.array(timeDischarge)**2))), 2))
        help = []
        for i in range(len(averageConfigurationStd)):
            help2 = [averageConfigurationStd[i]]
            help2.append(averageListDischargeStd[i])
            help.append(list(itertools.chain.from_iterable(help2)))
        averageConfigurationStd = help.copy()
        timeConfiguration = np.hstack(np.nansum(np.dstack((np.array(timeConfiguration), np.array(timeDischarge))), 2))
    averageConfiguration = averageConfiguration/timeConfiguration
    averageConfigurationStdData = averageConfigurationStdData/timeConfiguration
#1    averageConfigurationStd = np.sqrt(averageConfigurationStd)/timeConfiguration
#1    averageConfigurationStd = np.std(np.array(averageConfigurationStd))/timeConfiguration
    help = []
    for i in range(len(averageConfigurationStd)):
        help.append(np.nanstd(np.array(averageConfigurationStd[i])))    
    averageConfigurationStd = help

    if not os.path.exists('results/averageQuantities/{quantity}/{config}'.format(quantity=quantity, config=config)):
        os.makedirs('results/averageQuantities/{quantity}/{config}'.format(quantity=quantity, config=config)) 

    fig, ax = plt.subplots(2, 1, layout='constrained', figsize=(7, 10), sharex=True)

    ax[0].errorbar(LP_position[:14], averageConfiguration[:14], yerr=averageConfigurationStd[:14], fmt='b-', capsize=4, label='lower divertor unit')
    ax[0].errorbar(LP_position[:14], averageConfiguration[18:32], yerr=averageConfigurationStd[18:32], fmt='m-', capsize=4, label='upper divertor unit')
    ax[0].legend()
    ax[0].set_ylabel('Low iota: average ' + quantity)
    ax[1].errorbar(LP_position[14:], averageConfiguration[14:18], yerr=averageConfigurationStd[14:18], fmt='b-', capsize=4, label='lower divertor unit')
    ax[1].errorbar(LP_position[14:], averageConfiguration[32:], yerr=averageConfigurationStd[32:], fmt='m-', capsize=4, label='upper divertor unit')
    ax[1].legend()
    ax[1].set_xlabel('distance from pumping gap (m)')
    ax[1].set_ylabel('High iota: average ' + quantity)
    fig.savefig('results/averageQuantities/{quantity}/{config}/{quantity}{campaign}{config}AverageAllPositions.png'.format(quantity=quantity, campaign=campaign, config=config), bbox_inches='tight')
    plt.show()
    plt.close()

    for i, timeConfig in enumerate(timeConfiguration):
        if timeConfig == 0:
            timeConfigurationTotal[i] = 0

    return averageConfiguration, timeConfiguration, timeConfigurationTotal, averageConfigurationStd, averageConfigurationStdData
##################################################################################################################################################################

def frameCalculateAverageQuantityPerConfiguration(quantities: list[str], 
                                                  campaigns: list[str], 
                                                  configurations: list[str], 
                                                  LP_position: list[int|float],
                                                  config_short: bool|str =False, 
                                                  excluded: list[str] =[],
                                                  Calculated: bool =False,
                                                  calculatedFile: str ='results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv') -> None: 
    ''' This function is the frame work for calculateAverageQuantityPerConfiguration
        It determines the average value of each quantity given in "quantities" (ne, Te, Ts) at each Langmuir Probe position given by "LP_position"
        -> for each configuration given in "configurations" (if file with discharges exists) and matching "config_short"
        -> "config_short can be False, then the average of all configurations is determined, or e.g. 'EIM', then only EIM... configurations are considered for the total average
        -> distinguishes between "campaigns" 'OP22', 'OP23', and '' meaning both campaigns
        "excluded" is a list of discharges that should be excluded from averaging
        "Calculated" True means reading errors from calculatedFile instead of calculating them'''       
    
    averageReturn = []
    averageReturnStd = []

    if Calculated:
        averageFrame = pd.read_csv(calculatedFile, sep=';')
        for quantity in quantities:
            for campaign in campaigns:
                if type(config_short) == bool:
                    config_short = 'all'
                if campaign == '':
                    campaignTXT = 'OP223'
                else:
                    campaignTXT = campaign
                averageReturn.append(averageFrame[campaignTXT + config_short + quantity])
                averageReturnStd.append(averageFrame[campaignTXT + config_short + quantity + 'Std'])
        return averageReturn, averageReturnStd
        
    for quantity in quantities:
        if quantity == 'ne':
            quantityUnit = 'm$^{-3}$'
        elif quantity == 'Te':
            quantityUnit = 'eV'
        elif quantity == 'Ts':
            quantityUnit = 'K'
        else:
            quantityUnit = ''

        for campaign in campaigns:
            averageCampaign = np.array([0.] * 36)
            averageCampaignStd = np.array([0.] * 36)
            timeCampaign = np.array([0.] * 36)
            
            for config in configurations:
                
                if type(config_short) != bool:
                    configChosen = config_short
                    if not config.startswith(config_short):
                        print(config + 'not matching configuration group ' + config_short)
                        continue
                else:
                    configChosen = ''
                
                resultAverage = calculateAverageQuantityPerConfiguration(quantity, config, LP_position, campaign, excluded=excluded)
                if type(resultAverage) == str:
                    print(resultAverage)
                else:  
                    timeCampaign = np.hstack(np.nansum(np.dstack((np.array(timeCampaign), np.array(resultAverage[2]))), 2))
                    #timeCampaign = np.hstack(np.nansum(np.dstack((np.array(timeCampaign), np.array(resultAverage[1]))), 2))
                    averageCampaign = np.hstack(np.nansum(np.dstack((np.array(averageCampaign), ((np.array(resultAverage[2])*np.array(resultAverage[0]))) )), 2))
                    if quantity == 'Ts':
                        averageCampaignStd = np.hstack(np.nansum(np.dstack((np.array(averageCampaignStd), ((np.array(resultAverage[2])**2 * np.array(resultAverage[3])**2)) )), 2))
                    else:    
                        averageCampaignStd = np.hstack(np.nansum(np.dstack((np.array(averageCampaignStd), ((np.array(resultAverage[2])**2 * np.array(resultAverage[4])**2)) )), 2))
                    #averageCampaign = np.hstack(np.nansum(np.dstack((np.array(averageCampaign), ((np.array(resultAverage[1])*np.array(resultAverage[0]))) )), 2))
        
            averageCampaign = averageCampaign/timeCampaign
            averageCampaignStd = np.sqrt(averageCampaignStd)/timeCampaign

            fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(12, 10), sharex='col', sharey=True)

            ax[0][1].errorbar(LP_position[14:], averageCampaign[14:18], yerr=averageCampaignStd[14:18], fmt='k-', capsize=4, label='lower divertor unit')
            ax[1][1].errorbar(LP_position[14:], averageCampaign[32:], yerr=averageCampaignStd[32:], fmt='k-', capsize=4, label='upper divertor unit')
            ax[0][0].errorbar(LP_position[:14], averageCampaign[:14], yerr=averageCampaignStd[:14], fmt='k-', capsize=4, label='lower divertor unit')
            ax[1][0].errorbar(LP_position[:14], averageCampaign[18:32], yerr=averageCampaignStd[18:32], fmt='k-', capsize=4, label='upper divertor unit')
            for i in range(2):
                for j in range(2):
                    ax[i][j].legend()
                    ax[i][j].set_ylim(bottom=0)

            ax[1][1].set_xlabel('distance from pumping gap (m)')
            ax[1][0].set_xlabel('distance from pumping gap (m)')
            ax[0][0].set_ylabel(f'Low iota: average {quantity} lower divertor unit in ({quantityUnit})')
            ax[0][1].set_ylabel(f'Low iota: average {quantity} upper divertor unit in ({quantityUnit})')
            ax[0][1].set_ylabel(f'High iota: average {quantity} lower divertor unit in ({quantityUnit})')
            ax[1][1].set_ylabel(f'High iota: average {quantity} upper divertor unit in ({quantityUnit})')
            fig.savefig('results/averageQuantities/{quantity}/{quantity}{campaign}{configChosen}AverageAllPositionsHighIota.png'.format(quantity=quantity, campaign=campaign, configChosen=configChosen), bbox_inches='tight')
            plt.show()
            plt.close()

            averageReturn.append(averageCampaign)
            averageReturnStd.append(averageCampaignStd)

    return averageReturn, averageReturnStd

############################################################################################################################
def approximationOfLayerThicknessesBasedOnAverageParameterValues(LP_position: list[int|float], campaign: str,
                                                                 alpha: int|float, zeta: list[int|float], m_i: list[int|float], f_i: list[int|float], ions: list[str], k: int|float, n_target: int|float,
                                                                 importAverages: bool|str, configuration: str ='all') -> int|float: 
    ''' This function calculates the expectable erosion/deposition of the divertor for a set of average parameter values + input parameters
        ne, Te, Ts averages at each Langmuir Probe Position ("LP_position") on lower and upper divertor unit for each campaign (and configuration if provided)
        -> input parameters (second line of parameters)
        "importAverages" indicates, that average ne, Te, Ts are used as defined below (for False) or read from a .csv file (for str=file path)
        returns mass of net eroded material in g'''
    if campaign == 'OP22': 
        ne = [8.10861120e+18, 9.23699360e+18, 5.39394510e+18, 4.88005878e+18, 6.41963593e+19, 7.53689257e+18, 3.74975667e+18, 2.87621473e+18, 1.86239725e+18, 1.77501141e+18, 1.24395721e+18, 8.16433709e+17, 7.98950114e+17, 2.63917556e+18, 6.71595628e+18, 6.02079381e+18, 6.96774702e+18, 8.67705959e+18, 7.34334178e+18, 6.82833426e+18, 3.96166626e+18, 3.27682856e+18, 3.05768770e+18, 2.92846510e+18, 2.13915956e+18, 1.78086238e+18, 5.61754825e+19, 2.76740351e+18, 3.49056208e+18, 1.00981278e+18, 9.08946107e+17, 1.89625133e+19, 4.69150909e+18, 5.29825547e+18, 8.07835520e+18, 7.49023570e+18]
        Te = [23.10917898, 23.97398849, 17.24971306, 12.9683621, 16.1148856, 15.14815498, 11.83223752, 11.13380081, 10.22305666, 8.8592773, 7.93170485, 6.82595981, 6.60610872, 4.80160941, 17.62515757, 16.13908412, 25.78049094, 20.23960462, 22.49694823, 23.95433543, 20.73571079, 16.05200695, 16.11300381, 15.77758246, 10.83166445, 9.89855224, 10.14166381, 7.22341548, 8.01914169, 6.33356499, 11.30071018, 4.04176992, 9.98670479, 11.42823382, 17.45413223, 25.34487122]
        Ts = [348.53123333, 339.57542575, 327.58684535, 322.48441638, 320.37366489, 320.53570604, 316.64689784, 314.81928323, 313.70052689, 312.70113155, 312.37350838, 312.15832773, 311.93336754, 312.42157821, 333.43560488, 371.92962704, 433.55672356, 435.60782705, 357.57614827, 351.53122556, 336.32154333, 326.37350381, 323.99850832, 321.41492792, 316.27895614, 315.81451817, 314.67173041, 313.93423271, 312.82564765, 312.18334915, 312.06617817, 313.87187357, 379.7223462, 333.73038846, 357.42425781, 398.05262984]
        t = 20372.65

    elif campaign == 'OP23':
        ne = [7.23425072e+18, 6.49082692e+18, 4.74422826e+18, 4.13011762e+18, 1.00974487e+19, 4.06601437e+18, 4.58674313e+18, 3.35729073e+18, 1.63927672e+18, 2.32379908e+18, 1.83177974e+18, 1.51632845e+18, 5.19738099e+17, 9.77853187e+17, np.nan , np.nan , 3.99357880e+18, np.nan , 8.19180641e+18, 8.43372562e+18, 2.26185620e+18, 2.21738383e+18, 1.88094868e+18, 1.87003597e+18, 4.06435840e+18, 3.09974771e+18, 4.07980777e+18, 1.88866226e+18, 1.50819279e+18, 1.52277035e+18, 9.89528048e+17, 2.13408464e+18, 4.46755927e+18, 5.46513067e+18, 6.51305893e+18, 7.82164729e+18]
        Te = [22.01761885, 22.46042915, 16.40248943, 14.18213774, 17.20604823, 16.68423172, 12.41207622, 12.30609299, 14.7635153, 10.55438673, 10.20832988, 10.06609525, 26.80995318, 17.61596149, np.nan , np.nan , 29.90268815, np.nan , 22.29111767, 23.60803396, 21.16374238, 15.52334095, 15.09640528, 15.98598422, 13.43930376, 13.32485776, 13.91981687, 13.66644289, 10.7030418, 11.5603852, 30.29936855, 18.54352217, 9.81694099, 13.86485593, 18.71589961, 19.61945445]
        Ts = [344.02238096, 331.31824545, 321.68943264, 318.01602097, 317.16175369, 318.8853121, 315.30428473, 313.76628092, 312.89937246, 311.64312308, 311.17382574, 310.84767811, 311.19797395, 311.8039927, np.nan , np.nan , 377.54746229, np.nan , 340.8332816, 341.27643279, 333.74961652, 322.96430675, 319.93706882, 318.74008531, 315.99218539, 315.65745321, 314.32493327, 312.87997257, 312.18616397, 311.9843921, 312.43478398, 314.37532688, 350.47602205, 354.20017493, 370.1831045, 386.89669786]
        t = 28333.24
    
    elif campaign == '':
        ne = [7.67669550e+18, 7.87359786e+18, 5.06883517e+18, 4.50678007e+18, 4.18099586e+19, 5.84696312e+18, 4.16512236e+18, 3.11504844e+18, 1.75203223e+18, 2.04518742e+18, 1.53398982e+18, 1.16495639e+18, 6.57016472e+17, 1.64853914e+18, 6.71595628e+18, 6.02079381e+18, 4.21687487e+18, 8.67705959e+18, 7.73744930e+18, 7.59404418e+18, 3.13558215e+18, 2.76987019e+18, 2.49778347e+18, 2.42486796e+18, 3.05919152e+18, 2.40057371e+18, 3.24314296e+19, 2.35439188e+18, 2.55752020e+18, 1.23901201e+18, 9.53427533e+17, 1.03968918e+19, 4.47277606e+18, 5.45519706e+18, 6.60716485e+18, 7.79225959e+18]
        Te = [22.57062858, 23.22354637, 16.82666691, 13.5712462, 16.62277776, 15.90181622, 12.11971609, 11.71505922, 12.4771496, 9.70027448, 9.0603626, 8.43274704, 16.30989349, 10.37925827, 17.62515757, 16.13908412, 29.64032044, 20.23960462, 22.39974752, 23.79127039, 20.93864322, 15.80200345, 15.63251635, 15.87614281, 12.0691916, 11.52400789, 11.92908508, 10.25756316, 9.29313957, 8.79815706, 20.29700216, 10.68119372, 9.8210141, 13.722815, 18.64225509, 19.97987108]
        Ts = [346.28735421, 335.44220232, 324.63478839, 320.24767996, 318.77968648, 319.71480009, 315.97417336, 314.29167, 313.29910358, 312.17095723, 311.7723403, 311.50155343, 311.57167871, 312.1262853, 333.43560488, 371.92962704, 381.08950793, 435.60782705, 349.56278168, 346.6231405, 335.09058373, 324.74181514, 322.05464744, 320.13471141, 316.14129944, 315.7391233, 314.50492494, 313.4271456, 312.5185426, 312.08774737, 312.24329866, 314.11283431, 352.32557001, 352.90565834, 369.37623049, 387.60220297]
        t = 48705.89
        campaign = 'OP223'

    if type(importAverages) == str:
        if os.path.isfile(importAverages):
            averageFrame = pd.read_csv(importAverages, sep=';')
            ne = averageFrame[campaign + configuration + 'ne']
            Te = averageFrame[campaign + configuration + 'Te']
            Ts = averageFrame[campaign + configuration + 'Ts']
    ####!!!t must be different for configuration != 'all'!!!###
    zeta = list(itertools.chain.from_iterable([zeta, zeta]))

    erosion, deposition = [], []
    for i in range(len(ne)):
        Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness1, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = calculateErosionRelatedQuantitiesOnePosition(np.array([Te[i]]), np.array([Te[i]]), np.array([Ts[i]]), np.array([ne[i]]), np.array([t]), alpha, zeta[i], m_i, f_i, ions, k, n_target)
        erosion.append(erodedLayerThickness1)
        deposition.append(depositedLayerThickness)

    erosion = np.array(list(itertools.chain.from_iterable(erosion)))
    deposition = np.array(list(itertools.chain.from_iterable(deposition)))

    if configuration == 'all':
        configuration = ''
    plot.plotTotalErodedLayerThickness(LP_position, erosion, deposition, '', 'all', campaign, '', '', False, 'results/averageQuantities/{campaign}{config}AverageAllPositions_alpha{alpha:.3f}_fi{f_i}.png'.format(campaign=campaign, config=configuration, alpha=alpha, f_i=f_i))
    #plot.plotTotalErodedLayerThickness(LP_position, erosion, deposition, 'low', 'all', campaign, '', '', False, 'results/averageQuantities/{campaign}AverageAllPositions_LowIota_UpperDU_alpha{alpha:3f}_fi{f_i}.png'.format(campaign=campaign, alpha=alpha, f_i=f_i))
    #plot.plotTotalErodedLayerThickness(LP_position, erosion, deposition, 'high', 'all', campaign, '', '', False, 'results/averageQuantities/{campaign}AverageAllPositions_HighIota_LowerDU_alpha{alpha:3f}_fi{f_i}.png'.format(campaign=campaign, alpha=alpha, f_i=f_i))
    #plot.plotTotalErodedLayerThickness(LP_position, erosion, deposition, 'high' 'all', campaign, '', '', False, 'results/averageQuantities/{campaign}AverageAllPositions_HighIota_UpperDU_alpha{alpha:3f}_fi{f_i}.png'.format(campaign=campaign, alpha=alpha, f_i=f_i))
   
    mass = approximationErodedMaterialMassWholeCampaign(LP_position, erosion, deposition, n_target)
    
    return mass

###################################################################################################################
def getErrorBarsForFluxDensity(m_i: list[int|float], f_i: list[int|float], zeta:list[int|float], 
                               campaign: str, config_short: str, filename_avParams: str ='results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv') -> list[list[int:float]]:
    if not os.path.isfile(filename_avParams):
        return 'no average parameter value file'
    
    else:
        if type(config_short) == bool:
            config_short = 'all'
        if campaign == '':
            campaignTXT = 'OP223'
        else:
            campaignTXT = campaign
        
        avParams = pd.read_csv(filename_avParams, sep=';')
        deltaGamma = []
        for m, f in zip(m_i, f_i):
            ne = np.array(avParams[campaignTXT+config_short+'ne'])
            neStd = np.array(avParams[campaignTXT+config_short+'neStd'])
            Te = np.array(avParams[campaignTXT+config_short+'Te'])
            TeStd = np.array(avParams[campaignTXT+config_short+'TeStd'])
    
            if len(zeta) != len(ne):
                zeta = np.array(list(itertools.chain.from_iterable([zeta, zeta])))
            else:
                zeta = np.array(zeta)

            deltaGamma.append(np.sqrt(f**2 * np.sin(zeta)**2 * 2 * scipy.constants.e/m * (Te * neStd**2 + ne**2 * 1/(4 * Te) * TeStd**2)))
        
        return deltaGamma

###################################################################################################################
def getErrorBarsForDeposition(m_i: list[int|float], f_i: list[int|float], zeta:list[int|float], 
                              campaign: str, config_short: str, duration: int|float, n_target: int|float,
                              filename_avParams: str ='results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv') -> list[list[int:float]]:
    return duration/n_target * np.array(getErrorBarsForFluxDensity(m_i, f_i, zeta, campaign, config_short, filename_avParams)[3])

###################################################################################################################
def getErrorBarsForY(m_i: list[int|float], f_i: list[int|float], zeta:list[int|float], 
                     campaign: str, config_short: str, 
                     filename_avParams: str ='results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv',
                     E: int|float =100) -> list[list[int:float]]:
    if not os.path.isfile(filename_avParams):
        return 'average parameter value file is missing'
    
    avParams = pd.read_csv(filename_avParams, sep=';')
    if type(config_short) == bool:
        config_short = 'all'
    if campaign == '':
        campaignTXT = 'OP223'
    else:
        campaignTXT = campaign

    ne = np.array(avParams[campaignTXT+config_short+'ne'])
    Te = np.array(avParams[campaignTXT+config_short+'Te'])
    Ts = np.array(avParams[campaignTXT+config_short+'Ts'])
    TsStd = np.array(avParams[campaignTXT+config_short+'TsStd'])

    if len(zeta) != len(ne):
        zeta = np.array(list(itertools.chain.from_iterable([zeta, zeta])))
    else:
        zeta = np.array(zeta)

    gammaH = np.array([calc.calculateFluxIncidentIon(zeta[i], Te[i], Te[i], m_i[0], ne[i], f_i[0]) for i in range(len(Te))])
    deltaGamma = np.array(getErrorBarsForFluxDensity(m_i, f_i, zeta, campaign, config_short, filename_avParams)[0])

    deltaY = []
    for i in range(len(Ts)):
        yieldsH = []
        Yav = calc.calculateChemicalErosionYieldRoth('H', E, Ts[i], gammaH[i])
        for flux in np.linspace(gammaH[i]-deltaGamma[i], gammaH[i]+deltaGamma[i], 10):
            for T in np.linspace(Ts[i]-TsStd[i], Ts[i]+TsStd[i], 10):
                yieldsH.append(calc.calculateChemicalErosionYieldRoth('H', E, T, flux))
        
        if abs(Yav - min(yieldsH)) > abs(Yav - max(yieldsH)):
            deltaY.append(abs(Yav - min(yieldsH)))
        else:
            deltaY.append(abs(Yav - max(yieldsH)))
    #deltaY = np.array(deltaY)*0.    
    return deltaY
    
###################################################################################################################
def getErrorBarsForErosion(m_i: list[int|float], f_i: list[int|float], alpha: int|float, zeta:list[int|float], 
                              campaign: str, config_short: str, duration: int|float, n_target: int|float,
                              filename_avParams: str ='results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv') -> list[list[int:float]]:
    if not os.path.isfile(filename_avParams):
        return 'average parameter value file is missing'
    
    avParams = pd.read_csv(filename_avParams, sep=';')
    if type(config_short) == bool:
        config_short = 'all'
    if campaign == '':
        campaignTXT = 'OP223'
    else:
        campaignTXT = campaign

    ne = np.array(avParams[campaignTXT+config_short+'ne'])
    Te = np.array(avParams[campaignTXT+config_short+'Te'])
    Ts = np.array(avParams[campaignTXT+config_short+'Ts'])
    if len(zeta) != len(ne):
        zeta = np.array(list(itertools.chain.from_iterable([zeta, zeta])))
    else:
        zeta = np.array(zeta)

    gammaH = np.array([calc.calculateFluxIncidentIon(zeta[i], Te[i], Te[i], m_i[0], ne[i], f_i[0]) for i in range(len(Te))])
    gammaC = np.array([calc.calculateFluxIncidentIon(zeta[i], Te[i], Te[i], m_i[3], ne[i], f_i[3]) for i in range(len(Te))])
    gammaO = np.array([calc.calculateFluxIncidentIon(zeta[i], Te[i], Te[i], m_i[4], ne[i], f_i[4]) for i in range(len(Te))])

    YH = np.array([calc.calculateTotalErosionYield('H', Te[i], 'C', alpha, Ts[i], gammaH[i], n_target) for i in range(len(Te))])
    YC = np.array([calc.calculateTotalErosionYield('C', Te[i], 'C', alpha, Ts[i], gammaC[i], n_target) for i in range(len(Te))])
    YO = np.array([calc.calculateTotalErosionYield('O', Te[i], 'C', alpha, Ts[i], gammaO[i], n_target) for i in range(len(Te))])
    

    deltaY = np.array(getErrorBarsForY(m_i, f_i, zeta, campaign, config_short, filename_avParams))
    deltaGamma = getErrorBarsForFluxDensity(m_i, f_i, zeta, campaign, config_short, filename_avParams)
    deltaErosion = duration/n_target * np.sqrt((gammaH * np.array(deltaY))**2 + (YH * np.array(deltaGamma[0]))**2 + (YC * np.array(deltaGamma[3]))**2 + (YO * np.array(deltaGamma[4]))**2) 

    return deltaErosion

###################################################################################################################
def getErrorBarsForNetErosion(m_i: list[int|float], f_i: list[int|float], alpha: int|float, zeta:list[int|float], 
                              campaign: str, config_short: str, duration: int|float, n_target: int|float,
                              filename_avParams: str ='results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv') -> list[list[int:float]]:
    errorErosion = getErrorBarsForErosion(m_i, f_i, alpha, zeta, campaign, config_short, duration, n_target, filename_avParams)
    errorDeposition = getErrorBarsForDeposition(m_i, f_i, zeta, campaign, config_short, duration, n_target, filename_avParams)
    return np.hstack(np.nansum(np.dstack((errorErosion, errorDeposition)), 2))