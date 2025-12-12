''' This file is responsible for the final calculation of sputtering yields and the thickness of the erosion/deposition layer. 
    It processes the data with ProcessData using the functions defined in SputteringYieldFunctions after reading the data with ReadArchiveDB'''

import os
import w7xarchive
import itertools
import matplotlib
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate as integrate

#import src.MarkusFunctions
#import src.Unused as extensions
import src.SputteringYieldFunctions as calc
import src.ProcessData as process
import src.ReadArchieveDB as read
#import src.CXRS
from src.PositionsLangmuirProbes import OP2_TM2Distances, OP2_TM3Distances, OP2_TM8Distances

#avoids pop up of plot windows
matplotlib.use('agg')

#######################################################################################################################################################################
#######################################################################################################################################################################
#INPUT NOT TO BE CHANGED
#initialize common parameter values
e  =  scipy.constants.elementary_charge 
u = scipy.constants.u   #to convert M in [u] to m in [kg]: M * u = m
k_B = scipy.constants.Boltzmann
k = k_B/e  #Boltzmann constant in [eV/K]

#ion masses in [kg] for [H, D, T, C, O]
ions = ['H', 'D', 'T', 'C', 'O']
m_i = np.array([1.00794, 2.01210175, 3.0160495, 12.011, 15.9994]) * u 

#lines for [Be, C, Fe, Mo, W] and columns with [H, D, T, He, Self-Sputtering, O] (O only known for C), in [eV] according to Ref. 1
E_TF = np.array([[256, 282, 308, 720, 2208, 0],
                 [415, 447, 479, 1087, 5688, 9298],
                 [2544, 2590, 2635, 5517, 174122, 0], 
                 [4719, 4768, 4817, 9945, 533127, 0],
                 [9871, 9925, 9978, 20376, 1998893, 0]]) 

#heat of sublimation for [Be, C, Fe, Mo, W], in [eV] according to Ref. 1
E_s = np.array([3.38, 7.42, 4.34, 6.83, 8.68])

#Parameters for chemical erosion of C by H-isotopes, [H, D, T] according to Ref. 1
Q_y_chem = np.array([0.035, 0.1, 0.12])                                                                  
C_d_chem = np.array([250, 125, 83])                                                                               
E_thd_chem = [15, 15, 15]   #threshold energy for Y_damage                                                                      
E_ths_chem = [2, 1, 1]      #threshold energy for Y_surf                                                                        
E_th_chem=[31, 27, 29]   


#filter options to choose from for the discharges (configuration filter is applied later):
#!!! HELIUM AND HYDROGEN DISCHARGES ARE NOT DISTINGUISHED !!!
q1 = '!"Conditioning" AND '
q2 = '!"gas valve tests" AND '
q3 = '!"sniffer tests" AND '
q4 = '!"reference discharge" AND '
q41 = '"ReferenceProgram"'   # in OP2.1 does not work
q5 = '!"Open valves" AND '
q6 = 'id:XP_* AND tags.value:"ok" AND '
q66 = 'id:XP_* AND '
q71 = 'tags.value:"Reference"'  # for OP2.1
q44 = '"reference discharge"'   # for OP2.2, OP2.3
q45 = '"Reference discharge"'   # for OP1.2b
qNBI = 'tags.value:"NBI source 7" OR tags.value:"NBI source 8" AND'

#Parameters for net erosion specifically for divertor
lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct

#######################################################################################################################################################################
#######################################################################################################################################################################
#VARIABLE INPUT PARAMETERS
#ion concentrations (no unit) for [H, D, T, C, O]
f_i = [0.89, 0, 0, 0.04, 0.01]

#target density of CFC-HHF divertor in [1/m^3]
n_target = 9.526 * 1e28 #for rho=1.9 g/cm^3 (rho*1e6*N_A/M_C) N_A is avogadro number, M_C molecular mass of carbon

#incident angle of ions in rad
alpha = 2 * np.pi/9

#default values to be inserted if no measurement data exists for electron density ne in [1/m^3], electron temperature Te in [eV], or surface temperature of the target Ts in [K]
defaultValues = [np.nan, np.nan, 320]

#list of all released configurations (date: 01. Dec. 2025) from W7X-info
configurations = pd.read_csv('inputFiles/Overview2.csv')['configuration']
for configuration in configurations:
    pass#print(configuration)

#reference discharges for impurity concentration analysis of carbon and oxygen by CXRS and DRGA
impurityReferenceShots = ['20250304.075', '20250408.055', '20250408.079']

#alternative:
#manual set up of all configurations to be looked at
#configurations = ['EIM000-2520', 'EIM000-2620', 'KJM008-2520', 'KJM008-2620', 'FTM000-2620', 'FTM004-2520', 'DBM000-2520', 'FMM002-2520',
#                  'EIM000+2520', 'EIM000+2620', 'EIM000+2614', 'DBM000+2520', 'KJM008+2520', 'KJM008+2620', 'XIM001+2485', 'MMG000+2520', 
#                  'DKJ000+2520', 'IKJ000+2520', 'FMM002+2520', 'KTM000+2520', 'FTM004+2520', 'FTM004+2585', 'FTM000+2620', 'AIM000+2520', 'KOF000+2520']


#if needed: juice files saved under followig paths for OP2.2/2.3
juicePath = [r"\\share\groups\E3\Diagnostik\DIA-3\QRH_PWI\experiment_analysis\BA Lisa Steiniger\Programmes\PythonScript_LisaSteiniger\BachelorthesisCode\inputFiles\Juice\adb_juice_op22_step_0.2_v0.5_redux.csv",
             r"\\share\groups\E3\Diagnostik\DIA-3\QRH_PWI\experiment_analysis\BA Lisa Steiniger\Programmes\PythonScript_LisaSteiniger\BachelorthesisCode\inputFiles\Juice\adb_juice_op23_step_0.2_v0.6_redux.csv"]

#set filter options for the discharges here (configuration filter is applied later)
filterSelected = q1 + q2 + q3

#parameters for running program:
#are the discharge lists per configuration already saved (at least partially, missing files will be created anyways)? 
#-> results/configurations/dischargeList*.csv
#set this to False only if you want to reset your whole data set (e.g. when having changed the filter for the discharges)
filesExist = False                   

#should the measurement values be read out again?
#set True only if there was a problem with the reading routine
#if False, then already read out data wont be downloaded again
reReadData = False

#are missing values of ne, Te, Ts already intrapolated for the given combination of n_target, f_i, alpha (at least partially, missing files will be created anyways)? 
#-> results/calculationTablesNew/results*.csv
#set this to False when you changed n_target, f_i, or alpha
intrapolated = False

#should ne, Te, Ts, Y, Delta_ero, Delta_dep,... be plotted for original and extrapolated data?
#-> results/plots/*png
plottingOriginalData = False
plottingExtrapolatedData = True

#should the main program run or is just some other testing going on? -> only commands above "if not run:" will be executed
run = False

#######################################################################################################################################################################
#HERE IS THE RUNNING PROGRAM (SHOULD RUN WITHOUT INTERNAL CHANGES WHEN FINALLY FINISHED)
'''
LP_position = [OP2_TM2Distances]
LP_position.append(OP2_TM3Distances)
LP_position.append(OP2_TM8Distances)
LP_position = list(itertools.chain.from_iterable(LP_position))

process.calculateTotalErodedLayerThicknessWholeCampaign(configurations, LP_position)
'''
overview = pd.read_csv('inputFiles/Overview4.csv', sep=';')
iota = []
for IA, IB in zip(overview['IA [A]'], overview['IB [A]']):
    if IA < -1000 and IB < -1000:
        iota.append('low')
    elif IA > 1000 and IB > 1000:
        iota.append('high')
    else:
        iota.append('standard')
overview['iota'] = iota
overview.to_csv('inputFiles/Overview4.csv', sep=';')

if not run:
    exit()

if __name__ == '__main__':
    #36 Langmuir Probes (LPs) in OP2 are located in module 5 upper and lower divertor unit (symmetry)
    #read langmuir probe positions in [m] from pumping gap
    LP_position = [OP2_TM2Distances]
    LP_position.append(OP2_TM3Distances)
    LP_position.append(OP2_TM8Distances)
    LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list
    #LP-position: index 0 - 5 are langmuir probes on TM2h07, 6 - 13 on TM3h01, 14 - 17 on TM8h01 (distance from pumping gap is increasing)
    
    for configuration in configurations:     
        #find all discharge IDs according to the filters activated in "filterSelected" 
        #usually filters by !conditioning, !gas valve tests, !sniffer tests, and configuration (internal filter of "read.readAllShotNumbersFromLogbook")
        print('Logbook search: reading discharges for ' + configuration)
        discharges = read.readAllShotNumbersFromLogbook(configuration, filterSelected, filesExist=filesExist)
        
        #returns either a pd.DataFrame with dischargeID, duration, overviewTable,... or a string that no discharges were found    
        if type(discharges) == str:
            continue    #skip this configuration as no discharges were found
        else:
            pass    #continue with this configuration as some discharges were found
        
        #_lower indicates lower divertor unit, _upper upper divertor unit   
        
        #will hold the dischargeIDs that miss data because LP data/IRcam data/trigger data is not available 
        #for checking only, is not used in any other way than being printed to the terminal
        not_workingLP, not_workingIR, not_workingTrigger = [], [], []

        #run through discharges and read LP and IRcam data before performing calculations with them
        for counter, discharge in enumerate(discharges['dischargeID'][:]):
            discharge = str(discharge)
            print('xdrive/archive: reading' + configuration + ' ' + discharge)
            #tests if data was already read out and saved, in that case reading data is skipped for the discharge in question
            if not reReadData:
                if os.path.isfile('results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)):
                    continue
            #in case that data for this discharge was not read before

            #read Langmuir Probe data from xdrive
            return_LP = read.readLangmuirProbeDataFromXdrive(discharge)
            #returns either list of arrays if data is available, or string if no data was there

            if type(return_LP) == str:
                not_workingLP.append([discharge, counter])
                continue
                #discard that discharge if no LP data is available
            
            #seperate the returned data if LP data is available
            #all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
            #ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s]
            #index represent the active LPs on each divertor unit
            ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper, index_lower, index_upper = return_LP

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
            #index represent the active LPs on each divertor unit
            #LP_position in [m] from pumping gap, time_array in [s]
            return_IR = read.readSurfaceTemperatureFramesFromIRcam(discharge, time_array, ['lower', 'upper'], LP_position, [index_lower, index_upper])
            #returns list with last element is of type string if any trouble with data collection occured and 0 if everything worked
            #trouble leads to settig all missing values of Ts to 320K 
                        
            if type(return_IR[-1]) == str:
                if return_IR[-1] == 'incorrected trigger':
                    not_workingTrigger.append([discharge, counter])
                else:
                    not_workingIR.append([discharge, counter])
            
            #the other element(s) of the returned list hold 2D arrays with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
            Ts_lower, Ts_upper = return_IR[:-1]
            
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
            print('calculationTables: create ' + configuration + ' ' + discharge)
            process.processOP2Data(str(discharge), ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, index_lower, index_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, plotting=plottingOriginalData)

        print('configuration {config}: no trigger {trigger}, no IR {IR}, no LP {LP}'.format(config=configuration, trigger=len(not_workingTrigger), LP=len(not_workingLP), IR=len(not_workingIR)))
        
        #intrapolate missing values, calculate total eroded layer thickness, saves that for all discharges of one configuration in a table
        erosionTable = process.calculateTotalErodedLayerThicknessSeveralDischarges(configuration, discharges['dischargeID'], discharges['duration'], discharges['overviewTable'], alpha, m_i, f_i, ions, k, n_target, defaultValues, intrapolated=intrapolated, plotting=plottingExtrapolatedData)
        
        #sums up all layer thicknesses of all discharges of that configuration to end up with one value for erosion/deposition/net erosion per LP
        #pays attention to discharges with missing data by setting Delta_total = Delta_known * t_total/t_known
        print(process.calculateTotalErodedLayerThicknessWholeCampaignPerConfig(configuration))
    
    #sums up all layer thicknesses of all configurations
    process.calculateTotalErodedLayerThicknessWholeCampaign(configurations, LP_position)    

    #configuration percentages of total runtime in OP2.2/2.3
    read.getRuntimePerConfiguration(configurations=configurations)

    '''
    #plot manually extrapolated erosion over whole campaign
    #only applicable on the manual list of configuration at the moment
    print(configurations)
    erosion_position = np.array([0.] * 36)
    deposition_position = np.array([0.] * 36)
    config_missing = []
    no_data = pd.DataFrame({'LP': ['lower0', 'lower1', 'lower2', 'lower3', 'lower4', 'lower5', 'lower6', 'lower7', 'lower8', 'lower9', 'lower10', 'lower11', 'lower12', 'lower13', 'lower14', 'lower15', 'lower16', 'lower17', 
                             'upper0', 'upper1', 'upper2', 'upper3', 'upper4', 'upper5', 'upper6', 'upper7', 'upper8', 'upper9', 'upper10', 'upper11', 'upper12', 'upper13', 'upper14', 'upper15', 'upper16', 'upper17']})
    if not os.path.isfile('results/erosionFullCampaign/totalErosionWholeCampaignAllPositionsManual.csv'):
        print('file missing for manually extrapolated data')
        exit()
    else:
        erosion = pd.read_csv('results/erosionFullCampaign/totalErosionWholeCampaignAllPositionsManual.csv', sep=';')
    for counter, config in enumerate(configurations):
        erosion_position = np.hstack(np.nansum(np.dstack((np.array(erosion[config + '_erosion'][:36]), erosion_position)), 2))
        deposition_position = np.hstack(np.nansum(np.dstack((np.array(erosion[config + '_deposition'][:36]), deposition_position)), 2))
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
    plt.savefig('results/erosionFullCampaign/totalErosionWholeCampaignAllPositionsLowIotaManual.png', bbox_inches='tight')
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
    plt.savefig('results/erosionFullCampaign/totalErosionWholeCampaignAllPositionsHighIotaManual.png', bbox_inches='tight')
    plt.show()
    plt.close()
    '''
#get impurity concentration trends from Hexos by looking at reference discharges
#extensions.readHexosForReferenceDischarges()

#get impurity concentrations for C and O from CXRS for "impurityReferenceShots"
#for shot in impurityReferenceShots:
#    src.CXRS.readImpurityConcentrationFromCXRS(shot)


#reality check
#ne, Te, Ts, Ts2, timesteps = np.array([23.02 * 1e+18]), np.array([13.26]), np.array([550 + 273.15]), np.array([150 + 273.15]), np.array([10])
#Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(Te, Te, Ts, ne, timesteps, alpha, m_i, f_i, ions, k, n_target)
#print(Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness)
#Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(Te, Te, Ts2, ne, timesteps, alpha, m_i, f_i, ions, k, n_target)
#print(Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness)

#######################################################################################################################################################################
#######################################################################################################################################################################
#PROGRAM CODE FOR OP1.2b EXAMPLE DISCHARGES
#Markus data read in, "intervals" defines how many adjacent values are averaged
#MarkusFunctions.processMarkusData(m_i, f_i, ions, k, n_target, interval = 50)

#######################################################################################################################################################################
#######################################################################################################################################################################
#references for look-up values
# Ref.1: D. Naujoks. Plasma-Material Interaction in Controlled Fusion (Vol. 39 in Springer Series on Atomic, Optical, and Plasma Physics). Ed. by G. W. F. Drake, Dr. G. Ecker, and Dr. H. Kleinpoppen. Springer-Verlag Berlin Heidelberg, 2006. 

#DURCHSCHNITTLICHE ENTLADUNG NEHMEN UND ABTRAGUNG FÜR GANZE KAMPAGNE HOCHRECHNEN

#BERICHT: BILDER EINFÜGEN
