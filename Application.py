''' This file is responsible for the final calculation of sputtering yields and the thickness of the erosion/deposition layer. 
    It processes the data with ProcessData using the functions defined in SputteringYieldFunctions after reading the data with ReadArchiveDB'''

import shutil
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

from src.settings import e, k_B, k, u, E_TF, E_s, Q_y_chem, C_d_chem, E_thd_chem, E_ths_chem, E_th_chem, lambda_nr, lambda_nl, m_i, ions
#import src.MarkusFunctions as MarkusFunctions
import src.Unused as extensions
import src.SputteringYieldFunctions as calc
import src.ProcessData as process
import src.ReadArchieveDB as read
import src.PlotData as plot
#import src.CXRS
from src.PositionsLangmuirProbes import OP2_TM2Distances, OP2_TM3Distances, OP2_TM8Distances, OP2_TM2zeta_lowIota, OP2_TM3zeta_lowIota, OP2_TM8zeta_lowIota, OP2_TM2zeta_standardIota, OP2_TM3zeta_standardIota, OP2_TM8zeta_standardIota, OP2_TM2zeta_highIota, OP2_TM3zeta_highIota, OP2_TM8zeta_highIota, OP2_TM2xyz, OP2_TM3xyz, OP2_TM8xyz

#avoids pop up of plot windows
matplotlib.use('agg')

#######################################################################################################################################################################
#######################################################################################################################################################################
#INPUT NOT TO BE CHANGED

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

#######################################################################################################################################################################
#######################################################################################################################################################################
#VARIABLE INPUT PARAMETERS
#ion concentrations (no unit) for [H, D, T, C, O]
f_iList = [[0.93, 0.0, 0.0, 0.02, 0.01]]#,[0.91, 0.0, 0.0, 0.03, 0.01], [0.95, 0.0, 0.0, 0.01, 0.01]]#[0.868, 0, 0, 0.03, 0.0196]]#[[0.85, 0.0, 0.0, 0.06, 0.01], [0.87, 0.0, 0.0, 0.05, 0.01], [0.89, 0.0, 0.0, 0.04, 0.01], [0.91, 0.0, 0.0, 0.03, 0.01], [0.93, 0.0, 0.0, 0.02, 0.01], [0.95, 0.0, 0.0, 0.01, 0.01]]#[[0.933, 0, 0, 0.0335, 0.0], [0.9364, 0, 0, 0.03, 0.0012], [0.9411, 0, 0, 0.025, 0.003], [0.9457, 0, 0, 0.02, 0.0047], [0.9504, 0, 0, 0.015, 0.0065]]# [0.89, 0, 0, 0.04, 0.01]]
#f_iList = [[0.85, 0.0, 0.0, 0.06, 0.01]]
#target density of CFC-HHF divertor in [1/m^3]
n_target = 9.526 * 1e28 #for rho=1.9 g/cm^3 (rho*1e6*N_A/M_C) N_A is avogadro number, M_C molecular mass of carbon

#incident angle of ions in rad
alpha = 2 * np.pi/9

#default values to be inserted if no measurement data exists for electron density ne in [1/m^3], electron temperature Te in [eV], or surface temperature of the target Ts in [K]
defaultValues = [np.nan, np.nan, 320, np.nan, np.nan]
if np.isnan(defaultValues[-1]):
    T_default = 'NAN'    #additional information about treatment of missing T_s values for "safe" of various files 
else:
    T_default = str(defaultValues[-1]) + 'K'

#list of all released configurations (date: 01. Dec. 2025) from W7X-info
configurations = pd.read_csv('inputFiles/Overview2.csv')['configuration']

#if iota information is missing, activate ('inputFiles/Overview4.csv' with columns 'IA [A]' and 'IB [A]' must exist):
#read.determineIotaForAllConfigurations()

#list of campaigns to be looked at, '' means OP2.2 and OP2.3
campaigns = ['OP23', 'OP22']#['', 'OP22', 'OP23']

#reference discharges for impurity concentration analysis of carbon and oxygen by CXRS and DRGA
impurityReferenceShots = ['20250304.075', '20250408.055', '20250408.079']

HeBeamReferenceShots = ['20250507.007', '20250507.009', '20250508.071', '20250320.077']

excluded = []#['20241105.027', '20241105.028', '20241105.029', '20241105.030', '20241105.031', '20241105.032', '20241105.033', '20241105.034', '20241105.035', '20241105.036', '20241105.037', '20241105.038', '20241105.039', '20241105.040', '20241105.041', '20241105.043', '20241105.058', '20241105.061', '20241105.062', '20241105.063', '20241105.065', '20241105.066', '20241105.067', '20241105.068', '20241106.023', '20241106.028', '20241128.068', '20241128.070']

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
filesExist = True                   

#should the measurement values be read out again?
#set True only if there was a problem with the reading routine
#if False, then already read out data wont be downloaded again
reReadData = True

#are missing values of ne, Te, Ts already intrapolated for the given combination of n_target, f_i, alpha (at least partially, missing files will be created anyways)? 
#-> results/calculationTablesNew/results*.csv
#set this to False when you changed n_target, f_i, alpha, or zeta
intrapolated = False

#are ne, Te, Ts averages are already calculated for all campaigns?
avCalculated = False

#should ne, Te, Ts, Y, Delta_ero, Delta_dep,... be plotted for original and extrapolated data?
#-> results/plots/*png
plottingOriginalData = False
plottingExtrapolatedData = False

#should the main program run or is just some other testing going on? -> only commands above "if not run:" will be executed
run = True

'''
LP_position = [OP2_TM2Distances]
LP_position.append(OP2_TM3Distances)
LP_position.append(OP2_TM8Distances)
LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list
#LP_position: index 0 - 5 are langmuir probes on TM2h07, 6 - 13 on TM3h01, 14 - 17 on TM8h01 (distance from pumping gap is increasing)

#read incident angles of the magnetic field lines on the targets zeta (measured from the target surface towards the surface normal) in [rad] 
LP_zeta_low = [OP2_TM2zeta_lowIota]
LP_zeta_low.append(OP2_TM3zeta_lowIota)
LP_zeta_low.append(OP2_TM8zeta_lowIota)

LP_zeta_standard = [OP2_TM2zeta_standardIota]
LP_zeta_standard.append(OP2_TM3zeta_standardIota)
LP_zeta_standard.append(OP2_TM8zeta_standardIota)

LP_zeta_high = [OP2_TM2zeta_highIota]
LP_zeta_high.append(OP2_TM3zeta_highIota)
LP_zeta_high.append(OP2_TM8zeta_highIota)

LP_zeta_low = list(itertools.chain.from_iterable(LP_zeta_low))
LP_zeta_standard = list(itertools.chain.from_iterable(LP_zeta_standard))
LP_zeta_high = list(itertools.chain.from_iterable(LP_zeta_high))
LP_zetas = [LP_zeta_low, LP_zeta_standard, LP_zeta_high]
#same indices as LP_position

print(process.calculateTotalErodedLayerThicknessFromOverviewFile(m_i, f_iList, alpha, LP_zetas[1], n_target, 'OP23', 'all', True, LP_position))
'''
#######################################################################################################################################################################
#HERE IS THE RUNNING PROGRAM, NO CHANGES REQUIRED

#plot.plotSputteringYieldsInDependence(5e+18, 15, 320, alpha, np.deg2rad(2), m_i, f_i, ions, k, 1e+29)
#plot.plotEnergyDistribution(15, ['H', 'C', 'O'])
#print(calc.calculateTotalErosionYield('H', 15, 'C', alpha, 320, 6.66e+21, 1e+29, True))
#print(calc.calculateTotalErosionYield('C', 15, 'C', alpha, 320, 6.66e+21, 1e+29, True))
#print(calc.calculateTotalErosionYield('O', 15, 'C', alpha, 320, 6.66e+21, 1e+29, True))

if not run:
    exit()

if __name__ == '__main__':
    for f_i in f_iList:
        #36 Langmuir Probes (LPs) in OP2 are located in module 5 upper and lower divertor unit (symmetry)
        #read langmuir probe positions in [m] from pumping gap
        LP_position = [OP2_TM2Distances]
        LP_position.append(OP2_TM3Distances)
        LP_position.append(OP2_TM8Distances)
        LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list
        #LP_position: index 0 - 5 are langmuir probes on TM2h07, 6 - 13 on TM3h01, 14 - 17 on TM8h01 (distance from pumping gap is increasing)

        #read incident angles of the magnetic field lines on the targets zeta (measured from the target surface towards the surface normal) in [rad] 
        LP_zeta_low = [OP2_TM2zeta_lowIota]
        LP_zeta_low.append(OP2_TM3zeta_lowIota)
        LP_zeta_low.append(OP2_TM8zeta_lowIota)
        
        LP_zeta_standard = [OP2_TM2zeta_standardIota]
        LP_zeta_standard.append(OP2_TM3zeta_standardIota)
        LP_zeta_standard.append(OP2_TM8zeta_standardIota)
        
        LP_zeta_high = [OP2_TM2zeta_highIota]
        LP_zeta_high.append(OP2_TM3zeta_highIota)
        LP_zeta_high.append(OP2_TM8zeta_highIota)
        
        LP_zeta_low = list(itertools.chain.from_iterable(LP_zeta_low))
        LP_zeta_standard = list(itertools.chain.from_iterable(LP_zeta_standard))
        LP_zeta_high = list(itertools.chain.from_iterable(LP_zeta_high))
        LP_zetas = [LP_zeta_low, LP_zeta_standard, LP_zeta_high]
        #same indices as LP_position

        averageFrame = pd.DataFrame({}) #will hold average values of ne, Te, Ts at all LP positions for each campaign (total, EIM, FTM, DMB, KJM)
        erodedMass = []    
        configurationOverviewTable = pd.read_csv('inputFiles/Overview4.csv', sep=';')
        for campaign in campaigns:      
            configurations_OP = []  
            for configuration in configurations:     
                #find all discharge IDs according to the filters activated in "filterSelected" 
                #usually filters by !conditioning, !gas valve tests, !sniffer tests, and configuration (internal filter of "read.readAllShotNumbersFromLogbook")
                print('Logbook search: reading discharges for ' + configuration)
                discharges = read.readAllShotNumbersFromLogbook(configuration, filterSelected, q_add=campaign, filesExist=filesExist)
                
                #returns either a pd.DataFrame with dischargeID, duration, overviewTable,... or a string that no discharges were found    
                if type(discharges) == str:
                    continue    #skip this configuration as no discharges were found
                else:
                    configurations_OP.append(configuration)

                    indexIota = list(configurationOverviewTable['configuration']).index(configuration)
                    if list(configurationOverviewTable['iota'])[indexIota] == 'low':
                        LP_zeta = LP_zetas[0]
                    elif list(configurationOverviewTable['iota'])[indexIota] == 'standard':
                        LP_zeta = LP_zetas[1]
                    elif list(configurationOverviewTable['iota'])[indexIota] == 'high':
                        LP_zeta = LP_zetas[2]

                    pass    #continue with this configuration as some discharges were found

                #_lower indicates lower divertor unit, _upper upper divertor unit   
                
                #will hold the dischargeIDs that miss data because LP data/IRcam data/trigger data is not available 
                #for checking only, is not used in any other way than being printed to the terminal
                not_workingLP, not_workingIR, not_workingTrigger = [], [], []

                #run through discharges and read LP and IRcam data before performing calculations with them
                for counter, discharge in enumerate(discharges['dischargeID'][:]):
                    discharge = str(discharge)

                    if discharge[-2] == '.':
                        discharge = discharge + '00'
                    elif discharge[-3] == '.':
                        discharge = discharge + '0'

                    '''
                    if discharge in excluded:
                        plottingOriginalData = True
                        #plottingExtrapolatedData = True
                    else:
                        plottingOriginalData = False
                        #plottingExtrapolatedData = False
                    '''

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
                    ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper, index_lower, index_upper, sne_lower, sne_upper, sTe_lower, sTe_upper = return_LP

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
                            sTe_lower[j].append(0)
                            sne_lower[j].append(0)
                    for j in range(len(ne_upper)):
                        for i in range(len(time_array) - len(ne_upper[j])):
                            ne_upper[j].append(0)
                            sne_upper[j].append(0)
                            Te_upper[j].append(0)
                            sTe_upper[j].append(0)
                            t_upper[j].append(0)

                    Te_lower = np.array(Te_lower)
                    sTe_lower = np.array(Te_lower)
                    Te_upper = np.array(Te_upper)
                    sTe_upper = np.array(Te_upper)
                    ne_lower = np.array(ne_lower)
                    sne_lower = np.array(ne_lower)
                    ne_upper = np.array(ne_upper)
                    sne_upper = np.array(ne_upper)
                    t_lower = np.array(t_lower)
                    t_upper = np.array(t_upper)
                    #print(np.shape(Ts_lower), np.shape(Ts_upper), np.shape(ne_upper), np.shape(ne_lower), np.shape(Te_upper), np.shape(Te_lower))

                    #calculate sputtering related physical quantities (sputtering yields, erosion rates, layer thicknesses)
                    #all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
                    #ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s], Ts in [K], alpha in [rad], LP_position in [m] from pumping gap, m in [kg], k in [eV/K], n_target in [1/m^3]
                    #does not return something but writes measurement values and calculated values to 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)
                    #plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
                    print('calculationTables: create ' + configuration + ' ' + discharge)
                    process.processOP2Data(str(discharge), ne_lower, ne_upper, Te_lower, Te_upper, sne_lower, sne_upper, sTe_lower, sTe_upper, Ts_lower, Ts_upper, t_lower, t_upper, index_lower, index_upper, alpha, LP_position, LP_zeta, m_i, f_i, ions, k, n_target, plotting=plottingOriginalData)

                print('configuration {config}: no trigger {trigger}, no IR {IR}, no LP {LP}'.format(config=configuration, trigger=len(not_workingTrigger), LP=len(not_workingLP), IR=len(not_workingIR)))
                
                #intrapolate missing values, calculate total eroded layer thickness, saves that for all discharges of one configuration in a table
                erosionTable = process.calculateTotalErodedLayerThicknessSeveralDischarges(configuration, discharges['dischargeID'], discharges['duration'], discharges['overviewTable'], alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, intrapolated=intrapolated, plotting=plottingExtrapolatedData, excluded=excluded)
                
                #sums up all layer thicknesses of all discharges of that configuration to end up with one value for erosion/deposition/net erosion per LP
                #pays attention to discharges with missing data by setting Delta_total = Delta_known * t_total/t_known
                print(process.calculateTotalErodedLayerThicknessWholeCampaignPerConfig(configuration, campaign))
            
            #configuration percentages of total runtime in OP2.2/2.3
            read.getRuntimePerConfiguration(configurations, campaign)

            #get average ne, Te, Ts distribution per configuration
            quantities = ['ne', 'Te', 'Ts']
            for config_short in [False, 'EIM', 'FTM', 'DBM', 'KJM']:
                averages = process.frameCalculateAverageQuantityPerConfiguration(quantities, [campaign], configurations, LP_position, config_short, excluded, avCalculated)
                if type(config_short) == bool:
                    config_short = 'all'
                if campaign == '':
                    campaignTXT = 'OP223'
                else:
                    campaignTXT = campaign
                for i, quantity in enumerate(quantities):
                    averageFrame[campaignTXT + config_short + quantity] = averages[0][i]
                    averageFrame[campaignTXT + config_short + quantity + 'Std'] = averages[1][i]
            
            #sums up all layer thicknesses of all configurations
            for configLayerThickness, zeta in zip(['all', 'EIM', 'FTM', 'DBM', 'KJM'], [LP_zetas[1], LP_zetas[1], LP_zetas[2], LP_zetas[0], LP_zetas[1]]):
                erodedMass.append(process.calculateTotalErodedLayerThicknessWholeCampaign(n_target, m_i, f_i, alpha, zeta, configurations_OP, configLayerThickness, LP_position, campaign, T_default, False))    

            #when both campaigns are treated together first, all result/calculationTableNew files are already intrapolated with the corrected parameters and it must not be done again for OP2.2 and OP2.3    
            if campaigns[0] == '':
                intrapolated = True

        averageFrame.to_csv('results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv', sep=';')

        #print(erodedMass[0])
        massFile = open('results/masses.txt', 'a')
        for campaignIndex, OP in enumerate(campaigns):
            massFile.write(str(f_i[0]) + ', ' + str(f_i[3]) + ', ' + str(f_i[4]) + ': ')
            massFile.write(f'mass of net eroded/eroded/deposited material in g for {OP} all configurations: ' + str(erodedMass[campaignIndex*5][0]) +' / '+ str(-erodedMass[campaignIndex*5][1]) +' / '+ str(erodedMass[campaignIndex*5][2]) + '\n')
        massFile.close()

        conc = str(f_i[0]).split('.')[1] + '_' + str(f_i[3]).split('.')[1] + '_' + str(f_i[4]).split('.')[1]
        shutil.copytree('results/averageQuantities', f'results{OP}_{conc}/averageQuantities')
        shutil.copytree('results/calculationTablesNew', f'results{OP}_{conc}/calculationTablesNew')
        shutil.copytree('results/erosionMeasuredConfig', f'results{OP}_{conc}/erosionMeasuredConfig')
        shutil.copytree('results/erosionExtrapolatedConfig', f'results{OP}_{conc}/erosionExtrapolatedConfig')
        shutil.copytree('results/erosionFullCampaign', f'results{OP}_{conc}/erosionFullCampaign')

        if f_i == f_iList[0]:
            avCalculated = True
    #compare Langmuir Probe and HeBeam data for discharges given in HeBeamReferenceShots
    LPxyz = [OP2_TM2xyz]
    LPxyz.append(OP2_TM3xyz)
    LPxyz = list(itertools.chain.from_iterable(LPxyz))

    for shot in HeBeamReferenceShots:
        read.compareLangmuirProbesWithHeBeam(LPxyz, shot) #incomplete for LPs and different timesteps

    #subresults (flux densities, physical and chemical sputtering yield, eroded and deposited layer thicknesses) for a set of ne, Te, Ts, alpha, zeta, t
    #process.subresults(m_i, f_i, ions)

    #parameter studies
    f_i = f_iList[2]
    neList = np.linspace(1e+18, 1e+20, 20)
    TeList = np.linspace(10, 20, 20)
    TsList = np.linspace(30+273.15, 700+273.15, 20)
    alphaList = np.linspace(30 * np.pi/180, 70 * np.pi/180, 20)
    zetaList = np.linspace(1 * np.pi/180, 3 * np.pi/180, 20)
    timestep = np.array([50000]) #duration of the erosion period (e.g. plasma time in campaign OP2.2 and OP2.3)
    plot.parameterStudy(neList, TeList, TsList, alphaList, zetaList, m_i, f_i, ions, k, n_target)

    #approximation of layer thicknesses when average distribution of ne, Te, Ts is assumed
    mass = []
    for campaign in ['OP22', 'OP23', '']:
        for configuration, LP_zeta in zip(['all', 'EIM', 'FTM', 'DBM', 'KJM'], [LP_zetas[1], LP_zetas[1], LP_zetas[2], LP_zetas[0], LP_zetas[1]]):
            for f_i_vary in [f_i]:
                for alpha_vary in [alpha]:
                    mass.append(process.approximationOfLayerThicknessesBasedOnAverageParameterValues(LP_position, campaign, alpha_vary, LP_zeta, m_i, f_i_vary, ions, k, n_target, 'results/averageQuantities/averageParametersPerCampaignPerConfiguration.csv', configuration))
                    print(f_i_vary, alpha_vary, mass[-1])
    #get impurity concentration trends from Hexos by looking at reference discharges
    #extensions.readHexosForReferenceDischarges()
    #extensions.readHexosForReferenceDischargesAveraged()

    #get impurity concentrations for C and O from CXRS for "impurityReferenceShots"
    #for shot in impurityReferenceShots:
    #    src.CXRS.readImpurityConcentrationFromCXRS(shot)

#######################################################################################################################################################################
#######################################################################################################################################################################
#PROGRAM CODE FOR OP1.2b EXAMPLE DISCHARGES
#Markus data read in, "intervals" defines how many adjacent values are averaged, "alpha" is incident angle of ions in [rad]
#"LP_zeta" incident angle of magnetic field lines on targets at each Langmuir Probe position in [rad]
#from src.PositionsLangmuirProbes import OP1_TM3zeta, OP1_TM4zeta
#LP_zeta = [OP1_TM3zeta]
#LP_zeta.append(OP1_TM4zeta)
#LP_zeta = list(itertools.chain.from_iterable(LP_zeta))
#MarkusFunctions.processMarkusData(alpha, LP_zeta, m_i, f_i, ions, k, n_target, interval = 50)

#######################################################################################################################################################################
#######################################################################################################################################################################
#references for look-up values
# Ref.1: D. Naujoks. Plasma-Material Interaction in Controlled Fusion (Vol. 39 in Springer Series on Atomic, Optical, and Plasma Physics). Ed. by G. W. F. Drake, Dr. G. Ecker, and Dr. H. Kleinpoppen. Springer-Verlag Berlin Heidelberg, 2006. 

'''
nothing excluded, T_default = 320 K, f_i = [0.89, 0, 0, 0.04, 0.01], alpha = 40°, zeta according to M.Endler (compare distance from PG)
mass of net eroded/eroded/deposited material in g for  all configurations: -51.548842080843706 / -284.4943985212966 / 232.94555644045295
mass of net eroded/eroded/deposited material in g for OP22 all configurations: -41.05019345273703 / -157.2008130963548 / 116.15061964361779
mass of net eroded/eroded/deposited material in g for OP23 all configurations: -13.90214578745139 / -128.47154645796073 / 114.56940067050932
'''