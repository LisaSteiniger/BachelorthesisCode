''' This file is responsible for the final calculation of sputtering yields and the thickness of the erosion layer. 
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

import src.SputteringYieldFunctions as calc
import src.ProcessData as process
import src.ReadArchieveDB as read
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

#heat of sublimation for [Be, C, Fe, Mo, W], in [eV] according to Ref. 1
E_s = np.array([3.38, 7.42, 4.34, 6.83, 8.68])

#Parameters for chemical erosion of C by H-isotopes, [H, D, T] according to Ref. 1
Q_y_chem = np.array([0.035, 0.1, 0.12])                                                                  
C_d_chem = np.array([250, 125, 83])                                                                               
E_thd_chem = [15, 15, 15]   #threshold energy for Y_damage                                                                      
E_ths_chem = [2, 1, 1]      #threshold energy for Y_surf                                                                        
E_th_chem=[31, 27, 29]   

#target density of CFC-HHF divertor in [1/m^3]
n_target = 9.526*1e28 #for rho=1.9 g/cm^3 (rho*1e6*N_A/M_C) N_A is avogadro number, M_C molecular mass of carbon

#Parameters for net erosion specifically for divertor
lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct

#######################################################################################################################################################################
#######################################################################################################################################################################
#VARIABLE INPUT PARAMETERS
#incident angle in rad
alpha = 2 * np.pi/9

configurations = ['EIM000-2520', 'EIM000-2620', 'KJM008-2520', 'KJM008-2620', 'FTM000-2620', 'FTM004-2520', 'DBM000-2520', 'FMM002-2520',
                  'EIM000+2520', 'EIM000+2620', 'EIM000+2614', 'DBM000+2520', 'KJM008+2520', 'KJM008+2620', 'XIM001+2485', 'MMG000+2520', 
                  'DKJ000+2520', 'IKJ000+2520', 'FMM002+2520', 'KTM000+2520', 'FTM004+2520', 'FTM004+2585', 'FTM000+2620', 'AIM000+2520', 'KOF000+2520']

run = True

#######################################################################################################################################################################
#HERE IS THE RUNNING PROGRAM
#read cryopumps
'''
urlCVP = {'inletT': "ArchiveDB/raw/W7X/ControlStation.2139/BCA.5_DATASTREAM/0/BCA5_20TI8200",
          'returnT': "ArchiveDB/raw/W7X/ControlStation.2139/BCA.5_DATASTREAM/1/BCA5_21TI8200"}
for shotnumber in ['20250304.066', '20250304.075', '20250311.023', '20250313.031', '20250401.037', '20250403.030', '20250403.046',
                   '20250403.076', '20250408.055', '20250514.026', '20250514.029', '20250514.032', '20250514.037', '20250522.053', '20250522.055']:
    data = w7xarchive.get_signal_for_program(urlCVP, shotnumber)
    print(shotnumber)
    print(np.mean(np.array(data['inletT'][1])))
    print(np.mean(np.array(data['returnT'][1])))

LP_position = [OP2_TM2Distances]
LP_position.append(OP2_TM3Distances)
LP_position.append(OP2_TM8Distances)
LP_position = list(itertools.chain.from_iterable(LP_position)) 

#plot manually extrapolated erosion over whole campaign
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

if not run:
    exit()

if __name__ == '__main__':
    #read langmuir probe positions in [m] from pumping gap
    #index 0 - 5 are langmuir probes on TM2h, 6 - 13 on TM3h, 14 - 17 on TM8h (distance from pumping gap is increasing)
    LP_position = [OP2_TM2Distances]
    LP_position.append(OP2_TM3Distances)
    LP_position.append(OP2_TM8Distances)
    LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list
    
    for configuration in configurations:
        
        discharges = read.readAllShotNumbersFromLogbook(configuration)
        if type(discharges) == str:
            continue
        
        #_lower indicates lower divertor unit, _upper upper divertor unit    
        not_workingLP, not_workingIR, not_workingTrigger = [], [], []
        for counter, discharge in enumerate(discharges['dischargeID'][:]):
            
            #tests if data was already read out and saved, in that case reading data is skipped for the discharge in question
            if os.path.isfile('results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)):
                continue
            
            #in case that data for this discharge was not read before

            #read Langmuir Probe data from xdrive
            #all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
            #ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s]
            #index represent the active LPs on each divertor unit
            return_LP = read.readLangmuirProbeDataFromXdrive(discharge)
            if type(return_LP) == str:
                not_workingLP.append([discharge, counter])
                continue
            
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
            #Ts in [K], LP_position in [m] from pumping gap, time_array in [s]
            return_IR = read.readSurfaceTemperatureFramesFromIRcam(discharge, time_array, ['lower', 'upper'], LP_position, [index_lower, index_upper])
            if type(return_IR[-1]) == str:
                if return_IR == 'incorrected trigger':
                    not_workingTrigger.append([discharge, counter])
                else:
                    not_workingIR.append([discharge, counter])
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
            process.processOP2Data(discharge, ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, index_lower, index_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, True)

        print('configuration {config}: no trigger {trigger}, no IR {IR}, no LP {LP}'.format(config=configuration, trigger=len(not_workingTrigger), LP=len(not_workingLP), IR=len(not_workingIR)))
        
        #intrapolate missing values, calculate total eroded layer thickness
        erosionTable = process.calculateTotalErodedLayerThicknessSeveralDischarges(configuration, discharges['dischargeID'], discharges['duration'], discharges['overviewTable'], alpha, m_i, f_i, ions, k, n_target, intrapolated=False, plotting=True)
        
        print(process.calculateTotalErodedLayerThicknessWholeCampaignPerConfig(configuration))
    
    process.calculateTotalErodedLayerThicknessWholeCampaign(configurations, LP_position)    

    #configuration percentages of total runtime in OP2.2/2.3
    read.getRuntimePerConfiguration(configurations=configurations)

    #plot manually extrapolated erosion over whole campaign
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
#juice files saved under followig paths for OP2.2/2.3
juicePath = [r"\\share\groups\E3\Diagnostik\DIA-3\QRH_PWI\experiment_analysis\BA Lisa Steiniger\Programmes\PythonScript_LisaSteiniger\BachelorthesisCode\inputFiles\Juice\adb_juice_op22_step_0.2_v0.5_redux.csv",
             r"\\share\groups\E3\Diagnostik\DIA-3\QRH_PWI\experiment_analysis\BA Lisa Steiniger\Programmes\PythonScript_LisaSteiniger\BachelorthesisCode\inputFiles\Juice\adb_juice_op23_step_0.2_v0.6_redux.csv"]

#dischargeID
discharges = read.readAllShotNumbersFromLogbook() #for all discharges filtered by Dirk logbook_search, adjust filters directly in function
read.getRuntimePerConfiguration(discharges)

#discharges = read.readAllShotNumbersFromJuice(juicePath) #samples all discharge IDs from juice files under "juicePath", their configuration and duration, rerurned as dictoinary with keys 'dischargeID', 'configuration', 'duration'
dischargeIDs = discharges['dischargeID']

#dischargeIDs = ['W7X20241001.046', 'W7X20250304.086']#W7X20241022.046']   #20241127.034' #manually

#reality check
#ne, Te, Ts, Ts2, timesteps = np.array([23.02 * 1e+18]), np.array([13.26]), np.array([550 + 273.15]), np.array([150 + 273.15]), np.array([10])
#Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(Te, Te, Ts, ne, timesteps, alpha, m_i, f_i, ions, k, n_target)
#print(Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness)
#Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(Te, Te, Ts2, ne, timesteps, alpha, m_i, f_i, ions, k, n_target)
#print(Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness)

run = False

#######################################################################################################################################################################
#######################################################################################################################################################################
#PROGRAM CODE FOR OP1.2b EXAMPLE DISCHARGES
#Markus data read in, "intervals" defines how many adjacent values are averaged
#processMarkusData(m_i, f_i, ions, k, n_target, interval = 50)

#######################################################################################################################################################################
#PROGRAM CODE FOR OP2 DISCHARGE GIVEN AS "DISCHARGE"

#discharge = ['20241022.050']
#dischargeIndex = list(dischargeIDs).index('W7X' + discharge[0])
#durations = [discharges['duration'][dischargeIndex]]
#overviewTables = [discharges['overviewTable'][dischargeIndex]]
#overviewTable = [pd.read_csv('results/calculationTables/results_{discharge}.csv'.format(discharge=discharge[0]), sep=';')]
#LPindices = process.findIndexLP(overviewTable)
#print(LPindices)
#overviewTable = process.intrapolateMissingValues(discharge, overviewTable, LPindices, alpha, m_i, f_i, ions, k, n_target)
#erosion = process.calculateTotalErodedLayerThicknessOneDischarge(discharge, duration, overviewTable, alpha, m_i, f_i, ions, k, n_target, intrapolated=False)
#discharge = ['W7X20241022.050']
#discharges = discharges[700:701]
#process.calculateTotalErodedLayerThicknessSeveralDischarges(discharge, durations, overviewTables, alpha, m_i, f_i, ions, k, n_target, intrapolated=False)
#process.calculateTotalErodedLayerThicknessSeveralDischarges(discharges['dischargeID'], discharges['duration'], discharges['overviewTable'], alpha, m_i, f_i, ions, k, n_target, intrapolated=True)
process.calculateTotalErodedLayerThicknessWholeCampaign()

if not run:
    exit()

if __name__ == '__main__':
    #_lower indicates lower divertor unit, _upper upper divertor unit

    #configuration percentages of total runtime in OP2.2/2.3
    #read.getRuntimePerConfiguration(discharges)

    #read langmuir probe positions in [m] from pumping gap
    #index 0 - 5 are langmuir probes on TM2h, 6 - 13 on TM3h, 14 - 17 on TM8h (distance from pumping gap is increasing)
    LP_position = [OP2_TM2Distances]
    LP_position.append(OP2_TM3Distances)
    LP_position.append(OP2_TM8Distances)
    LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list

    not_workingLP, not_workingIR, not_workingTrigger = [], [], []
    for counter, discharge in enumerate(dischargeIDs[:]):
        #cut 'W7X' in front of the discharge ID
        discharge = discharge[3:]
        print(not_workingTrigger)#not_workingLP, not_workingIR, 

        #read Langmuir Probe data from xdrive
        #all arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        #ne in [1/m^3], Te in [eV] and assumption that Te=Ti, t in [s]
        #index represent the active LPs on each divertor unit
        return_LP = read.readLangmuirProbeDataFromXdrive(discharge)
        if type(return_LP) == str:
            not_workingLP.append([discharge, counter])
            continue
        
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
        #Ts in [K], LP_position in [m] from pumping gap, time_array in [s]
        return_IR = read.readSurfaceTemperatureFramesFromIRcam(discharge, time_array, ['lower', 'upper'], LP_position, [index_lower, index_upper])
        if type(return_IR) == str:
            if return_IR == 'No IRcam stream for any time in discharge':
                not_workingIR.append([discharge, counter])
            else:
                not_workingTrigger.append([discharge, counter])
            continue
        Ts_lower, Ts_upper = return_IR
        
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
        process.processOP2Data(discharge, ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, index_lower, index_upper, alpha, LP_position, m_i, f_i, ions, k, n_target, True)

    print(not_workingTrigger, not_workingLP, not_workingIR)

    #intrapolate missing values, calculate total eroded layer thickness
    process.calculateTotalErodedLayerThicknessSeveralDischarges(discharges['dischargeID'], discharges['duration'], discharges['overviewTable'], alpha, m_i, f_i, ions, k, n_target, intrapolated=False)

#Warning from GitBash when heatflux code was added: LF will be replaced by CRLF the next time Git touches it
#######################################################################################################################################################################
#read data from archieveDB
'''
#######################################################################################################################################################################
#######################################################################################################################################################################
#references for look-up values
# Ref.1: D. Naujoks. Plasma-Material Interaction in Controlled Fusion (Vol. 39 in Springer Series on Atomic, Optical, and Plasma Physics). Ed. by G. W. F. Drake, Dr. G. Ecker, and Dr. H. Kleinpoppen. Springer-Verlag Berlin Heidelberg, 2006. 

#KONTROLLIERE DELTA_EROSION AUF FEHLENDE WERTE (20241022.049 upper15) 
#-> WENN FÜR EINEN ZEITPUNKT KEIN NE ODER SO VORLIEGT, MUSS INTRAPOLIERT WERDEN (DURCHSCHNITT VOM VORHERIGEN UND NACHFOLGENDEN WERT AUßER ES GIBT IHN NICHT, DANN NUR DEN EXISTENTEN WIEDERVERWENDEN)
#GENAUSO FÜR ALLE ANDEREN PARAMETER UND BIS ZUM ENDE DER ENTLADUNG
#-> SPRICH NACH DEM RAUSSCHREIBEN ÜBERARBEITEN? EXTRAPOLIERTE DATEN KENNZEICHNEN!!!
#FÜR EINE ENTLADUNG TESTEN, DANN FÜR ALLE

#DURCHSCHNITTLICHE ENTLADUNG NEHMEN UND ABTRAGUNG FÜR GANZE KAMPAGNE HOCHRECHNEN

#EROSIONSSCHICHTDICKE FÜR JEDE LP AUFSUMMIEREN -> WAS TUN MIT FEHLENDEN WERTEN FÜR GANZE ENTLADUNG?
#-> FÜR FTM LP0 BIS LP13 EGAL, FÜR EIM/KJM LP14 BIS LP17 EGAL? ANDERE KONFIGUARTIONEN EH EGAL? 
#-> LAUFZEIT FÜR JEDE LP IN JEDER KONFIGURATION BERECHNEN, ANTEIL VON GESAMTENTLADUNGSZEIT IN DIESER KONFIGURATION, MULTIPLIKATION UM AUF 100% ZU KOMMEN?
#-> PLOTTEN FÜR EINZELENTLADUNG UND ÜBER ALLE ENTLADUNGEN

#BERÜCKSICHTIGEN DER WIEDERABLAGERUNG

#RAUSSUCHEN VON ENTLADUNGEN FÜR IMPURITIES: EIM, LANG, STABILE PLASMAPARAMETER, ALLE LP UND IR DATEN VORHANDEN, IN OP2.3, KURZ NACH/ZWISCHEN/KURZ VOR BORIERUNG

#BERICHT: BILDER EINFÜGEN
