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

from src.settings import e, k_B, k, u, E_TF, E_s, Q_y_chem, C_d_chem, E_thd_chem, E_ths_chem, E_th_chem, lambda_nr, lambda_nl, m_i, ions
#import src.MarkusFunctions as MarkusFunctions
#import src.Unused as extensions
import src.SputteringYieldFunctions as calc
import src.ProcessData as process
import src.ReadArchieveDB as read
#import src.CXRS
from src.PositionsLangmuirProbes import OP2_TM2Distances, OP2_TM3Distances, OP2_TM8Distances, OP2_TM2zeta, OP2_TM3zeta, OP2_TM8zeta

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
f_i = [0.89, 0, 0, 0.04, 0.01]

#target density of CFC-HHF divertor in [1/m^3]
n_target = 9.526 * 1e28 #for rho=1.9 g/cm^3 (rho*1e6*N_A/M_C) N_A is avogadro number, M_C molecular mass of carbon

#incident angle of ions in rad
alpha = 2 * np.pi/9

#default values to be inserted if no measurement data exists for electron density ne in [1/m^3], electron temperature Te in [eV], or surface temperature of the target Ts in [K]
defaultValues = [np.nan, np.nan, 320]
if np.isnan(defaultValues[-1]):
    T_default = 'T_defaultToNAN'    #additional information about treatment of missing T_s values for "safe" of various files 
else:
    T_default = str(defaultValues[-1]) + 'K'

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
filesExist = True                   

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
plottingExtrapolatedData = False

#should the main program run or is just some other testing going on? -> only commands above "if not run:" will be executed
run = False

#######################################################################################################################################################################
#HERE IS THE RUNNING PROGRAM (SHOULD RUN WITHOUT INTERNAL CHANGES WHEN FINALLY FINISHED)
#reality check
#ne, Te, Ts, Ts2, timesteps = np.array([1.02 * 1e+18]), np.array([13.26]), np.array([30 + 273.15]), np.array([850 + 273.15]), np.array([50000])
#Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness1, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(Te, Te, Ts, ne, timesteps, alpha, m_i, f_i, ions, k, n_target)
#print(Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness)
#Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness2, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(Te, Te, Ts2, ne, timesteps, alpha, m_i, f_i, ions, k, n_target)
#print(Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness)
#print(erodedLayerThickness1[0] - erodedLayerThickness2[0])

#TEST RESULTS IF TEMPERATURE IS NOT SET TO 320K BY DEFAULT BUT TO NAN -> JUST LIKE ne AND Te IN SEVERALDISCHARGES

#HeBeam = np.load('inputFiles\HeBeam\he_beam_data_20250507.007_AEH51.npz')
#print(HeBeam['ne'])
#print(HeBeam['Te'])
'''
LP_position = [OP2_TM2Distances]
LP_position.append(OP2_TM3Distances)
LP_position.append(OP2_TM8Distances)
LP_position = list(itertools.chain.from_iterable(LP_position))

neOP22=[8.10861120e+18, 9.23699360e+18, 5.39394510e+18, 4.88005878e+18, 6.41963593e+19, 7.53689257e+18, 3.74975667e+18, 2.87621473e+18, 1.86239725e+18, 1.77501141e+18, 1.24395721e+18, 8.16433709e+17, 7.98950114e+17, 2.63917556e+18, 6.71595628e+18, 6.02079381e+18, 6.96774702e+18, 8.67705959e+18, 7.34334178e+18, 6.82833426e+18, 3.96166626e+18, 3.27682856e+18, 3.05768770e+18, 2.92846510e+18, 2.13915956e+18, 1.78086238e+18, 5.61754825e+19, 2.76740351e+18, 3.49056208e+18, 1.00981278e+18, 9.08946107e+17, 1.89625133e+19, 4.69150909e+18, 5.29825547e+18, 8.07835520e+18, 7.49023570e+18]
neOP23=[7.23425072e+18, 6.49082692e+18, 4.74422826e+18, 4.13011762e+18, 1.00974487e+19, 4.06601437e+18, 4.58674313e+18, 3.35729073e+18, 1.63927672e+18, 2.32379908e+18, 1.83177974e+18, 1.51632845e+18, 5.19738099e+17, 9.77853187e+17, np.nan , np.nan , 3.99357880e+18, np.nan , 8.19180641e+18, 8.43372562e+18, 2.26185620e+18, 2.21738383e+18, 1.88094868e+18, 1.87003597e+18, 4.06435840e+18, 3.09974771e+18, 4.07980777e+18, 1.88866226e+18, 1.50819279e+18, 1.52277035e+18, 9.89528048e+17, 2.13408464e+18, 4.46755927e+18, 5.46513067e+18, 6.51305893e+18, 7.82164729e+18]
ne=[7.67669550e+18, 7.87359786e+18, 5.06883517e+18, 4.50678007e+18, 4.18099586e+19, 5.84696312e+18, 4.16512236e+18, 3.11504844e+18, 1.75203223e+18, 2.04518742e+18, 1.53398982e+18, 1.16495639e+18, 6.57016472e+17, 1.64853914e+18, 6.71595628e+18, 6.02079381e+18, 4.21687487e+18, 8.67705959e+18, 7.73744930e+18, 7.59404418e+18, 3.13558215e+18, 2.76987019e+18, 2.49778347e+18, 2.42486796e+18, 3.05919152e+18, 2.40057371e+18, 3.24314296e+19, 2.35439188e+18, 2.55752020e+18, 1.23901201e+18, 9.53427533e+17, 1.03968918e+19, 4.47277606e+18, 5.45519706e+18, 6.60716485e+18, 7.79225959e+18]

TeOP22=[23.10917898, 23.97398849, 17.24971306, 12.9683621, 16.1148856, 15.14815498, 11.83223752, 11.13380081, 10.22305666, 8.8592773, 7.93170485, 6.82595981, 6.60610872, 4.80160941, 17.62515757, 16.13908412, 25.78049094, 20.23960462, 22.49694823, 23.95433543, 20.73571079, 16.05200695, 16.11300381, 15.77758246, 10.83166445, 9.89855224, 10.14166381, 7.22341548, 8.01914169, 6.33356499, 11.30071018, 4.04176992, 9.98670479, 11.42823382, 17.45413223, 25.34487122]
TeOP23=[22.01761885, 22.46042915, 16.40248943, 14.18213774, 17.20604823, 16.68423172, 12.41207622, 12.30609299, 14.7635153, 10.55438673, 10.20832988, 10.06609525, 26.80995318, 17.61596149, np.nan , np.nan , 29.90268815, np.nan , 22.29111767, 23.60803396, 21.16374238, 15.52334095, 15.09640528, 15.98598422, 13.43930376, 13.32485776, 13.91981687, 13.66644289, 10.7030418, 11.5603852, 30.29936855, 18.54352217, 9.81694099, 13.86485593, 18.71589961, 19.61945445]
Te=[22.57062858, 23.22354637, 16.82666691, 13.5712462, 16.62277776, 15.90181622, 12.11971609, 11.71505922, 12.4771496, 9.70027448, 9.0603626, 8.43274704, 16.30989349, 10.37925827, 17.62515757, 16.13908412, 29.64032044, 20.23960462, 22.39974752, 23.79127039, 20.93864322, 15.80200345, 15.63251635, 15.87614281, 12.0691916, 11.52400789, 11.92908508, 10.25756316, 9.29313957, 8.79815706, 20.29700216, 10.68119372, 9.8210141, 13.722815, 18.64225509, 19.97987108]

TsOP22=[348.53123333, 339.57542575, 327.58684535, 322.48441638, 320.37366489, 320.53570604, 316.64689784, 314.81928323, 313.70052689, 312.70113155, 312.37350838, 312.15832773, 311.93336754, 312.42157821, 333.43560488, 371.92962704, 433.55672356, 435.60782705, 357.57614827, 351.53122556, 336.32154333, 326.37350381, 323.99850832, 321.41492792, 316.27895614, 315.81451817, 314.67173041, 313.93423271, 312.82564765, 312.18334915, 312.06617817, 313.87187357, 379.7223462, 333.73038846, 357.42425781, 398.05262984]
TsOP23=[344.02238096, 331.31824545, 321.68943264, 318.01602097, 317.16175369, 318.8853121, 315.30428473, 313.76628092, 312.89937246, 311.64312308, 311.17382574, 310.84767811, 311.19797395, 311.8039927, np.nan , np.nan , 377.54746229, np.nan , 340.8332816, 341.27643279, 333.74961652, 322.96430675, 319.93706882, 318.74008531, 315.99218539, 315.65745321, 314.32493327, 312.87997257, 312.18616397, 311.9843921, 312.43478398, 314.37532688, 350.47602205, 354.20017493, 370.1831045, 386.89669786]
Ts=[346.28735421, 335.44220232, 324.63478839, 320.24767996, 318.77968648, 319.71480009, 315.97417336, 314.29167, 313.29910358, 312.17095723, 311.7723403, 311.50155343, 311.57167871, 312.1262853, 333.43560488, 371.92962704, 381.08950793, 435.60782705, 349.56278168, 346.6231405, 335.09058373, 324.74181514, 322.05464744, 320.13471141, 316.14129944, 315.7391233, 314.50492494, 313.4271456, 312.5185426, 312.08774737, 312.24329866, 314.11283431, 352.32557001, 352.90565834, 369.37623049, 387.60220297]

tOP23=28333.24
tOP22=20372.65
t=48705.89

for n_e, T_e, T_s, time, campaign in [[neOP22, TeOP22, TsOP22, tOP22, 'OP22'], [neOP23, TeOP23, TsOP23, tOP23, 'OP23'], [ne, Te, Ts, t, 'OP223']]:
    erosion, deposition = [], []
    for i in range(len(n_e)):
        Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness1, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = process.calculateErosionRelatedQuantitiesOnePosition(np.array([T_e[i]]), np.array([T_e[i]]), np.array([T_s[i]]), np.array([n_e[i]]), np.array([time]), alpha, m_i, f_i, ions, k, n_target)
        erosion.append(erodedLayerThickness1)
        deposition.append(depositedLayerThickness)
    plt.plot(LP_position[14:], 0 - np.array(erosion[14:18]), 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[14:], 0 - np.array(erosion[32:]), 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[14:], deposition[14:18], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[14:], deposition[32:], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[14:], 0 - np.array(erosion[14:18]) + np.array(deposition[14:18]), 'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[14:], 0 - np.array(erosion[32:]) + np.array(deposition[32:]), 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('layer thickness in (m)')
    plt.savefig('results/averageQuantities/{campaign}AverageAllPositionsHighIota.png'.format(campaign=campaign), bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(LP_position[:14], 0 - np.array(erosion[:14]), 'b--', label='erosion lower divertor unit')
    plt.plot(LP_position[:14], 0 - np.array(erosion[18:32]), 'm--', label='erosion upper divertor unit')
    plt.plot(LP_position[:14], deposition[:14], 'b:', label='deposition lower divertor unit')
    plt.plot(LP_position[:14], deposition[18:32], 'm:', label='deposition upper divertor unit')
    plt.plot(LP_position[:14], 0 - np.array(erosion[:14]) + np.array(deposition[:14]), 'b-', label='net erosion lower divertor unit')
    plt.plot(LP_position[:14], 0 - np.array(erosion[18:32]) + np.array(deposition[18:32]), 'm-', label='net erosion upper divertor unit')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel('layer thickness in (m)')
    plt.savefig('results/averageQuantities/{campaign}AverageAllPositionsLowIota.png'.format(campaign=campaign), bbox_inches='tight')
    plt.show()
    plt.close()

'''    
'''
process.calculateTotalErodedLayerThicknessWholeCampaign(configurations, LP_position)

overview = pd.read_csv('inputFiles/Overview4.csv', sep=';')
iota = []
for IA, IB in zip(overview['IA [A]'], overview['IB [A]']):
    if IA < -2000 and IB < -2000:
        iota.append('high')
    elif IA > 2000 and IB > 2000:
        iota.append('low')
    else:
        iota.append('standard')
overview['iota'] = iota
overview.to_csv('inputFiles/Overview4.csv', sep=';')

for quantity in ['ne', 'Te', 'Ts']:
    for campaign in ['OP23', 'OP22', '']:
        averageCampaign = [0.] * 36
        timeCampaign = [0.] * 36
        averageCampaign = np.array(averageCampaign)
        timeCampaign = np.array(timeCampaign)
        for config in configurations:
            resultAverage = process.calculateAverageQuantityPerConfiguration(quantity, config, LP_position, campaign)
            if type(resultAverage) == str:
                print(resultAverage)
            else:  
                timeCampaign = np.hstack(np.nansum(np.dstack((np.array(timeCampaign), np.array(resultAverage[1]))), 2))
                averageCampaign = np.hstack(np.nansum(np.dstack((np.array(averageCampaign), ((np.array(resultAverage[1])*np.array(resultAverage[0]))) )), 2))
    
        averageCampaign = averageCampaign/timeCampaign

        plt.plot(LP_position[14:], averageCampaign[14:18], 'b', label='lower divertor unit')
        plt.plot(LP_position[14:], averageCampaign[32:], 'm', label='upper divertor unit')
        plt.legend()
        plt.xlabel('distance from pumping gap (m)')
        plt.ylabel('average ' + quantity)
        plt.savefig('results/averageQuantities/{quantity}/{quantity}{campaign}AverageAllPositionsHighIota.png'.format(quantity=quantity, campaign=campaign), bbox_inches='tight')
        plt.show()
        plt.close()

        plt.plot(LP_position[:14], averageCampaign[:14], 'b', label='lower divertor unit')
        plt.plot(LP_position[:14], averageCampaign[18:32], 'm', label='upper divertor unit')
        plt.legend()
        plt.xlabel('distance from pumping gap (m)')
        plt.ylabel('average ' + quantity)
        plt.savefig('results/averageQuantities/{quantity}/{quantity}{campaign}AverageAllPositionsLowIota.png'.format(quantity=quantity, campaign=campaign), bbox_inches='tight')
        plt.show()
        plt.close()

        print(averageCampaign)
'''
if not run:
    exit()

if __name__ == '__main__':
    #36 Langmuir Probes (LPs) in OP2 are located in module 5 upper and lower divertor unit (symmetry)
    #read langmuir probe positions in [m] from pumping gap
    LP_position = [OP2_TM2Distances]
    LP_position.append(OP2_TM3Distances)
    LP_position.append(OP2_TM8Distances)
    LP_position = list(itertools.chain.from_iterable(LP_position)) #flattens to 1D list
    #LP_position: index 0 - 5 are langmuir probes on TM2h07, 6 - 13 on TM3h01, 14 - 17 on TM8h01 (distance from pumping gap is increasing)

    #read incident angles of the magnetic field lines on the targets zeta (measured from the target surface towards the surface normal) in [rad] 
    LP_zeta = [OP2_TM2zeta]
    LP_zeta.append(OP2_TM3zeta)
    LP_zeta.append(OP2_TM8zeta)
    LP_zeta = list(itertools.chain.from_iterable(LP_zeta))
    #same indices as LP_position
    
    for campaign in ['OP23', 'OP22', '']:      
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
                process.processOP2Data(str(discharge), ne_lower, ne_upper, Te_lower, Te_upper, Ts_lower, Ts_upper, t_lower, t_upper, index_lower, index_upper, alpha, LP_position, LP_zeta, m_i, f_i, ions, k, n_target, plotting=plottingOriginalData)

            print('configuration {config}: no trigger {trigger}, no IR {IR}, no LP {LP}'.format(config=configuration, trigger=len(not_workingTrigger), LP=len(not_workingLP), IR=len(not_workingIR)))
            
            #intrapolate missing values, calculate total eroded layer thickness, saves that for all discharges of one configuration in a table
            erosionTable = process.calculateTotalErodedLayerThicknessSeveralDischarges(configuration, discharges['dischargeID'], discharges['duration'], discharges['overviewTable'], alpha, LP_zeta, m_i, f_i, ions, k, n_target, defaultValues, intrapolated=intrapolated, plotting=plottingExtrapolatedData)
            
            #sums up all layer thicknesses of all discharges of that configuration to end up with one value for erosion/deposition/net erosion per LP
            #pays attention to discharges with missing data by setting Delta_total = Delta_known * t_total/t_known
            print(process.calculateTotalErodedLayerThicknessWholeCampaignPerConfig(configuration, campaign))
        
        #sums up all layer thicknesses of all configurations
        process.calculateTotalErodedLayerThicknessWholeCampaign(configurations_OP, LP_position, campaign, T_default)    

        #configuration percentages of total runtime in OP2.2/2.3
        read.getRuntimePerConfiguration(configurations, campaign)

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

#DURCHSCHNITTLICHE ENTLADUNG NEHMEN UND ABTRAGUNG FÜR GANZE KAMPAGNE HOCHRECHNEN

