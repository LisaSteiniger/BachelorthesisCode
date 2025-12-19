import w7xarchive
import os
import scipy
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.ProcessData as process
import src.PlotData as plot

if __name__ == '__main__':
    e  =  scipy.constants.elementary_charge 
    k_B = scipy.constants.Boltzmann
    k = k_B/e 

    #lines for [Be, C, Fe, Mo, W] and columns with [H, D, T, He, Self-Sputtering, O] (O only known for C), in [eV]
    E_TF = np.array([[256, 282, 308, 720, 2208, 0],
                    [415, 447, 479, 1087, 5688, 9298],
                    [2544, 2590, 2635, 5517, 174122, 0], 
                    [4719, 4768, 4817, 9945, 533127, 0],
                    [9871, 9925, 9978, 20376, 1998893, 0]]) 

    #[Be, C, Fe, Mo, W], in [eV]  
    E_s = np.array([3.38, 7.42, 4.34, 6.83, 8.68])

    #Parameters for chemical erosion of C by H-isotopes, [H, D, T] 
    Q_y_chem = np.array([0.035, 0.1, 0.12])                                                                  
    C_d_chem = np.array([250, 125, 83])                                                                               
    E_thd_chem = [15, 15, 15]   #threshold energy for Y_damage                                                                      
    E_ths_chem = [2, 1, 1]      #threshold energy for Y_surf                                                                        
    E_th_chem=[31, 27, 29]   

    #Parameters for net erosion specifically for divertor
    lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct

else:
    from src.settings import e, u, k_B, k, E_TF, E_s, Q_y_chem, C_d_chem, E_thd_chem, E_ths_chem, E_th_chem, lambda_nr, lambda_nl

#######################################################################################################################################################################
def readArchieveDB_ne_ECRH(dischargesCSV: pd.DataFrame):
    ''' NOT YET COMPLETED
        Should find absolute and relative plasma time in a certain combination of configuration, ECRH heating power and ne'''
    url = {'ECRH': 'Test/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V1/0/Ptot_ECRH',
           'ne': ''}
    
    for discharge in dischargesCSV['dischargeID']:
        data = w7xarchive.get_signal_for_program(url, discharge[3:]) 
        ne_time = data['ne'][0]
        ne = data['ne'][1]
        ECRH_time = data['ECRH'][0]
        ECRH = data['ECRH'][1]
        #now find steps in there and how long they take

#######################################################################################################################################################################
def readAllConfigurationsFromJuice(juicePath: list[str]) -> list[str]:
    ''' This function scans juice Data Base for all configurations being present there
        The naming is not identical with the naming in the Logbook and W7X-info
        Returns list with all configuration names'''   
    files = []
    for i in range(len(juicePath)):
        files.append(pd.read_csv(juicePath[i],  index_col=0))
    juice = pd.concat(files)

    return list(np.unique(juice['configuration']))
    
#######################################################################################################################################################################
def readAllShotNumbersFromJuice(juicePath: list[str], safe: str ='results/dischargeIDlistAll.csv') -> pd.DataFrame:   
    ''' if no file is saved under "safe", function reads the IDs of all discharges carried out in campaign(s) for which juice file(s) are given under "juicePath" and saves them as .csv file under "safe"
        if such a file is existent, the discharge IDs are read from it and returned
        returned object is identical in both cases: DataFrame with keys 'dischargeID', 'configuration', 'duration', 'overviewTable' (default safe for calculation tables of a discharge)'''
    if os.path.isfile(safe):
        return pd.read_csv(safe, sep=';')
    
    else:
        files = []
        for i in range(len(juicePath)):
            files.append(pd.read_csv(juicePath[i],  index_col=0))
        juice = pd.concat(files)

        dischargeIDlist = list(np.unique(juice['shot']))
        configurations = []
        durations, discharges, overviewTable = [], [], []
        for discharge in dischargeIDlist:
            shot = juice[juice['shot']==discharge]
            print(shot.head()['configuration'])
            discharges.append(list(shot['shot'])[0])
            configurations.append(list(shot['configuration'])[0])
            durations.append(list(shot['t_shot_stop'])[0])
            overviewTable.append('results/calculationTables/results_{discharge}.csv'.format(discharge=list(shot['shot'])[0][3:])) 

        dischargeIDcsv = pd.DataFrame({'dischargeID':dischargeIDlist, 'configuration':configurations, 'duration':durations, 'overviewTable':overviewTable})
        dischargeIDcsv.to_csv(safe, sep=';')

        return dischargeIDcsv

#######################################################################################################################################################################
def getRuntimePerConfigurationOld(dischargeIDcsv: pd.DataFrame, safe: str ='results/configurationRuntimes.csv') -> None:
    ''' This function determines the absolute and relative runtime of each configuration over all discharges given by "dischargeIDcsv" DataFrame and writes them as .csv file to "safe"
        The DataFrame contains all discharges of all configurations and has the keys 'dischargeID', 'duration', 'configuration', and 'overviewTables' (default safe for calculation tables of a discharge)'''
    #find all configurations present in the DataFrame
    configurations = list(np.unique(dischargeIDcsv['configuration']))
    configurations = list(np.unique([x[:3] for x in configurations])) #if B is not to important but only e.g. 'EIM'
    #print(configurations)

    runtimes = pd.DataFrame({})
    absoluteRuntimes, relativeRuntimes, absoluteNumberOfDischarges, relativeNumberOfDischarges = [], [], [], []
    totalRuntime = np.nansum(dischargeIDcsv['duration'])
    totalNumber = len(dischargeIDcsv['duration'])

    for configuration in configurations:
        filter = np.array([i.startswith(configuration) for i in dischargeIDcsv['configuration']])
        runtime = np.nansum(dischargeIDcsv[filter]['duration'])
        absoluteRuntimes.append(runtime)
        relativeRuntimes.append(runtime/totalRuntime * 100)
        absoluteNumberOfDischarges.append(sum(filter))
        relativeNumberOfDischarges.append((sum(filter)/totalNumber) * 100)

    configurations.append('all')
    absoluteRuntimes.append(totalRuntime)
    relativeRuntimes.append(100)
    absoluteNumberOfDischarges.append(totalNumber)
    relativeNumberOfDischarges.append(100)

    runtimes['configuration'] = configurations
    runtimes['absolute number of discharges'] = absoluteNumberOfDischarges
    runtimes['relative number of discharges (%)'] = relativeNumberOfDischarges
    runtimes['absolute runtime (s)'] = absoluteRuntimes
    runtimes['relative runtime (%)'] = relativeRuntimes
    runtimes = runtimes.sort_values('absolute runtime (s)')

    runtimes.to_csv(safe, sep=';')

#######################################################################################################################################################################
def readArchieveDB():
    #just example for using J. Brunners package w7xarchive to read ne and Te data from ArchiveDB
    shotnumbersOP1 = ['20181018.041']
    shotnumbersOP2 = ['20250424.050']

    data_urls_OP2 = {'LP_ne':"ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnit}_Probe_{number}_DATASTREAM/V1/1/Plasma_Density".format(divertorUnit=divertorUnitOP2, number=numberOP2),
                     'LP_Te':"ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnit}_Probe_{number}_DATASTREAM/V1/2/Electron_Temperature".format(divertorUnit=divertorUnitOP2, number=numberOP2)}
    
    data_urls_OP1 = {'LP_ne':"ArchiveDB/raw/Minerva/Minerva.ElectronDensity.QRP.{divertorUnit}.{number}/ne_DATASTREAM/V1/0/ne".format(divertorUnit=divertorUnitOP1, number=numberOP1),
                     'LP_Te':"ArchiveDB/raw/Minerva/Minerva.ElectronTemperature.QRP.{divertorUnit}.{number}/Te_DATASTREAM/V1/0/Te".format(divertorUnit=divertorUnitOP1, number=numberOP1)}
    
    divertorUnitsOP1 = ['lowerTestDivertorUnit', 'upperTestDivertorUnit'] #for OP1.2 Langmuir Probes, #numbersOP1 in range [1, 20]
    divertorUnitsOP2 = ['LowerDivertor', 'UpperDivertor'] #for OP2 Langmuir Probes, numbersOP2 in range [1, 14]
    
    for shotnumber in shotnumbersOP1:
        #for OP1
        for divertorUnitOP1 in divertorUnitsOP1:
            for numberOP1 in range(1, 2):#21):
                #called like this time is in seconds from t1 (program start?)
                #add error of the datastreams

                #enables parallel download if data_urls_OP1 is dictionary
                data_OP1 = w7xarchive.get_signal_for_program(data_urls_OP1, shotnumber) 
                #data_OP1['LP_ne'][0] to access time trace of ne measurement, data_OP1['LP_ne'][1] to access ne data
                #data_OP1['LP_Te'][0] to access time trace of Te measurement, data_OP1['LP_Te'][1]to access Te data
               
    for shotnumber in shotnumbersOP2:
        #for OP2
        for divertorUnitOP2 in divertorUnitsOP2:
            for numberOP2 in range(1, 2):#14):
                #called like this time is in seconds from t1
                data_OP2 = w7xarchive.get_signal_for_program(data_urls_OP2, shotnumber)            

#######################################################################################################################################################################
def readCryoPumpStatus(dischargeID: str|list[str]) -> tuple[str, float, float]|list[tuple[str, float, float]]:
    '''This function reads out the status of the cryopumps for a given discharge or list of discharges
        Returns a tuple or list of tuples with three elements each: dischargeID, inlet temperature, and returned temperature'''
    
    urlCVP = {'inletT': "ArchiveDB/raw/W7X/ControlStation.2139/BCA.5_DATASTREAM/0/BCA5_20TI8200",
            'returnT': "ArchiveDB/raw/W7X/ControlStation.2139/BCA.5_DATASTREAM/1/BCA5_21TI8200"}

    if type(dischargeID) == list:
        statusList = []
        for shotnumber in dischargeID:
            data = w7xarchive.get_signal_for_program(urlCVP, shotnumber)
            statusList.append([shotnumber, np.mean(np.array(data['inletT'][1])), np.mean(np.array(data['returnT'][1]))])
    else:
        data = w7xarchive.get_signal_for_program(urlCVP, dischargeID)
        statusList = [shotnumber, np.mean(np.array(data['inletT'][1])), np.mean(np.array(data['returnT'][1]))]

    return statusList

#######################################################################################################################################################################
def readHexosForReferenceDischarges(dischargeID: list[str] =['20241008.40', '20241009.9', '20241016.14', '20241114.10', '20241127.10', '20241205.9', '20241210.7', '20241212.9', 
                                                                '20250311.62', '20250312.71', '20250325.65', '20250424.6', '20250506.9', '20250513.6', '20250513.54', '20250514.10', '20250521.9']) -> None:
    ''' This function reads out the intensities of the spectral lines of CII and OIII ions in the core plasma measured by HEXOS for a given list of (reference) discharges
        -> this should help to find trends in the impurity concentrations of carbon and oxygen during OP2.2/2.3'''
    
    urlHexos = {'CII': "ArchiveDB/raw/W7XAnalysis/QSD_HEXOS/EmissionLinesSpec4_auto_DATASTREAM/V1/35/C-II_133.510nm_simpson",
                'OIII': "ArchiveDB/raw/W7XAnalysis/QSD_HEXOS/EmissionLinesSpec4_auto_DATASTREAM/V1/15/O-III_83.430nm_simpson"}

    fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(15, 10), sharey='row')
    for shotnumber in dischargeID:
        if float(shotnumber)<20250000:
            i = 0
        else:
            i = 1

        data = w7xarchive.get_signal_for_program(urlHexos, shotnumber)

        #try to average the signal for each phase of the reference discharge
        data_C, data_O, time_C, time_O = [], [], [], []
        steps_timeC = [[list(data['CII'][0]).index(0.2), list(data['CII'][0]).index(2.8)], [list(data['CII'][0]).index(3.2), list(data['CII'][0]).index(8.8)], [list(data['CII'][0]).index(9.2), list(data['CII'][0]).index(14.8)]]
        steps_timeO = [[list(data['OIII'][0]).index(0.2), list(data['OIII'][0]).index(2.8)], [list(data['OIII'][0]).index(3.2), list(data['OIII'][0]).index(8.8)], [list(data['OIII'][0]).index(9.2), list(data['OIII'][0]).index(14.8)]]
        
        for step_C, step_O in zip(steps_timeC, steps_timeO):
            time_C.append(data['CII'][0][step_C[0]:step_C[1]])
            time_O.append(data['OIII'][0][step_O[0]:step_O[1]])
            data_C.append(data['CII'][1][step_C[0]:step_C[1]])
            data_O.append(data['OIII'][1][step_O[0]:step_O[1]])

        #flatten lists
        list(itertools.chain.from_iterable(time_C))
        list(itertools.chain.from_iterable(data_C))
        list(itertools.chain.from_iterable(time_O))
        list(itertools.chain.from_iterable(data_O))

        ax[0][i].plot(time_C, data_C, label=shotnumber)
        ax[1][i].plot(time_O, data_O, label=shotnumber)

    ax[0][0].set_xlabel('time t in (s)')
    ax[0][1].set_xlabel('time t in (s)')
    ax[1][0].set_xlabel('time t in (s)')
    ax[1][1].set_xlabel('time t in (s)')
    ax[0][0].set_ylabel('intensity of C-II spectral line (133.510nm) OP2.2')
    ax[0][1].set_ylabel('intensity of C-II spectral line (133.510nm) OP2.3')
    ax[1][0].set_ylabel('intensity of O-III spectral line (83.430nm) OP2.2')
    ax[1][1].set_ylabel('intensity of O-III spectral line (83.430nm) OP2.3')
    ax[0][0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1][0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('results/impurities/HexosTrends_CII_OIII_throughReferenceDischarges.png', bbox_inches='tight')
    fig.show()
    plt.close()

#######################################################################################################################################################################   
def readHexosForReferenceDischargesAveraged(dischargeID: list[str] =['20241008.40', '20241009.9', '20241016.14', '20241114.10', '20241127.10', '20241205.9', '20241210.7', '20241212.9', 
                                                                '20250311.62', '20250312.71', '20250325.65', '20250424.6', '20250506.9', '20250513.6', '20250513.54', '20250514.10', '20250521.9']) -> None:
    ''' This function reads out the intensities of the spectral lines of CII and OIII ions in the core plasma measured by HEXOS for a given list of (reference) discharges
        -> this should help to find trends in the impurity concentrations of carbon and oxygen during OP2.2/2.3'''
    
    urlHexos = {'CII': "ArchiveDB/raw/W7XAnalysis/QSD_HEXOS/EmissionLinesSpec4_auto_DATASTREAM/V1/35/C-II_133.510nm_simpson",
                'OIII': "ArchiveDB/raw/W7XAnalysis/QSD_HEXOS/EmissionLinesSpec4_auto_DATASTREAM/V1/15/O-III_83.430nm_simpson"}

    fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(15, 10), sharey='row')
    for shotnumber in dischargeID:
        if float(shotnumber)<20250000:
            i = 0
        else:
            i = 1

        data = w7xarchive.get_signal_for_program(urlHexos, shotnumber)

        #try to average the signal for each phase of the reference discharge
        data_C, data_O, time_C, time_O = [], [], [], []
        steps_timeC = [[list(np.round(np.array(data['CII'][0]), 1)).index(0.2), 
                        list(np.round(np.array(data['CII'][0]), 1)).index(2.8)], 
                       [list(np.round(np.array(data['CII'][0]), 1)).index(3.2), 
                        list(np.round(np.array(data['CII'][0]), 1)).index(8.8)], 
                       [list(np.round(np.array(data['CII'][0]), 1)).index(9.2), 
                        list(np.round(np.array(data['CII'][0]), 1)).index(14.8)]]
        steps_timeO = [[list(np.round(np.array(data['OIII'][0]), 1)).index(0.2), 
                        list(np.round(np.array(data['OIII'][0]), 1)).index(2.8)], 
                       [list(np.round(np.array(data['OIII'][0]), 1)).index(3.2), 
                        list(np.round(np.array(data['OIII'][0]), 1)).index(8.8)], 
                       [list(np.round(np.array(data['OIII'][0]), 1)).index(9.2), 
                        list(np.round(np.array(data['OIII'][0]), 1)).index(14.8)]]
        
        for step_C, step_O in zip(steps_timeC, steps_timeO):
            time_C.append([data['CII'][0][step_C[0]], data['CII'][0][step_C[1]]])
            time_O.append([data['OIII'][0][step_O[0]], data['OIII'][0][step_O[1]]])
            data_C.append([np.mean(np.array(data['CII'][1][step_C[0]:step_C[1]])), np.mean(np.array(data['CII'][1][step_C[0]:step_C[1]]))])
            data_O.append([np.mean(np.array(data['OIII'][1][step_O[0]:step_O[1]])), np.mean(np.array(data['OIII'][1][step_O[0]:step_O[1]]))])

        #flatten lists
        time_C = list(itertools.chain.from_iterable(time_C))
        data_C = list(itertools.chain.from_iterable(data_C))
        time_O = list(itertools.chain.from_iterable(time_O))
        data_O = list(itertools.chain.from_iterable(data_O))

        ax[0][i].plot(time_C, data_C, label=shotnumber)
        ax[1][i].plot(time_O, data_O, label=shotnumber)

    ax[0][0].set_xlabel('time t in (s)')
    ax[0][1].set_xlabel('time t in (s)')
    ax[1][0].set_xlabel('time t in (s)')
    ax[1][1].set_xlabel('time t in (s)')
    ax[0][0].set_ylabel('averaged intensity of C-II spectral line (133.510nm) OP2.2')
    ax[0][1].set_ylabel('averaged intensity of C-II spectral line (133.510nm) OP2.3')
    ax[1][0].set_ylabel('averaged intensity of O-III spectral line (83.430nm) OP2.2')
    ax[1][1].set_ylabel('averaged intensity of O-III spectral line (83.430nm) OP2.3')
    ax[0][0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1][0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('results/impurities/HexosTrends_CII_OIII_throughReferenceDischargesAverages.png', bbox_inches='tight')
    fig.show()
    plt.close()

#######################################################################################################################################################################
def calculateFluxErodedParticlesGross(flux_electron: int|float, Y: list, f: list, P_redeposition: int|float, stickingCoefficient: int|float, Y_selfSputtering: int|float) -> float:                                                          
    ''' This function calculates the gross flux density of eroded particles (redeposition not considered)
        Provide flux density of electrons "flux_electron" in [1/(s*m^2)]
        Sputtering yield "Y" and ion concentration "f" must be np.arrays for containing the values of all ion species at moment x
        -> exept the element from which the target is made, its sputtering yield is given by "Y_selfSputtering"
        -> "Y" must contain all relevant erosion processes (physical and chemical sputtering, ...)
        The propability of redeposition "P_redeposition" and the sticking coefficient "stickingCoefficient" of the target material must be known
        Returns single value for eroded particle flux density (gross) at moment x'''
    return (flux_electron * np.sum(Y * f))/(1 - P_redeposition * (Y_selfSputtering + 1 - stickingCoefficient))

def calculateFluxRedepositedParticles(flux_electron: int|float, Y: list, f: list, P_redeposition: int|float, stickingCoefficient: int|float, Y_selfSputtering: int|float) -> float:                                                          
    ''' This function calculates the flux density of redeposited particles 
        Provide flux density of electrons "flux_electron" in [1/(s*m^2)]
        Sputtering yield "Y" and ion concentration "f" must be np.arrays for containing the values of all ion species at moment x
        -> exept the element from which the target is made, its sputtering yield is given by "Y_selfSputtering"
        -> "Y" must contain all relevant erosion processes (physical and chemical sputtering, ...)
        The propability of redeposition "P_redeposition" and the sticking coefficient "stickingCoefficient" of the target material must be known
        Returns single value for redeposited particle flux density at moment x'''
    return (stickingCoefficient * P_redeposition * flux_electron * np.sum(Y * f))/(1 - P_redeposition * (Y_selfSputtering + 1 - stickingCoefficient))

def calculateFluxErodedParticlesNet(flux_electron: int|float, Y: list, f: list, P_redeposition: int|float, stickingCoefficient: int|float, Y_selfSputtering: int|float) -> float:                                                          
    ''' This function calculates the net flux density of eroded particles (redeposition considered)
        Provide flux density of electrons "flux_electron" in [1/(s*m^2)]
        Sputtering yield "Y" and ion concentration "f" must be np.arrays for containing the values of all ion species at moment x
        -> exept the element from which the target is made, its sputtering yield is given by "Y_selfSputtering"
        -> "Y" must contain all relevant erosion processes (physical and chemical sputtering, ...)
        The propability of redeposition "P_redeposition" and the sticking coefficient "stickingCoefficient" of the target material must be known
        Returns single value for eroded particle flux density (net) at moment x'''
    return calculateFluxErodedParticlesGross(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering) - calculateFluxRedepositedParticles(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering)

#######################################################################################################################################################################
#Calculation of Net Erosion Specifically for Divertor Plates according to Ref. 1, NOT READY TO USE YET

#Choose applying lambda_n in dependance of y, z, theta
def chooseLambdaN(y, z, theta):
    if y * np.sin(theta) + z * np.cos(theta)<0:
        return lambda_nl
    if y * np.sin(theta) + z * np.cos(theta)>0:
        return lambda_nr
    else:   #dont know what to return
        return 1

#Development of electron density and electron temperature
def developementElectronDensity(n_e_0, y, z, theta, lambda_n):
    lambda_n = chooseLambdaN(y, z, theta)
    return n_e_0 * np.exp(-(y * np.sin(theta) + z * np.cos(theta))/lambda_n)

#Development of electron density and electron temperature
def developementElectronTemperature(T_e_0, y, z, theta, lambda_n):
    lambda_n = chooseLambdaN(y, z, theta)
    return T_e_0 * np.exp(-(y * np.sin(theta) + z * np.cos(theta))/lambda_n)

#Calculation of gross erosion particle flux
def calculateFluxErodedParticlesGrossDivertor():
    pass

#Calculation of deposition particle flux
def calculateFluxRedepositedParticlesDivertor(flux_erodedParticles, y, dy):
    return -flux_erodedParticles(y - dy)

#Calculation of net erosion particle flux
def calculateFluxErodedParticlesNetDivertor(flux_erodedParticles, y, dy):
    return flux_erodedParticles(y) - flux_erodedParticles(y - dy)

#Calculation of an estimate for the layer thickness of eroded material at the strike line region
#Plasma ions exclude carbon impurities!
def calculateDeltaErodedLayerStrikeline(Y_plasmaIons, flux_plasmaIons, t_discharge, n_target, Y_selfsputtering, probability_promptRedeposition, stickingCoefficient, dy, C=3.5): 
    return np.tanh(C * dy/abs(lambda_nl)) * (t_discharge * flux_plasmaIons * Y_plasmaIons * (1 - stickingCoefficient * probability_promptRedeposition))/(n_target * (1 - probability_promptRedeposition * (Y_selfsputtering + 1 - stickingCoefficient)))

#######################################################################################################################################################################
def processOP2Dataold(discharge: str, 
                      ne_lower: list[list[int|float]],
                      ne_upper: list[list[int|float]], 
                      Te_lower: list[list[int|float]], 
                      Te_upper: list[list[int|float]], 
                      Ts_lower: list[list[int|float]], 
                      Ts_upper: list[list[int|float]], 
                      t_lower: list[list[int|float]], 
                      t_upper: list[list[int|float]], 
                      alpha: int|float, 
                      LP_position: list[int|float], 
                      LP_zeta: list[int|float],
                      m_i: list[int|float], 
                      f_i: list[int|float], 
                      ions: list[str], 
                      k: int|float, 
                      n_target: int|float, 
                      plotting: bool =False) -> None:
    ''' This function calculates sputtering related physical quantities (sputtering yields, erosion rates, layer thicknesses) for various time steps of a discharge "discharge" at different positions
        "*_upper" and "*_lower" determine which divertor unit is considered
        Electron density "ne_*" in [1/m^3], electron temperature "Te_*" in [eV] and assumption that "Te_*" = ion temperature "Ti_*", surface temperature of the target "Ts_*" in [K], time "t_*" in [s]
        -> all arrays "ne_*", "Te_*", "Ts_*" are of ndim=2 with each line representing measurements over time at one Langmuir probe positions (=each column representing measurements at all positions at one time)
        -> times are given in [s] from trigger t1 of the discharge
        Incident angle of the ions  "alpha" is given in [rad]
        The distance of each langmuir probe from the pumping gap is given in "LP_position" in [m], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad], probe indices 0 - 5 are on TM2h07, 6 - 13 on TM3h01, and 14 - 17 on TM8h01
        The ion masses "m_i" are in [kg], the ion concentrations "f_i", and ion names "ions" should have the same length as "m_i"
        The Boltzmann constant "k" must be in [eV/K]
        The atomic target density "n_target" should be provided in [1/m^3]
        The parameter "plotting" determines, if measurement data, sputtering yields and erosion rates/layer thicknesses are plotted
        -> if True, plots are saved in safe = 'results/plots/overview_{exp}-{discharge}_{divertorUnit}{position}.png'.format(exp=discharge[:-4], discharge=discharge[-3:], divertorUnit=divertorUnit, position=position)
        
        This function does not return something but writes measurement values and calculated values to 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)'''
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
            return_erosion = process.calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, timesteps, alpha, LP_zeta[LP_position[position - 1]], m_i, f_i, ions, k, n_target)
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