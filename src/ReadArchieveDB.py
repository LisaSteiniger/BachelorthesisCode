'''This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
It also works with juice to find information about the discharges (percentage of EIM/FTM/KJM, ...).'''

import os
import w7xarchive   #see https://git.ipp-hgw.mpg.de/kjbrunne/w7xarchive/-/blob/master/doc/workshop.ipynb for introduction
import matplotlib.pyplot as plt
import numpy as np
from src.dlp_data.dlp_data import extract_divertor_probe_data as extract

def readLangmuirProbeDataFromXdrive(dischargeID):
    #if 8 files are not found, that is ok -> they are at TM8h and only exist for high iota discharges
    data_lower, data_upper = extract.fetch_xdrive_data(shot = dischargeID)
    if data_lower[0].units['time'] == 's' and data_lower[0].units['ne'] == '10$^{18}$m$^{-3}$' and data_lower[0].units['Te'] == 'eV':
        ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper = [], [], [], [], [], []
        for i in range(len(data_lower)):
            ne_lower.append(list(data_lower[i].ne))
            ne_upper.append(data_upper[i].ne)
            Te_lower.append(data_lower[i].Te)
            Te_upper.append(data_upper[i].Te)
            t_lower.append(data_lower[i].time)
            t_upper.append(data_upper[i].time)
        
        #careful, not all subarrays have the same shape, len(Te_upper[0]) == len(ne_upper[0]), but not neccessarily len(Te_upper[0]) == len(Te_upper[1])
        return ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper
    else:
        return 'wrong units'

def readArchieveDB():
    shotnumbersOP1 = ['20181018.041']
    shotnumbersOP2 = ['20250424.050']

    #for OP1.2
    #Langmuir Probes, #numbersOP1 in range [1, 20]
    divertorUnitsOP1 = ['lowerTestDivertorUnit', 'upperTestDivertorUnit']

    #IR cameras

    #for OP2
    #Langmuir Probes, numbersOP2 in range [1, 14]
    divertorUnitsOP2 = ['LowerDivertor', 'UpperDivertor']
    #IR cameras
    
    for shotnumber in shotnumbersOP1:
        #for OP1
        for divertorUnitOP1 in divertorUnitsOP1:
            for numberOP1 in range(1, 2):#21):
                #called like this time is in seconds from t1 (program start?)
                data_urls_OP1 = {'LP_ne':"ArchiveDB/raw/Minerva/Minerva.ElectronDensity.QRP.{divertorUnit}.{number}/ne_DATASTREAM/V1/0/ne".format(divertorUnit=divertorUnitOP1, number=numberOP1),
                                'LP_Te':"ArchiveDB/raw/Minerva/Minerva.ElectronTemperature.QRP.{divertorUnit}.{number}/Te_DATASTREAM/V1/0/Te".format(divertorUnit=divertorUnitOP1, number=numberOP1)}
                #add error of the datastreams

                #enables parallel download if data_urls_OP1 is dictionary
                data_OP1 = w7xarchive.get_signal_for_program(data_urls_OP1, shotnumber) 
                #data_OP1['LP_ne'][0] to access time trace of ne measurement, data_OP1['LP_ne'][1] to access ne data
                #data_OP1['LP_Te'][0] to access time trace of Te measurement, data_OP1['LP_Te'][1]to access Te data
                
                #just to test if reading the data works properly
                plt.figure()
                plt.plot(data_OP1['LP_ne'][0], data_OP1['LP_ne'][1])
                plt.plot(data_OP1['LP_Te'][0], data_OP1['LP_Te'][1])
                plt.gca().set_xlabel("time [s]")
                plt.gca().set_ylabel("data")
                plt.show()
                pass

    for shotnumber in shotnumbersOP2:
        #for OP2
        for divertorUnitOP2 in divertorUnitsOP2:
            for numberOP2 in range(1, 2):#14):
                #called like this time is in seconds from t1
                data_urls_OP2 = {'LP_ne':"ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnit}_Probe_{number}_DATASTREAM/V1/1/Plasma_Density".format(divertorUnit=divertorUnitOP2, number=numberOP2),
                                'LP_Te':"ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnit}_Probe_{number}_DATASTREAM/V1/2/Electron_Temperature".format(divertorUnit=divertorUnitOP2, number=numberOP2)}
                #add error of the datastreams

                data_OP2 = w7xarchive.get_signal_for_program(data_urls_OP2, shotnumber)
                #just to test if reading the data works properly
                
                plt.figure()
                plt.plot(data_OP2['LP_ne'][0], data_OP2['LP_ne'][1])
                plt.plot(data_OP2['LP_Te'][0], data_OP2['LP_Te'][1])
                plt.gca().set_xlabel("time [s]")
                plt.gca().set_ylabel("data")
                plt.show()
                pass

def readMarkusData(settings_dict, interval, settings_Gao):
    #get n_e data from Langmuir Probes
    ne_path = os.path.join('Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='ne', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time, positions, ne_values = dict(np.load(ne_path,allow_pickle=True)).values() #n_e_values is 2dimensional with lines representing positions and columns the times

    #get T_e data from Langmuir Probes
    Te_path = os.path.join('Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='Te', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time, positions, Te_values = dict(np.load(Te_path,allow_pickle=True)).values() #T_e_values is 2dimensional with lines representing positions and columns the times

    #get variances for n_e and T_e, same data structure as for n_e, T_e
    ne_path = os.path.join('Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='variancene', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time_var, positions_var, ne_values_var = dict(np.load(ne_path,allow_pickle=True)).values()

    Te_path = os.path.join('Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='varianceTe', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time_var, positions_var, Te_values_var = dict(np.load(Te_path,allow_pickle=True)).values()

    #convert n_e and its variances in 1/m^3, positions and their variances in m
    ne_values = ne_values * 1e18
    ne_values_var = ne_values_var * 1e18
    positions = positions * 1e-3
    positions_var = positions_var * 1e-3

    #get standard deviation for n_e, T_e and filter out values with stDev > values/2
    ne_values_var = np.sqrt(np.abs(ne_values_var))
    Te_values_var = np.sqrt(np.abs(Te_values_var))
    ne_values[np.where(ne_values_var * 2 > ne_values)] = np.nan
    Te_values[np.where(Te_values_var * 2 > Te_values)] = np.nan

    #filter out values that exceed limits
    ne_limit = 100 * 1e18
    Te_limit = 500
    ne_values[np.where(ne_values > ne_limit)] = np.nan
    Te_values[np.where(Te_values > Te_limit)] = np.nan

    #if interval = 50 values are averaged, intervalNumber averages are determined
    intervalNumber = time.shape[0]//interval

    #creates intervalNumber arrays with approx. the same number of entries and subsequent times -> e.g. for n_e of shape 5, 100 and intervalNumber=10, 10 arrays are created with shape 5, 10 where the 10 times are adjacent for one block
    splitarray_ne = np.array_split(ne_values, intervalNumber, axis=1) 
    splitarray_Te = np.array_split(Te_values, intervalNumber, axis=1)
    splitarray_ne_var = np.array_split(ne_values_var, intervalNumber, axis=1)
    splitarray_Te_var = np.array_split(Te_values_var, intervalNumber, axis=1)
    splitarray_time = np.array_split(time, intervalNumber, axis=0)

    #averages n_e in each interval
    ne_list = [] 
    for i in range(intervalNumber):
        Mittel = np.nanmean(splitarray_ne[i], axis=1)
        Mittel = np.transpose([Mittel])
        ne_list.append(Mittel)               #adds [[n_e_av1_pos1], [n_e_av1_pos2], ...]
    ne_values = np.array(np.hstack(ne_list)) #flattens n_e_list to 2D array with lines being positions and columns averaged n_e values at different times

    #averages T_e in each interval
    Te_list = []
    for i in range(intervalNumber):
        Mittel = np.nanmean(splitarray_Te[i], axis=1)
        Mittel = np.transpose([Mittel])
        Te_list.append(Mittel)
    Te_values = np.array(np.hstack(Te_list))

    #averages n_e stDev in each interval
    ne_var_list = []
    for i in range(intervalNumber):
        Mittel = np.nanmean(splitarray_ne_var[i], axis=1)
        Mittel = np.transpose([Mittel])
        ne_var_list.append(Mittel)
    ne_values_var = np.array(np.hstack(ne_var_list))

    #averages T_e stDev in each interval
    Te_var_list = []
    for i in range(intervalNumber):
        Mittel = np.nanmean(splitarray_Te_var[i], axis=1)
        Mittel = np.transpose([Mittel])
        Te_var_list.append(Mittel)
    Te_values_var = np.array(np.hstack(Te_var_list))  

    #average time in each interval
    time_list = []
    for i in range(intervalNumber):
        Mittel = np.nanmean(splitarray_time[i], axis=0)
        Mittel = np.transpose([Mittel])
        time_list.append(Mittel)
    time = np.array(np.hstack(time_list))

    #transpose n_e and T_e so that columns being positions and lines averaged values at different times
    ne_values=np.transpose(ne_values)
    Te_values=np.transpose(Te_values)
    ne_values_var=np.transpose(ne_values_var)
    Te_values_var=np.transpose(Te_values_var)

    #read Gao's data
    #lines represent one time each, columns are positions in data? S corresponds to position
    Gao_path_data = os.path.join('Daten von Gao/{exp}_{discharge}_{divertor}_{l}_{finger}/data.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger']))
    Gao_path_t = os.path.join('Daten von Gao/{exp}_{discharge}_{divertor}_{l}_{finger}/t.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger']))
    Gao_path_S = os.path.join('Daten von Gao/{exp}_{discharge}_{divertor}_{l}_{finger}/S.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger']))
    Gao_path_data_PF = os.path.join('Daten von Gao PF/{exp}_{discharge}_{divertor}_{l}_{finger}_{PF}/data_PF.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger'], PF='PF'))

    #read out data for temperatures in [°C]?
    data_Ts1= open(Gao_path_data, 'r').readlines()
    data_Ts = []
    for line in data_Ts1:
        data_Ts.append(line.split())
    data_Ts = np.array(data_Ts) 
    data_Ts = data_Ts.astype(float)

    #read out data for time
    t1 = open(Gao_path_t, 'r').readlines()
    t = []
    for line in t1:
        t.append(line.split())
    t = np.array(t)
    t = t.astype(float)

    #read out data for positions
    S1 = open(Gao_path_S, 'r').readlines()
    S = []
    for line in S1:
        S.append(line.split()) 
    S = np.array(S)
    S = S.astype(float)

    #read out data for power fluxes in [MW/m^2]
    data_PF1= open(Gao_path_data_PF, 'r').readlines()
    data_PF = []
    for line in data_PF1:
        data_PF.append(line.split())
    data_PF = np.array(data_PF)
    data_PF = data_PF.astype(float)

    #modify data of both data sets to get it ready for calculation
    tii = 6                                                       #F�r Gao Daten in sek                        Nur f�rs Plotten
    ti = int((tii/2) * 1004/interval)                              #F�r Lukas Rudischhauser Daten: 07.014 - time:[0, 8]       Nur f�rs Plotten

    #reverse order inside a line?
    ne_values_spiegel = ne_values[:,::-1]                                     
    Te_values_spiegel = Te_values[:,::-1]

    dim = ne_values.shape[1]

    #create/modify positions
    steps = []                  
    for i in range(dim - 1):
        steps.append(positions[i] - positions[i + 1])    
    steps = sum(steps)/(dim - 1)
    positions_spiegel = positions - dim * steps

    #append reversed values to original data
    positions = np.hstack((positions, positions_spiegel))                     
    ne_values = np.hstack((ne_values, ne_values_spiegel))
    Te_values = np.hstack((Te_values, Te_values_spiegel))

    #filter out negative positions
    index = np.where(positions < 0)
    cut_index = index[0][0]
    positions = np.delete(positions, np.s_[cut_index:], axis=0)              
    ne_values = np.delete(ne_values, np.s_[cut_index:], axis=1)
    Te_values = np.delete(Te_values, np.s_[cut_index:], axis=1)

    #reverse order to start with smallest position and go to largest
    positions=positions[::-1]                              
    ne_values=ne_values[:,::-1]
    Te_values=Te_values[:,::-1]

    #modify position data from both data sets to bring them in agreement
    Sround = np.round(S, 3)                                                   
    positionsround = np.round(positions, 3)                                  
    list_delete = []
    for i in range(S.shape[0]):
        a = Sround[i]
        b = Sround[i] + 0.001
        c = Sround[i] - 0.001
        if a in positionsround:
            pass
        elif b in positionsround:
            pass
        elif c in positionsround:
            pass
        else:
            list_delete.append(i)
    #deletes the data(S, Ts, PF(plasmaflux)) of yu, that was measured at positions to far from lukas positions -> different axis due to shapes of arrays
    S = np.delete(S, list_delete, axis=0)
    data_Ts = np.delete(data_Ts, list_delete, axis=1)
    data_PF = np.delete(data_PF, list_delete, axis=1)

    #further calculations use only ne_values, Te_values and positions from this data
    return ne_values, Te_values, positions, data_Ts, t, S