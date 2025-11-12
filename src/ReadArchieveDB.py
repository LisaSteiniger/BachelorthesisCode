''' This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
    As most data for OP2.2/2.3 is not yet uploaded to ArchiveDB, functions for reading from xdrive and other storage options are provided to guarantee access to a mayority of the data
    Nevertheless, some discharges remain unavailable and even existing data is not cross checked and processed further (e.g. no removal of surface layer effects on divertor temperature)'''

import requests
import itertools
import os
import w7xarchive   #see https://git.ipp-hgw.mpg.de/kjbrunne/w7xarchive/-/blob/master/doc/workshop.ipynb for introduction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.dlp_data import extract_divertor_probe_data as extract
from src.heatflux_T import heatflux_T_download

#######################################################################################################################################################################
def readAllShotNumbersFromJuice(juicePath, safe='results/dischargeIDlistAll.csv'):   
    ''' if no file is saved under "safe", function reads the IDs of all discharges carried out in campaign(s) for which juice file(s) are given under "juicePath" and saves them as .csv file under "safe"
        if such a file is existent, the discharge IDs are read from it and returned
        returned object is identical in both cases: dictionary object with 'dischargeID' key holding list of strings reprensenting all discharge IDs'''
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
def getRuntimePerConfiguration(dischargeIDcsv, safe='results/configurationRuntimes.csv'):
    ''' This function determines the absolute and relative runtime of each configuration over all discharges given by "dischargeIDcsv" and writes them as .csv file to safe'''
    configurations = list(np.unique(dischargeIDcsv['configuration']))
    configurations = list(np.unique([x[:3] for x in configurations]))
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
def readSurfaceTemperatureFramesFromIRcam(dischargeID, times, divertorUnits, LP_position, index_divertorUnit):
    ''' returns surface temperature of divertor in "discharge" to given "times" at given "divertorUnits" at position closest to langmuir probes
        returns arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        provide times in [s], divertorunits either "upper" or "lower" LP_position as array with distances of all probes from pumping gap in [m], and index_divertorUnit a list of lists with the indices of active langmuir probes in a divertor unit
        LPs are numbered as follows: TM2h07 holds probe 0 to 5, TM3h01 6 to 13, TM8h01 14 to 17 -> LP_positions should take numbers as indices, index_divertorUnits applies same scheme
        for reading temperature, TM2h06 and TM3h02 are chosen as 07 and 01 are subject to leading edges'''
    T_data = []
    '''
    if LP_number < 5:
        targetElements = ['TM8h_01']
        LP_position_indices = [[14, 15, 16, 17]]
    elif LP_number < 15:
        targetElements = ['TM2h_06', 'TM3h_02']
        LP_position_indices = [range(4), range(4, 14)]
    else:
        targetElements = ['TM2h_06', 'TM3h_02', 'TM8h_01']
        LP_position_indices = [range(4), range(4, 14), range(14, 18)]
    '''
    for divertorUnit, LP_indices in zip(divertorUnits, index_divertorUnit):
        T_data_divertor = []
        if divertorUnit == 'upper':
            camera = 'AEF51'
        elif divertorUnit == 'lower':
            camera = 'AEF50'
        else: 
            print('Undefined divertor Unit')
            exit

        port = camera
        archiveComp = ['Test', 'raw', 'W7XAnalysis', 'QRT_IRCAM_new']

        #test if any data is available for the discharge at any time
        time_from, time_to = w7xarchive.get_program_from_to(dischargeID)            
        signal_name = np.append(archiveComp, port + '_meanHF_TMs_DATASTREAM')
        normed_signal = w7xarchive.get_base_address(signal_name)
        url = w7xarchive.get_base_address(normed_signal)
        versioned_signal, url_tail = w7xarchive.get_stream_address(url)
        highest_version = w7xarchive.get_last_version(versioned_signal, time_from, time_to)
        if highest_version is None:
            print('No IRcam stream for any time in discharge')
            return 'No IRcam stream for any time in discharge'

        #test for number of triggers, discard discharges with less/more than one trigger
        prog = w7xarchive.get_program_info(dischargeID)
        if (prog["trigger"] is None) or len(prog["trigger"]['1']) < 1:
            print(f'The program {dischargeID} has no trigger {"1"}')
            return f'The program {dischargeID} has no trigger {"1"}'
        elif len(prog["trigger"]['1']) > 1:
            print(f'The program {dischargeID} has more than one trigger {"1"}')
            return f'The program {dischargeID} has more than one trigger {"1"}'
        
        for time in times:
            test = heatflux_T_download.heatflux_T_process(dischargeID, camera)
            pulse_duration = test.availtimes[-1] - 1
            #test if any data is available for the discharge time interval
            port = 'AEF50'
            archiveComp = ['Test', 'raw', 'W7XAnalysis', 'QRT_IRCAM_new']
            stream_channel_name = port + '_temperature_tar_baf_DATASTREAM'
            t1 = w7xarchive.get_program_t1(dischargeID)
            tsstamp = t1 + int(time * 1e9)
            testamp = t1 + int((time + 0.01) * 1e9)
            signal_name, time_from, time_to = np.hstack([archiveComp, stream_channel_name, '0', stream_channel_name]), max(0, tsstamp), min(test.availtimes[-1], testamp)

            # since we cannot be 100% certain that this function is always
            # called preceeded by get_base_address,
            # we call it again in here (shouldn't have a significant performance hit).
            url = w7xarchive.get_base_address(signal_name)

            # split stream and rest in the url
            versioned_signal, url_tail = w7xarchive.get_stream_address(url)
            #version_str = w7xarchive.helpers.get_version_from_url(url)

            # grab the latest available version
            highest_version = w7xarchive.get_last_version(versioned_signal, int(time_from), int(time_to))

            if highest_version is None:
                print('No IRcam stream available for discharge {dischargeID} in interval')
                T_data_dt = [0] * len(LP_indices) #no values are present
                #continue #jump to next time 
                #return 'No IRcam stream available for discharge {dischargeID} in interval'
            else:
                if time + 0.01 > pulse_duration: #if no frames are present for the time interval
                    T_data_dt = [0] * len(LP_indices)
                else:
                    test.get_Frames(time,  time + 0.1, T = True) #get frame for whole divertor unit in certain time interval
                    T_data_dt = []

                    for LP_index in LP_indices:
                        if LP_index < 6:
                            targetElement = 'TM2h_06'
                        elif LP_index < 14:
                            targetElement = 'TM3h_02'
                        else:
                            targetElement = 'TM8h_01'    

                        test.get_Profiles(targetElement, AverageNearby=1000) #get data for one target element
                        distance = test.stackS.tolist() #distance from pumping gap in [m] for all measured positions on the target element

                        closestIndex = distance.index(min(distance, key=lambda x: abs(x - LP_position[LP_index])))  #find measurement position of T that is closest to langmuir probe
                        #print(distance[closestIndex])
                        T_data_dt.append(test.datas.T[closestIndex][0] + 273.15)   #conversion from degrees C to K 
                
            T_data_divertor.append(T_data_dt)
        T_data.append(T_data_divertor)

    if len(T_data) == 2:    
        return [np.array(T_data[0]).T, np.array(T_data[1]).T]
    else:
        return np.array(T_data[0]).T
    
#######################################################################################################################################################################
def readLangmuirProbeDataFromXdrive(dischargeID):
    ''' returns electron temperature Te in [eV] and electron density ne in [1/m^3] plus corresponding mesurement times in [s] of langmuir probes read from xdrive as well as inormation about which LPs were active
        returns nested list of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        if 8 files are not found, that is ok -> they are at TM8h and only exist for high iota discharges
        if files are missing, index_upper and index_lower provide the indices for the active probes
        LPs are numbered as follows: TM2h07 holds probe 0 to 5, TM3h01 6 to 13, TM8h01 14 to 17 -> index_divertorUnits applies same scheme'''
    ########################new########################
    #'''
    xdrive_directory = "//x-drive/Diagnostic-logbooks/QRP-LangmuirProbes/QRP02-Divertor Langmuir Probes/Analysis/OP2/"
    directory = f"{xdrive_directory}/{dischargeID}/"
    #file = directory + f"{dischargeID}_probe_{probename}.txt"
    pathLP = directory
    print(pathLP)
    if not os.path.exists(pathLP):
        print('No LP data available')
        return 'No LP data available'
    #'''
    ########################new########################
    #internal naming of probes in xdrive
    probes_lower = [50201, 50203, 50205, 50207, 50209, 50211,  50218, 50220, 50222, 50224, 50226, 50228, 50230, 50232, 50246, 50248, 50249, 50251]
    probes_upper = [51201, 51203, 51205, 51207, 51209, 51211,  51218, 51220, 51222, 51224, 51226, 51228, 51230, 51232, 51246, 51248, 51249, 51251]
    
    #index lists for each divertor unit in cas that all LPs were active
    index_lower = list(range(18))
    index_upper = list(range(18))
    
    #data stores ne, Te, t, fails record missing LP data
    data_lower, data_upper, fails_lower, fails_upper = extract.fetch_xdrive_data(shot = dischargeID)
    
    #remove indices of inactive/missing LPs
    for fail in fails_lower:
        index_lower.remove(probes_lower.index(fail))
    for fail in fails_upper:
        index_upper.remove(probes_upper.index(fail))

    #test units as program needs those units to calculate correctly 
    if len(index_lower) != 0:
        test_index = index_lower
    elif len(index_upper) != 0:
        test_index = index_upper
    else:
        return 'No LP data available'
    
    if data_lower[test_index[0]].units['time'] == 's' and data_lower[test_index[0]].units['ne'] == '10$^{18}$m$^{-3}$' and data_lower[test_index[0]].units['Te'] == 'eV':
        ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper = [], [], [], [], [], []
        for i in index_lower:
            #ne_lower.append(list(data_lower[i].ne)) 
            
            #filter out measurements that are nonexisting (file exists but fake measurement value for t = 0 is inserted)
            filter_lower = np.array([j == 0 for j in data_lower[i].time])

            ne_lower.append(list(np.array(data_lower[i].ne)[~filter_lower]*1e+18))  #values are given as ne [1e+18 1/m^3]
            Te_lower.append(list(np.array(data_lower[i].Te)[~filter_lower]))
            t_lower.append(list(np.array(data_lower[i].time)[~filter_lower]))
        
        for i in index_upper:
            #ne_lower.append(list(data_lower[i].ne)) 
            
            #filter out measurements that are nonexisting (file exists but fake measurement value for t = 0 is inserted)
            filter_upper = np.array([j == 0 for j in data_upper[i].time])

            ne_upper.append(list(np.array(data_upper[i].ne)[~filter_upper]*1e+18))  #values are given as ne [1e+18 1/m^3]
            Te_upper.append(list(np.array(data_upper[i].Te)[~filter_upper]))
            t_upper.append(list(np.array(data_upper[i].time)[~filter_upper]))
        
        #careful, not all subarrays have the same shape, len(Te_upper[0]) == len(ne_upper[0]), but not neccessarily len(Te_upper[0]) == len(Te_upper[1])
        return [ne_lower, ne_upper, Te_lower, Te_upper, t_lower, t_upper, index_lower, index_upper]
    else:
        return 'wrong units'

#######################################################################################################################################################################
def readMarkusData(settings_dict, interval, settings_Gao):
    ''' returns electron temperature Te in [eV] and electron density ne in [1/m^3] plus corresponding mesurement times in [s] of langmuir probes
        returns surface temperature of divertor in [°C] at position closest to langmuir probes plus corresponding  relative to pumping gap in [m] and measurement time in [s]
        returns arrays of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        reads and processes data from downloaded files under path, interval determines how many time frames are averaged'''
    #get n_e data from Langmuir Probes
    ne_path = os.path.join('inputFiles/Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='ne', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time, positions, ne_values = dict(np.load(ne_path,allow_pickle=True)).values() #n_e_values is 2dimensional with lines representing positions and columns the times

    #get T_e data from Langmuir Probes
    Te_path = os.path.join('inputFiles/Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='Te', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time, positions, Te_values = dict(np.load(Te_path,allow_pickle=True)).values() #T_e_values is 2dimensional with lines representing positions and columns the times

    #get variances for n_e and T_e, same data structure as for n_e, T_e
    ne_path = os.path.join('inputFiles/Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='variancene', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time_var, positions_var, ne_values_var = dict(np.load(ne_path,allow_pickle=True)).values()

    Te_path = os.path.join('inputFiles/Daten_LP', settings_dict['exp'], '{c}_{q}_{tw}_{m}_{s}.npz'.format(c=settings_dict['c'], q='varianceTe', tw=settings_dict['tw'], m=settings_dict['m'], s=settings_dict['s']))
    time_var, positions_var, Te_values_var = dict(np.load(Te_path,allow_pickle=True)).values()

    #convert n_e and its variances in [1/m^3], positions and their variances in [m]
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

    #if interval = 50, 50 values are averaged, intervalNumber averages are determined
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

    #read Gao's data for surface temperature
    #lines represent one time each, columns are positions in data? S corresponds to position
    Gao_path_data = os.path.join('inputFiles/Daten von Gao/{exp}_{discharge}_{divertor}_{l}_{finger}/data.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger']))
    Gao_path_t = os.path.join('inputFiles/Daten von Gao/{exp}_{discharge}_{divertor}_{l}_{finger}/t.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger']))
    Gao_path_S = os.path.join('inputFiles/Daten von Gao/{exp}_{discharge}_{divertor}_{l}_{finger}/S.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger']))
    Gao_path_data_PF = os.path.join('inputFiles/Daten von Gao PF/{exp}_{discharge}_{divertor}_{l}_{finger}_{PF}/data_PF.txt'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertor=settings_Gao['divertor'], l='l', finger=settings_Gao['finger'], PF='PF'))

    #read out data for temperatures in [°C]
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
    #deletes the data(S, Ts, PF(plasmaflux)) of yu, that was measured at positions too far from lukas positions -> different axis due to shapes of arrays
    S = np.delete(S, list_delete, axis=0)
    data_Ts = np.delete(data_Ts, list_delete, axis=1)
    data_PF = np.delete(data_PF, list_delete, axis=1)

    #further calculations use only ne_values, Te_values and positions from this data
    return ne_values, Te_values, positions, data_Ts, t, S

#######################################################################################################################################################################
def readArchieveDB_ne_ECRH(dischargesCSV):
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
def readArchiveDB():
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
def readAllShotNumbersFromLogbook(safe ='results/configurations/dischargeList_OP223.csv'):
    url = 'https://w7x-logbook.ipp-hgw.mpg.de/api/_search'

    # q = 'tags.diagnostic\ result:"raw\ data"'
    # !! Helium Hydrogen discharges werden hier nicht unterschieden

    #filter options
    q1 = '!"Conditioning" AND ' #e.g. Pulse Trains
    q2 = '!"gas valve tests" AND '  #start of the day, no plasma present
    q3 = '!"sniffer tests" AND '    #at the beginning of the day, no plasma present, just ECRH
    q4 = '!"reference discharge" AND '
    q41 = '"ReferenceProgram"'   # in OP2.1 does not work
    q5 = '!"Open valves" AND '  #gas valves remain open after end of discharge
    q6 = 'id:XP_* AND tags.value:"ok" AND '
    q66 = 'id:XP_* AND '
    q71 = 'tags.value:"Reference"'  # for OP2.1
    q44 = '"reference discharge"'   # for OP2.2, OP2.3
    q45 = '"Reference discharge"'   # for OP1.2b

    id, duration, duration_DB, configuration, overviewTable = [], [], [], [], []    #id stores dischargeIDs, duration their ECRH heating period duration, configuration theit configuration

    for config in ['EIM000-2520', 'EIM000-2620', 'KJM008-2520', 'KJM008-2620', 'FTM000-2620', 'FTM004-2520', 'DBM000-2520', 'FMM002-2520',
                'EIM000+2520', 'EIM000+2620', 'EIM000+2614', 'DBM000+2520', 'KJM008+2520', 'KJM008+2620', 'XIM001+2485', 'MMG000+2520', 
                'DKJ000+2520', 'IKJ000+2520', 'FMM002+2520', 'KTM000+2520', 'FTM004+2520', 'FTM004+2585', 'FTM000+2620', 'AIM000+2520', 'KOF000+2520']:

        #filter for configurations (config)
        q7 = 'tags.value:"{config}"'.format(config=config)
        print(q7)
        #apply neccessary filters
        q = q1 + q2 + q3 + q66 + q7  

        # p = {'time':op_phase, 'size':'9999', 'q' : q }
        # p = {'time':'[2018 TO 2018]', 'size':'9999', 'q' : q }   # OP1.2b
        # p = {'time':'[2022 TO 2023]', 'size':'9999', 'q' : q }   # OP2.1
        p = {'time':'[2024 TO 2025]', 'size':'9999', 'q' : q }   # OP2.2 and OP2.3
        
        res = requests.get(url, params=p).json()

        if res['hits']['total']==0:
            print('no discharges found')
            continue

        for discharge in res['hits']['hits']:
            dischargeID = discharge['_id'][3:].split('.')
            if len(dischargeID[1]) == 3:
                dischargeID = dischargeID[0] + '.' + dischargeID[1]
            elif len(dischargeID[1]) == 2:
                dischargeID = dischargeID[0] + '.0' + dischargeID[1]
            else:
                dischargeID = dischargeID[0] + '.00' + dischargeID[1]
            #id.append(dischargeID)
            id.append(discharge['_id'])
            overviewTable.append('results/calculationTables/results_{discharge}.csv'.format(discharge=dischargeID))
     
            for tag in discharge['_source']['tags']: 
                if 'catalog_id' in tag.keys():
                    if tag['catalog_id'] == '1#3':
                        duration.append(tag['ECRH duration'])
                        continue
            
            if len(duration) != len(id):
                duration.append(np.nan)
            t = w7xarchive.get_program_from_to(discharge['_id'][3:]) #this returns the time from when first data streams become available (-61s before discharge start) till some time after the end of the discharge
            #t1 = w7xarchive.get_program_t0(discharge['_id'][3:])
            #t2 = w7xarchive.get_program_t1(discharge['_id'][3:])
            #t = w7xarchive.get_program_triggerts
            duration_DB.append((t[1] - t[0]) * 1e-9)

        configuration.append([config] * len(res['hits']['hits']))

    dischargeTable = pd.DataFrame({'configuration': list(itertools.chain.from_iterable(configuration)), 'dischargeID': id, 'duration': duration, 'durationDB': duration_DB, 'overviewTable': overviewTable})
    dischargeTable.to_csv(safe, sep=';')

    return dischargeTable


#SOME NOTES ABOUT WHAT IS IN DATA, CONFIGURATION, ...

# reversed field ========== missing for surface_layers
# see in Matlab file Aktualisierung für surface_layers_fixed
#config='"EIM000-2520"'   # 251 in OP2.2/2.3 (13 missing)
#config='"EIM000-2620"'    # 119 (5 missing)
#config='"KJM008-2520"'   # 79 in OP2.2/2.3 (1 missing)
#config='"KJM008-2620"'   # 96 in OP2.2/2.3 (14 missing)
#config='"FTM000-2620"'   # 40 in OP2.2/2.3, 3 missing
#config='"FTM004-2520"'   # no discharges in OP2.2/2.3
#config='"DBM000-2520"'   # 18 in OP2.2/2.3
#config='"FMM002-2520"'   # 15 in OP2.2/2.3

# normal field ==================================================
#config='"EIM000+2520"'   # 896 (252 missing), standard config 
#config='"EIM000+2620"'   # keine discharge
#config='"EIM000+2614"'   # 103 (5 missing)
#config='"DBM000+2520"'   # 86 (1 missing), low iota config
#config='"KJM008+2520"'  # 284 (169 missing), high mirror config
#config='"KJM008+2620"'   # 28 (11 missing)
#config='"XIM001+2485"'   # 57 (5 missing), negative mirror config
#config='"MMG000+2520"'   # 14 (0 missing), low shear config
#config='"DKJ000+2520"'   # 44 (2 missing), outward-shifted config
#config='"IKJ000+2520"'   # 53 (1 missing)
#config='"FMM002+2520"'   # 58 (7 missing), Reversed field config
#config='"KTM000+2520"'   # 4 (1 missing), high mirror config
#config='"FTM004+2520"'   #  149 (36 missing), high iota config
#config='"FTM004+2585"'    # 53 (4 missing)   
#config='"FTM000+2620"'   #  56 (14 missing)
#config='"AIM000+2520"'   # 55 (4 missing), low mirror 
#config='"KOF000+2520"'   # 19 (2 missing), low-shear configuration increased in iota
# config='"EJM+252"'        # OP1.2b
#config='"EJM007+2520"'    # OP2.1
#config='"EJM001+2520"'    # OP2.1
# ---------------------------------------------------------------

# q = 'id:XP_* AND tags.value:"ok" AND "TC" AND ' + config
# q = 'id:XP_* AND tags.value:"ok" AND "FZJ" AND ' + config
# q = 'id:XP_* AND tags.value:"ok" AND "Leakages in EIM" AND ' + config
# q = 'id:XP_* AND tags.value:"ok" AND "vrh_003" AND ' + config
# q = 'id:XP_* AND tags.value:"ok" AND "FZJ-MAT2" AND ' + config


#res.keys()                                             ['took', 'timed_out', 'hits']
#res['took']                                            no idea what this returns
#res['timed_out']                                       returns status of timed_out, either false or true

#res['hits'].keys()                                     ['total', 'max_score', 'hits', 'total_relation']
#res['hits']['total']                                   total number of discharges for the applied filter
#res['hits']['max_score']                               no idea what this is, not maximum run time of longest discharge
#res['hits']['total_relation']                          determines number of all hits (total) to considered hits?

#res['hits']['hits']                                    returns list with all discharges in the filtered version, res['hits']['hits'][0] is first discharge
#res['hits']['hits'][0].keys()                          ['_index', '_type', '_id', '_score', '_source']
#res['hits']['hits'][0]['_index']                       returns where this is read from (w7xlogbook)
#res['hits']['hits'][0]['_type']                        returns type of data (XP_logs)
#res['hits']['hits'][0]['_id']                          returns dischargeID 
#res['hits']['hits'][3]['_score']                       no idea what is returned here

#res['hits']['hits'][0]['_source'].keys()               ['id', 'name', 'description', 'session_comment', 'from', 'upto', 'execution_status', 'PP', 'tags', 'component_status', 'version', 'scenarios', 'from_str', 'upto_str', 'comments'] 
#res['hits']['hits'][0]['_source']['upto_str']          'from_str', 'upto_str' are timestamps in ns from start of program till end of program, but this is not the ECRH heating time
#res['hits']['hits'][0]['_source']['from_str']
#res['hits']['hits'][0]['_source']['from']              as lines above but as integer not string
#res['hits']['hits'][0]['_source']['upto']
#res['hits']['hits'][0]['_source']['id']                returns dischargeID
#res['hits']['hits'][0]['_source']['name']              uninteresting
#res['hits']['hits'][0]['_source']['execution_status']  no idea
#res['hits']['hits'][0]['_source']['PP']                no idea
#res['hits']['hits'][0]['_source']['tags']              returns list of dictionaries, see some examples below


#ECRH duration{'catalog_id': '1#3', 'unit': 's', 'component': 'CBG ECRH', 'valueNumeric': 15.001, 'name': 'ECRH duration', 'description': 'Total ECRH duration', 'ECRH duration': 15.001, 'category': 'Heating', 'value': '15.001'}, 
#more ECRH   {'catalog_id': '1#0', 'unit': 'kW', 'component': 'CBG ECRH', 'valueNumeric': 4500.0, 'ECRH': 4500.0, 'name': 'ECRH', 'description': 'Max. power all ECRH gyrotrons', 'category': 'Heating', 'value': '4500.0'}, 
#            {'catalog_id': '1#1', 'unit': 'MJ', 'component': 'CBG ECRH', 'valueNumeric': 38.63324999999999, 'name': 'ECRH energy', 'description': 'Total ECRH energy', 'category': 'Heating', 'value': '38.63324999999999', 'ECRH energy': 38.63324999999999}, 
#            {'catalog_id': '1#3', 'unit': 's', 'component': 'CBG ECRH', 'valueNumeric': 15.001, 'name': 'ECRH duration', 'description': 'Total ECRH duration', 'ECRH duration': 15.001, 'category': 'Heating', 'value': '15.001'}, 
#            {'catalog_id': '1#2', 'unit': None, 'component': 'CBG ECRH', 'ECRH pol': 'X2-mode', 'name': 'ECRH pol', 'description': 'ECRH polarisation mode', 'category': 'Heating', 'value': 'X2-mode'}, 
#            {'catalog_id': '1#4', 'A1': None, 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'A1', 'description': 'Gyrotron A1 active / max. power', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#49', 'unit': 'kW', 'component': 'CBG ECRH', 'A5': None, 'name': 'A5', 'description': 'Gyrotron A5 active / max. power', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#13', 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'B1', 'description': 'Gyrotron B1 active / max. power', 'category': 'Heating', 'value': None, 'B1': None}, 
#            {'catalog_id': '1#58', 'unit': 'kW', 'component': 'CBG ECRH', 'B5': None, 'name': 'B5', 'description': 'Gyrotron B5 active / max. power', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#67', 'unit': 'kW', 'component': 'CBG ECRH', 'C5': None, 'name': 'C5', 'description': 'Gyrotron C5 active / max. power', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#31', 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'D1', 'description': 'Gyrotron D1 active / max. power', 'category': 'Heating', 'value': None, 'D1': None}, 
#            {'catalog_id': '1#76', 'D5': None, 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'D5', 'description': 'Gyrotron D5 active / max. power', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#40', 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'E1', 'description': 'Gyrotron E1 active / max. power', 'category': 'Heating', 'E1': None, 'value': None}, 
#            {'catalog_id': '1#85', 'E5': None, 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'E5', 'description': 'Gyrotron E5 active / max. power', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#104', 'unit': 'kW', 'component': 'CBG ECRH', 'name': 'F5', 'description': 'Gyrotron F5 active / max. power', 'category': 'Heating', 'value': None, 'F5': None}, 
#            {'catalog_id': '1#73', 'unit': None, 'component': 'CBG ECRH', 'name': 'C5 oblique', 'C5 oblique': None, 'description': 'Gyrotron C5 oblique', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#37', 'unit': None, 'component': 'CBG ECRH', 'name': 'D1 oblique', 'description': 'Gyrotron D1 oblique', 'category': 'Heating', 'D1 oblique': None, 'value': None}, 
#            {'catalog_id': '1#82', 'unit': None, 'component': 'CBG ECRH', 'D5 oblique': None, 'name': 'D5 oblique', 'description': 'Gyrotron D5 oblique', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#46', 'E1 oblique': None, 'unit': None, 'component': 'CBG ECRH', 'name': 'E1 oblique', 'description': 'Gyrotron E1 oblique', 'category': 'Heating', 'value': None}, 
#            {'catalog_id': '1#91', 'unit': None, 'component': 'CBG ECRH', 'name': 'E5 oblique', 'description': 'Gyrotron E5 oblique', 'category': 'Heating', 'value': None, 'E5 oblique': None}, 

#NBI energy{'catalog_id': '1#95', 'unit': 'MJ', 'component': 'CDX21 ', 'valueNumeric': 1.1250000000001277, 'name': 'NBI21 energy', 'description': 'Total NBI21 energy', 'NBI21 energy': 1.1250000000001277, 'category': 'Heating', 'value': '1.1250000000001277'}, 
#NBI duration{'catalog_id': '1#96', 'unit': 's', 'component': 'CDX21 ', 'valueNumeric': 5.0000009999999975, 'name': 'NBI21 duration', 'description': 'Total NBI21 duration', 'category': 'Heating', 'value': '5.0000009999999975', 'NBI21 duration': 5.0000009999999975}, 

#result (ok, poor, failed){'catalog_id': '0#7', 'unit': None, 'component': None, 'name': 'Result', 'description': 'Program result', 'category': 'Experiment', 'value': 'ok', 'Result': 'ok'}, 

