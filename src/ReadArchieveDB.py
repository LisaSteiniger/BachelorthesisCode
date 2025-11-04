''' This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
    As most data for OP2.2/2.3 is not yet uploaded to ArchiveDB, functions for reading from xdrive and other storage options are provided to guarantee access to a mayority of the data
    Nevertheless, some discharges remain unavailable and even existing data is not cross checked and processed further (e.g. no removal of surface layer effects on divertor temperature)'''

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
        durations = []
        discharges = []
        for discharge in dischargeIDlist:
            shot = juice[juice['shot']==discharge]
            print(shot.head()['configuration'])
            discharges.append(list(shot['shot'])[0])
            configurations.append(list(shot['configuration'])[0])
            durations.append(list(shot['t_shot_stop'])[0])

        dischargeIDcsv = pd.DataFrame({'dischargeID':dischargeIDlist, 'configuration':configurations, 'duration':durations})
        dischargeIDcsv.to_csv(safe, sep=';')
        return dischargeIDcsv

#######################################################################################################################################################################
def getRuntimePerConfiguration(dischargeIDcsv, safe='results/configurationRuntimes.csv'):
    ''' This function determines the absolute and relative runtime of each configuration over all discharges given by "dischargeIDcsv" and writes them as .csv file to safe'''
    configurations = list(np.unique(dischargeIDcsv['configuration']))
    configurations = list(np.unique([x[:4] for x in configurations]))
    #print(configurations)

    runtimes = pd.DataFrame({})
    absoluteRuntimes, relativeRuntimes = [], []
    totalRuntime = sum(dischargeIDcsv['duration'])

    for configuration in configurations:
        filter = np.array([i.startswith(configuration) for i in dischargeIDcsv['configuration']])
        runtime = sum(dischargeIDcsv[filter]['duration'])
        absoluteRuntimes.append(runtime)
        relativeRuntimes.append(runtime/totalRuntime * 100)

    configurations.append('all')
    absoluteRuntimes.append(totalRuntime)
    relativeRuntimes.append(100)

    runtimes['configuration'] = configurations
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

        for time in times:
            ########################new2########################
            port = 'AEF50'
            archiveComp = ['Test', 'raw', 'W7XAnalysis', 'QRT_IRCAM_new']

            #test if any data is available for the discharge at any time
            time_from, time_to = w7xarchive.get_program_from_to(dischargeID)            
            signal_name = np.append(archiveComp, port + '_meanHF_TMs_DATASTREAM')
            normed_signal = w7xarchive.get_base_address(signal_name)
            url = w7xarchive.get_base_address(normed_signal)
            versioned_signal, url_tail = w7xarchive.get_stream_address(url)
            highest_version = w7xarchive.get_last_version(versioned_signal, time_from, time_to)
            if highest_version is None:
                print('No IRcam stream for any time in discharge {dischargeID}')
                return 'No IRcam stream for any time in discharge {dischargeID}'
            #########################new2########################

            test = heatflux_T_download.heatflux_T_process(dischargeID, camera)
            pulse_duration = test.availtimes[-1] - 1
            ########################new########################
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
                return 'No IRcam stream available for discharge {dischargeID} in interval'
            ########################new########################

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
def readArchieveDB():
    '''just example for using J. Brunners package w7xarchive to read data from ArchiveDB'''
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
                