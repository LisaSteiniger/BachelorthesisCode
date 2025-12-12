''' This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
    As most data for OP2.2/2.3 is not yet uploaded to ArchiveDB, functions for reading from xdrive and other storage options are provided to guarantee access to a mayority of the data
    Nevertheless, some discharges remain unavailable and even existing data is not cross checked and processed further (e.g. no removal of surface layer effects on divertor temperature)'''
#import certifi
import requests
import itertools
import os
import w7xarchive   #see https://git.ipp-hgw.mpg.de/kjbrunne/w7xarchive/-/blob/master/doc/workshop.ipynb for introduction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.dlp_data import extract_divertor_probe_data as extract
from src.heatflux_T import heatflux_T_download

#####################################################################################################################################################################
def getRuntimePerConfiguration(configurations: list[str] =['EIM000-2520', 'EIM000-2620', 'KJM008-2520', 'KJM008-2620', 'FTM000-2620', 'FTM004-2520', 'DBM000-2520', 'FMM002-2520',
                                                           'EIM000+2520', 'EIM000+2620', 'EIM000+2614', 'DBM000+2520', 'KJM008+2520', 'KJM008+2620', 'XIM001+2485', 'MMG000+2520', 
                                                           'DKJ000+2520', 'IKJ000+2520', 'FMM002+2520', 'KTM000+2520', 'FTM004+2520', 'FTM004+2585', 'FTM000+2620', 'AIM000+2520', 'KOF000+2520'],
                               dischargeIDcsv: str ='results/configurations/dischargeList_OP223_', 
                               safe: str ='results/configurations/configurationRuntimes.csv') -> None:
    ''' This function determines the absolute and relative runtime of each configuration over all discharges given by .csv files saved under "dischargeIDcsv" + "*", the results are written to .csv file saved under "safe"
        Each configuration has its own file, so function iterates over those files and sums up the number of discharges and their durations
        The DataFrame from each file contains all discharges of this configuration and has the keys 'dischargeID', 'duration', 'configuration', and 'overviewTables' (default safe for calculation tables of a discharge)'''

    runtimes = pd.DataFrame({})
    absoluteRuntimes, relativeRuntimes, absoluteNumberOfDischarges, relativeNumberOfDischarges = [], [], [], []
    totalRuntime = 0
    totalNumber = 0

    for configuration in configurations:
        if not os.path.isfile(dischargeIDcsv + configuration + '.csv'): #if there is no file, no discharge were found
            absoluteRuntimes.append(0)
            absoluteNumberOfDischarges.append(0)
        else:
            dischargeList = pd.read_csv(dischargeIDcsv + configuration + '.csv', sep=';')
            absoluteRuntimes.append(np.nansum(dischargeList['duration']))
            absoluteNumberOfDischarges.append(len(dischargeList['dischargeID']))
        totalRuntime += absoluteRuntimes[-1]
        totalNumber += absoluteNumberOfDischarges[-1]

    for runtime, number in zip(absoluteRuntimes, absoluteNumberOfDischarges):
        relativeRuntimes.append(runtime/totalRuntime * 100)
        relativeNumberOfDischarges.append((number/totalNumber) * 100)

    configurations = list(configurations)
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
    configurations.remove('all')

#######################################################################################################################################################################
def readSurfaceTemperatureFramesFromIRcam(dischargeID: str, 
                                          times: list[int|float], 
                                          divertorUnits: list[str], 
                                          LP_position: list[int|float], 
                                          index_divertorUnit: list[list[int]], 
                                          defaultTemperature: int|float =0) -> list:
    ''' This function reads the surface temperature of the divertor targets approximately at the positions of the Langmuir Probes (for reading temperature, TM2h06 and TM3h02 are chosen as 07 and 01 are subject to leading edges)
        The times at which the temperature is read out is given by the ID of the discharge ("dischargeID") and "times"
        "times" is an array like object containing timesteps in [s] from trigger t1 of that discharge to the measurement times of the Langmuir Probes
        "divertorUnits" is an array like object containing either 'lower', 'upper', or both -> determines the divertor unit of module 5 to look at, should be of same length as "index_divertorUnits"
        "index_divertorUnits" is an array like object containing array like objects itself (same length as "divertorUnits") -> the elemnts themselves hold the numbers of all active Langmuir Probes in that divertor unit 
        -> LPs are numbered as follows: TM2h07 holds probe 0 to 5, TM3h01 6 to 13, TM8h01 14 to 17 -> LP_positions should take numbers as indices, index_divertorUnits applies same scheme
        "LP_position" is an array like object holding the distances of all probes from pumping gap in [m]
        "defaultTemperature" is the temperature in [K] that is inserted when data is missing -> 0K allows latter filtering these missing values 

        Returns list of lists with array like objects of ndim=2 representing the surface temperatures of divertor in "dischargeID" to given "times" at given "divertorUnits" at position closest to langmuir probes 
        Each line represents measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        last element of the list is  a comment on the availability of the data -> if it is 0, data was accessible, if it is a str, data was missing
        '''
    T_data = []
    comment = 0
    
    #look at each divertor unit separately
    for divertorUnit, LP_indices in zip(divertorUnits, index_divertorUnit):
        T_data_divertor = []

        #depending on the divertor unit, either the camera in port AEF50 or AEF51 provides data
        if divertorUnit == 'upper':
            camera = 'AEF51'
        elif divertorUnit == 'lower':
            camera = 'AEF50'
        else: 
            print('Undefined divertor Unit')
            exit

        archiveComp = ['Test', 'raw', 'W7XAnalysis', 'QRT_IRCAM_new']
        
        #test for number of triggers, discard discharges with less/more than one trigger for both cameras
        prog = w7xarchive.get_program_info(dischargeID)
        if (prog["trigger"] is None) or len(prog["trigger"]['1']) < 1:
            print(f'The program {dischargeID} has no trigger {"1"}')
            T_data = [[[defaultTemperature] * len(times)] * len(LP_indices)] * 2
            return [T_data[0], T_data[1], 'incorrected trigger'] 
        elif len(prog["trigger"]['1']) > 1:
            print(f'The program {dischargeID} has more than one trigger {"1"}')
            T_data = [[[defaultTemperature] * len(times)] * len(LP_indices)] * 2
            return [T_data[0], T_data[1], 'incorrected trigger'] 
            

        #test if any data is available for the discharge at any time for the camera
        time_from, time_to = w7xarchive.get_program_from_to(dischargeID)            
        signal_name = np.append(archiveComp, camera + '_meanHF_TMs_DATASTREAM')
        normed_signal = w7xarchive.get_base_address(signal_name)
        url = w7xarchive.get_base_address(normed_signal)
        versioned_signal, url_tail = w7xarchive.get_stream_address(url)
        highest_version = w7xarchive.get_last_version(versioned_signal, time_from, time_to)
        if highest_version is None:
            print('No IRcam stream for any time in discharge at ' + divertorUnit + ' divertor unit')
            T_data.append([[defaultTemperature] * len(LP_indices)] * len(times))
            comment = 'no IR data'
            continue #with the other divertor unit if existent
        
        #if one t1 trigger is present and data is available for any time in the discharge
        for time in times:
            test = heatflux_T_download.heatflux_T_process(dischargeID, camera)
            pulse_duration = test.availtimes[-1] - 1
            
            #test if any data is available for the discharge in the selected time interval for this divertor unit 
            #camera = 'AEF50'
            archiveComp = ['Test', 'raw', 'W7XAnalysis', 'QRT_IRCAM_new']
            stream_channel_name = camera + '_temperature_tar_baf_DATASTREAM'

            #all times in [ns]
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

            if highest_version is None: #if no frames are present for the time interval
                print('No IRcam stream available for discharge {dischargeID} in interval')
                T_data_dt = [defaultTemperature] * len(LP_indices) #no values are present
            else:
                if time + 0.01 > pulse_duration: #if no frames are present for the time interval because its running out of the discharge duration
                    T_data_dt = [defaultTemperature] * len(LP_indices)
                else:
                    test.get_Frames(time,  time + 0.1, T = True) #get frame for whole divertor unit in certain time interval
                    T_data_dt = []

                    #now find the positions of the Langmuir Probes 
                    for LP_index in LP_indices:
                        if LP_index < 6:
                            targetElement = 'TM2h_06'
                        elif LP_index < 14:
                            targetElement = 'TM3h_02'
                        else:
                            targetElement = 'TM8h_01'    

                        test.get_Profiles(targetElement, AverageNearby=1000) #get data for one target element, average of temperature over the whole TE and 0.1 s interval
                        distance = test.stackS.tolist() #distance from pumping gap in [m] for all measured positions on the target element

                        closestIndex = distance.index(min(distance, key=lambda x: abs(x - LP_position[LP_index])))  #find measurement position of T that is closest to langmuir probe
                        T_data_dt.append(test.datas.T[closestIndex][0] + 273.15)   #conversion from degrees C to K 
                
            T_data_divertor.append(T_data_dt)
        T_data.append(T_data_divertor)

    #returns list of length 2 or 3 depending on the number of divertor units that were investigated
    if len(T_data) == 2:    
        return [np.array(T_data[0]).T, np.array(T_data[1]).T, comment]
    else:
        return [np.array(T_data[0]).T, comment]
    
#######################################################################################################################################################################
def readLangmuirProbeDataFromXdrive(dischargeID: str) -> str|list[list]:
    ''' This functions reads the electron density and electron temperature measured by the pop-up Langmuir Probes in "dischargeID" from xdrive
        LPs are numbered as follows: TM2h07 holds probe 0 to 5, TM3h01 6 to 13, TM8h01 14 to 17 -> index_divertorUnits applies same scheme
        
        If data is available:
        Returns electron temperature Te in [eV], electron density ne in [1/m^3], and corresponding mesurement times in [s] for upper and lower divertor unit
        Each of them is nested list of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)
        Additionally returns information about which LPs were active (index_upper and index_lower provide the indices for the active probes)
        All these lists are elements of the returned list

        If no data is available or wrong units are used: string is returned
        '''
    #test if langmuir probe data is available for that discharge
    xdrive_directory = "//x-drive/Diagnostic-logbooks/QRP-LangmuirProbes/QRP02-Divertor Langmuir Probes/Analysis/OP2/"
    #directory = f"{xdrive_directory}/{dischargeID}/"
    pathLP = f"{xdrive_directory}/{dischargeID}/"#directory
    print(pathLP)

    if not os.path.exists(pathLP):
        print('No LP data available')
        return 'No LP data available'
    
    #internal naming of probes in xdrive
    probes_lower = [50201, 50203, 50205, 50207, 50209, 50211,  50218, 50220, 50222, 50224, 50226, 50228, 50230, 50232, 50246, 50248, 50249, 50251]
    probes_upper = [51201, 51203, 51205, 51207, 51209, 51211,  51218, 51220, 51222, 51224, 51226, 51228, 51230, 51232, 51246, 51248, 51249, 51251]
    
    #index lists for each divertor unit in case that all LPs were active
    index_lower = list(range(18))
    index_upper = list(range(18))
    
    #data stores ne, Te, t, fails_ record missing LP data
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
def readAllShotNumbersFromLogbook(config: str, 
                                  filterSelected: str, 
                                  filesExist: bool =False, 
                                  safe: str ='results/configurations/dischargeList_OP223_', 
                                  overviewTableLink: str ='results/calculationTables/results_') -> pd.DataFrame|str:
    ''' This function is responsible for finding all discharges in the Logbook according to the given filter in OP2.2 and OP2.3
        "config" applies internal configuration filter
        "filterSelected" is externally given filter (no conditioning, no gas valve tests, no sniffer tests,...)
        "filesExist" determines, if possibly existing files with discharges for that configuration are overwritten or read out and returned
        "safe" is the basic structure of the path where the created .csv file is saved
        "overviewTableLink" is the basic structure of the path where latter calculation tables for a discharge are going to be safed

        returns string if no discharges are found
        returns pd.DataFrame if discharges are found
        keys: "configuration", "dischargeID", "duration", "duration_planned", "durationHeating", and default safe for calculation tables "overviewTable"
        "duration" is time difference between trigger t1 and t4, "duration_planned" is planned ECRH duration, "durationHeating" is summed duration of ECRH and NBI
        DataFrame is saved as .csv file under modified "safe"'''
    
    if filesExist:  #existing files are read out, missing ones are created
        if os.path.isfile(safe + config + '.csv'):
            print('confiuration discharge file exists and is read out')
            return pd.read_csv(safe + config + '.csv', sep=';')
        
    else:           #existing files are overwritted, missing ones are created
        pass

    #where to read data from (logbook for discharges and configurations, archive for trigger)
    url = 'https://w7x-logbook.ipp-hgw.mpg.de/api/_search'#'https://w7x-logbook.ipp-hgw.mpg.de/api/_search'
    urlTriggerBase = 'http://archive-webapi.ipp-hgw.mpg.de/programs.json?from='

    #filter for configurations (config)
    q7 = 'tags.value:"{config}"'.format(config=config)

    #combine neccessary filters   
    q = filterSelected + q7 
    p = {'time':'[2024 TO 2025]', 'size':'9999', 'q' : q }   # OP2.2 and OP2.3

    dischargeIDs, duration_Trigger, duration_planned, duration_ECRH_NBI, overviewTables = [], [], [], [], []    

    res = requests.get(url, params=p).json()#, verify=certifi.where()).json()

    if res['hits']['total']==0:
        print('no discharges found')
        return 'no discharges found'

    for discharge in res['hits']['hits']:
        #discharge['_id'] is e.g. 'XP_20250508.1'
        #should be transformed to '20250508.001'

        dischargeID = discharge['_id'].split('.')
        if len(dischargeID[1]) == 3:
            dischargeID = dischargeID[0][3:] + '.' + dischargeID[1]
        elif len(dischargeID[1]) == 2:
            dischargeID = dischargeID[0][3:] + '.0' + dischargeID[1]
        else:
            dischargeID = dischargeID[0][3:] + '.00' + dischargeID[1]

        #try to read heating duration of ECRH and NBI (not accurate!)
        heating_data = w7xarchive.get_signal_for_program({'ECRH': "Test/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V1/0/Ptot_ECRH",
                                                          'NBIS3': "ArchiveDB/codac/W7X/ControlStation.2176/BE000_DATASTREAM/4/s3_Pel/scaled", 
                                                          'NBIS4': "ArchiveDB/codac/W7X/ControlStation.2176/BE000_DATASTREAM/5/s4_Pel/scaled", 
                                                          'NBIS7': "ArchiveDB/codac/W7X/ControlStation.2092/BE000_DATASTREAM/4/s7_Pel/scaled", 
                                                          'NBIS8': "ArchiveDB/codac/W7X/ControlStation.2092/BE000_DATASTREAM/5/s8_Pel/scaled", 
                                                          'ne': "ArchiveDB/raw/W7X/ControlStation.2185/Density_DATASTREAM/0/Density"},
                                                          dischargeID)
        #merge heating DataFrames on rounded time axis
        for i, key in enumerate(heating_data.keys()):
            if i == 0:
                merged_heating = pd.DataFrame({'time': np.round(np.array(heating_data[key][0]), 3), key: heating_data[key][1]})
            else:
                merged_heating = pd.merge(merged_heating, pd.DataFrame({'time': np.round(np.array(heating_data[key][0]), 3), key: heating_data[key][1]}), 'outer', on='time')
        
        #merged is only created if any data stream is existant
        if len(heating_data.keys()) > 1:
            heating = np.zeros_like(np.array(merged_heating['time']))
            for key in merged_heating.keys():
                if 'ECRH' in key:
                    heating = heating + np.nan_to_num(np.array(merged_heating[key]))/1000
                elif 'NBI' in key:
                    heating = heating + np.nan_to_num(np.array(merged_heating[key]))
            merged_heating['heating'] = heating

            filter_heating_discharge = [x > 1 for x in merged_heating['heating']]
            if 'ne' in heating_data.keys():
                filter_ne_discharge = [x > 1e19 for x in merged_heating['ne']]
                if sum(filter_heating_discharge) == 0 or sum(filter_ne_discharge) == 0:
                    continue
            else:
                print('Is that a real discharge? '+ str(dischargeID))
                continue
            
            filter_heating = [x > 0.1 for x in merged_heating['heating']]
            if sum(filter_heating) > 0:
                duration_ECRH_NBI.append(merged_heating['time'][len(filter_heating) - 1 - filter_heating[::-1].index(True)] - merged_heating['time'][filter_heating.index(True)])
            else:
                if True:
                    duration_ECRH_NBI.append(np.nan)

        #if no data stream existed append np.nan
        else:
            print('Is that a real discharge? '+ str(dischargeID))
            continue#duration_ECRH_NBI.append(np.nan)
        
        dischargeIDs.append(dischargeID) 
        overviewTables.append(overviewTableLink + dischargeID + '.csv')
        print(dischargeID)

        #read trigger times of t1 and t4 if they both exist (otherwise append np.nan to duration_Trigger)
        urlTrigger = urlTriggerBase + dischargeID + '#'
        resTrigger = requests.get(urlTrigger).json()
        if 'programs' in resTrigger.keys():
            if type(resTrigger['programs']) == list:
                if 'trigger' in resTrigger['programs'][0].keys():
                    if type(resTrigger['programs'][0]['trigger']) == dict:
                        if '1' in resTrigger['programs'][0]['trigger'].keys() and '4' in resTrigger['programs'][0]['trigger'].keys():
                            if type(resTrigger['programs'][0]['trigger']['4']) == list and type(resTrigger['programs'][0]['trigger']['1']) == list:
                                if len(resTrigger['programs'][0]['trigger']['4']) > 0 and len(resTrigger['programs'][0]['trigger']['1']) > 0:
                                    duration_Trigger.append((resTrigger['programs'][0]['trigger']['4'][0] - resTrigger['programs'][0]['trigger']['1'][0])/1e9)
                                else:
                                    print('Trigger "1" or "4" is an empty list')
                                    duration_Trigger.append(np.nan)
                            else:
                                print('Trigger "1" or "4" is not a list')
                                duration_Trigger.append(np.nan)
                        else:
                            print('Trigger "1" or "4" does not exist')
                            duration_Trigger.append(np.nan)
                    else:
                        print('Trigger is not a dictionary')
                        duration_Trigger.append(np.nan)
                else:
                    print('No trigger in "programs"')
                    duration_Trigger.append(np.nan)
            else:
                print('"programs" is no list')
                duration_Trigger.append(np.nan)
        else:
            print('Key "programs" does not exist')
            duration_Trigger.append(np.nan)


        #try to get planned ECRH heating durtion (planned != real (program abort) and no inclusion of NBI heating nor ICRH heating)
        for tag in discharge['_source']['tags']: 
            if 'catalog_id' in tag.keys():
                if tag['catalog_id'] == '1#3':
                    duration_planned.append(tag['ECRH duration'])
                    continue
        #if no ECRH duration tag was found
        if len(duration_planned) != len(dischargeIDs):
            duration_planned.append(np.nan)

        if abs(duration_planned[-1] - duration_ECRH_NBI[-1]) > 20:
            if True:
                print('control that discharge')
    
    configuration = [config] * len(dischargeIDs)

    dischargeTable = pd.DataFrame({'configuration': configuration, 'dischargeID': dischargeIDs, 'duration': duration_Trigger, 'duration_planned': duration_planned, 'durationHeating': duration_ECRH_NBI, 'overviewTable': overviewTables})
    dischargeTable.to_csv(safe + config + '.csv', sep=';')
    
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

