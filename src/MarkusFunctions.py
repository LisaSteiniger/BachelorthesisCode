import os
import numpy as np
import pandas as pd
import src.ProcessData as process

def readMarkusData(settings_dict: dict, interval: int, settings_Gao:dict) -> tuple[list, list, list, list, list, list]:
    ''' This function reads data from the files Markus Kandler downloaded for his Bachelor Thesis and prepares it for further processing
        "interval" determines how many time frames are averaged to obtain the measurement value
        "settings_*" is used to definethe paths to the downloaded files

        Returns electron temperature Te in [eV] and electron density ne in [1/m^3] plus corresponding mesurement times in [s] of langmuir probes
        Returns surface temperature of divertor in [°C] at position closest to langmuir probes plus corresponding measurement time in [s]
        Each of them is an array like object of ndim=2 with each line representing measurements over time at one LP probe positions (=each column representing measurements at all positions at one time)'''
    
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
def compareMarkus(safe: str, tableOverview: pd.DataFrame) -> pd.DataFrame:
    ''' This function is responsible for the comparison of the results delivered by Markus Kandlers modified program vs. the calculation performed by this project
        The files of M. Kandler must be created seperately and beforehand by running "Python skripts by Markus Kandler" "FinaleRechnungen.py"
        
        Returns DataFrame with the sputtering yields and erosion rates as well as the difference "Delta_*" between the two programs results'''
    dataMarkus = pd.read_csv('{safe}_Markus.csv'.format(safe=safe[:-4]), sep=';')
        
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['{key}_Markus'.format(key=key)] = dataMarkus[key]
    
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['Delta_{key}'.format(key=key)] = np.round((tableOverview[key] - dataMarkus[key])/tableOverview[key], 10)

    return tableOverview

#######################################################################################################################################################################
def processMarkusData(alpha: int|float, LP_zeta: list[int|float], m_i: list[int|float], f_i: list[int|float], ions: list[str], k: int|float, n_target: int|float, interval: int = 50) -> None: 
    ''' This function fully processes M. Kandlers data sets from reading, over processing them to calculating sputtering yields and layer thicknesses
        -> reads data for exemplary discharges from OP1.2b from downloaded files, processes it and writes it to an overview .csv file together with calculated values for erosion related quamtities
        "alpha" is the incident angle of ions on the target in [rad] (measured from the surface normal)
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad] 
        "m_i", "f_i", "ions" must be masses in [kg], concentrations [unitless], and names of impurities (same length of the lists) 
        "k" is the Boltzmann constant in [eV/K]
        "n_target" is the atomic density of the divertor target in [1/m^3]
        "interval" = 50 -> 50 subsequent values of the value arrays below are averaged'''
    
    #to identify path to downloaded langmuir probe files with electron density and temperature
    settings_dict_list = [{'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'lowerTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},

                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'},
                        {'exp': '20180807.014', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 8.0],'m':'DWSE' ,'s':'VmaxClusterSlicer(hp=2)'}]#,

                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'},
                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'},
                        #{'exp': '20180814.006', 'c':'upperTestDivertorUnit' ,'tw':[0.0, 10.5],'m':'minerva' ,'s':'PF'}]
    
    #to identify path to downloaded IRcam files with divertor surface temperatures
    settings_Gao_list = [{'exp': '20180807', 'discharge': '014', 'divertor' : '3lh', 'finger' : '11'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '3lh', 'finger' : '12'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '3lh', 'finger' : '13'},

                        {'exp': '20180807', 'discharge': '014', 'divertor' : '5uh', 'finger' : '11'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '5uh', 'finger' : '12'},
                        {'exp': '20180807', 'discharge': '014', 'divertor' : '5uh', 'finger' : '13'}]#,

                        #{'exp': '20180814', 'discharge': '006', 'divertor' : '5uh', 'finger' : '11'},
                        #{'exp': '20180814', 'discharge': '006', 'divertor' : '5uh', 'finger' : '12'},
                        #{'exp': '20180814', 'discharge': '006', 'divertor' : '5uh', 'finger' : '13'}]

    for settings_dict, settings_Gao in zip(settings_dict_list, settings_Gao_list):
        ne_values, Te_values, positions, data_Ts, t, S = readMarkusData(settings_dict, interval, settings_Gao) 
        #ne in [1/m^3], Te in [eV], positions in [m], data_Ts in [°C], t in [s], S in [m]

        #modify data arrays so that they fit the functions below
        tii = np.hstack(t)           #times for Gao
        ti = (tii/2) * 1004/interval #same number of times for Lukas data
        tiii = ti.astype(int)

        #take the right amount of averages from Lukas data (here every tenth value)
        n_e_values = []
        T_e_values = []
        for ti in tiii:
            if ti > len(ne_values) - 1: 
                ti = len(ne_values) - 1
            n_e_values.append(ne_values[ti])
            T_e_values.append(Te_values[ti])
        n_e_values = np.array(n_e_values)
        T_e_values = np.array(T_e_values)

        #surface temperature array has less measurement positions, extrapolate to further end of target finger by appending the last value as often as neccessary
        T_s_values = []
        if len(data_Ts[0]) < len(n_e_values[0]):
            for Ts in data_Ts:
                Ts = Ts.tolist()
                for i in range (len(n_e_values[0]) - len(data_Ts[0])):
                    Ts.append(Ts[-1]) 
                #print(Ts)
                T_s_values.append(Ts)
        T_s_values = np.array(T_s_values)
        T_i_values = T_e_values

        dt = [0]
        for i, t_ii in enumerate(tii):
            if i < (len(tii) - 1):
                dt.append(tii[1 + i] - t_ii)
        dt = np.array(dt)

        #test for same shape, 9 times (lines) and 15 positions (columns)
        #print(np.shape(n_e_values))
        #print(np.shape(T_e_values))
        #print(np.shape(T_s_values))
        #print(np.shape(dt))
        
        #calculates erosion related quantities and writes to .csv file saved as "safe" together with used measurement data
        safe = 'results/compareMarkus/tableCompareMarkus_{exp}.{discharge}_{divertorUnit}_{moduleFinger}.csv'.format(exp=settings_Gao['exp'], discharge=settings_Gao['discharge'], divertorUnit=settings_Gao['divertor'], moduleFinger=settings_Gao['finger'])
        calculateErosionRelatedQuantitiesSeveralPositions(T_s_values.T, T_e_values.T, T_i_values.T, n_e_values.T, alpha, LP_zeta, dt, safe, m_i, f_i, ions, k, n_target, compareResults=True)

#######################################################################################################################################################################
def calculateErosionRelatedQuantitiesSeveralPositions(T_s_values: list[list], 
                                                      T_e_values: list[list], 
                                                      T_i_values: list[list], 
                                                      n_e_values: list[list], 
                                                      dt: list, 
                                                      alpha: int|float, 
                                                      LP_zeta: list[int:float],
                                                      safe: str, 
                                                      m_i: list[int|float], 
                                                      f_i: list[int|float], 
                                                      ions: list[str], 
                                                      k: int|float, 
                                                      n_target: int|float, 
                                                      compareResults: bool =False) -> None:
    ''' This function calculates sputtering yields for hydrogen, deuterium, tritium, carbon and oxygen on carbon targets, the combined erosion rates and layer thicknesses
        Results are written to an .csv file but not returned as a DataFrame

        "*_values" arrays are 2 dimensional with columns representig the time and lines representing positions, e.g _values[0] being the array _values at position 1 over all times
        -> electron density ne in [1/m^3], electron temperature Te and ion temperature Ti in [eV], surface temperature of the target Ts in [K]
        "dt" is a one dimensional array holding the corresponding time steps to "*_values" in [s]
        "alpha" is the ion incident angle in rad
        The incident angle of the magnetic field lines on the target is given at each langmuir probe position from the target surface towards the surface normal in "LP_zeta" in [rad]
        "safe" determines where to save the resulting .csv file
        "m_i", "f_i", "ions" must be masses in [kg], concentrations [unitless], and names of impurities (same length of the lists) 
        "k" is the Boltzmann constant in [eV/K]
        "n_target" is the atomic density of the divertor target in [1/m^3]
        "compareResults" determines if results should be compared to results of M. Kandlers modified program'''
    position_counter = 0 #for how many positions there are usable measurement values (= equal number of T_s, T_e, n_e, dt values)?

    #arrays will be 2 dimensional with e.g Y_H[0] being the array of erosion yields at position 1 over all times
    Y_H, Y_D, Y_T, Y_C, Y_O = [], [], [], [], [] 
    erosionRate_dt_position, erodedLayerThickness_dt_position, erodedLayerThickness_position = [], [], []
    depositionRate_dt_position, depositedLayerThickness_dt_position, depositedLayerThickness_position = [], [], []

    
    for T_s, T_e, T_i, n_e in zip(T_s_values, T_e_values, T_i_values, n_e_values): #calculate erosion related quantities for each position
        if len(T_i) == len(T_s) and len(T_i) == len(n_e) and len(T_i) == len(dt): #otherwise code won't run
            position_counter += 1

            #returns arrays over all time steps at this location
            return_erosion = process.calculateErosionRelatedQuantitiesOnePosition(T_e, T_i, T_s, n_e, dt, alpha, LP_zeta[position_counter-1], m_i, f_i, ions, k, n_target)
            if type(return_erosion) == str:
                continue
            Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = return_erosion
            Y_H.append(Y_0)
            Y_D.append(Y_1)
            Y_T.append(Y_2)
            Y_C.append(Y_3)
            Y_O.append(Y_4)
            erosionRate_dt_position.append(erosionRate_dt)
            erodedLayerThickness_dt_position.append(erodedLayerThickness_dt)
            erodedLayerThickness_position.append(erodedLayerThickness)
            depositionRate_dt_position.append(depositionRate_dt)
            depositedLayerThickness_dt_position.append(depositedLayerThickness_dt)
            depositedLayerThickness_position.append(depositedLayerThickness)

    #times are not actually times in [s] but numbered time steps 1 to x
    time = []
    time = [list(range(1, len(Y_H[0]) + 1))] * position_counter 
    #print(time)  

    #positions are no actually distances from pumping gap in [m] but numbered langmuir probes
    position = []
    for i in range(1, position_counter + 1):
        position.append([i] * (len(time[0])))
    np.hstack(position)
    #print(position)

    #print(len(np.hstack(position)), len(np.hstack(Y_H)), len(np.hstack(Y_D)), len(np.hstack(Y_T)), len(np.hstack(Y_C)), len(np.hstack(Y_O)), len(np.hstack(erosionRate_dt_position)), len(np.hstack(erodedLayerThickness_dt_position)), len(np.hstack(erodedLayerThickness_position)))
    tableOverview = {'LangmuirProbe':np.hstack(position),
                        #'Position':
                        'time': np.hstack(time),
                        'Y_H':np.hstack(Y_H), 
                        'Y_D':np.hstack(Y_D),  
                        'Y_T':np.hstack(Y_T),  
                        'Y_C':np.hstack(Y_C), 
                        'Y_O':np.hstack(Y_O), 
                        'erosionRate':np.hstack(erosionRate_dt_position),
                        'erodedLayerThickness':np.hstack(erodedLayerThickness_dt_position),
                        'totalErodedLayerThickness':np.hstack(erodedLayerThickness_position),
                        'depositionRate':np.hstack(depositionRate_dt_position),
                        'depositedLayerThickness':np.hstack(depositedLayerThickness_dt_position),
                        'totalDepositedLayerThickness':np.hstack(depositedLayerThickness_position)}
    tableOverview = pd.DataFrame(tableOverview)
    
    #if results should be compared to Markus Kandlers results
    if compareResults==True:
        tableOverview = compareMarkus(safe, tableOverview)
    
    #does not return anything specifically but prints results to .csv file
    tableOverview.to_csv(safe, sep=';')
