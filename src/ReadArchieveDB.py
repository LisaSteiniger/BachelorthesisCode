'''This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
It also works with juice to find information about the discharges (percentage of EIM/FTM/KJM, ...).'''

import w7xarchive   #see https://git.ipp-hgw.mpg.de/kjbrunne/w7xarchive/-/blob/master/doc/workshop.ipynb for introduction
import matplotlib.pyplot as plt

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