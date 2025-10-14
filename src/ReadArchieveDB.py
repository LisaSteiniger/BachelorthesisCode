'''This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
It also works with juice to find information about the discharges (percentage of EIM/FTM/KJM, ...).'''

import w7xarchive
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
            #called like this time is in seconds from t1
            data_url_OP1_LP_ne = "ArchiveDB/raw/Minerva/Minerva.ElectronDensity.QRP.{divertorUnit}.{number}/ne_DATASTREAM/V1/0/ne".format(divertorUnit=divertorUnitOP1, number=numberOP1)
            data_url_OP1_LP_Te = "ArchiveDB/raw/Minerva/Minerva.ElectronTemperature.QRP.{divertorUnit}.{number}/Te_DATASTREAM/V1/0/Te".format(divertorUnit=divertorUnitOP1, number=numberOP1)
            times, data_OP1_ne = w7xarchive.get_signal_for_program(data_url_OP1_LP_ne, shotnumber)
            times, data_OP1_Te = w7xarchive.get_signal_for_program(data_url_OP1_LP_Te, shotnumber)
            #just to test if reading the data works properly
            plt.figure()
            plt.plot(times, data_OP1_ne)
            plt.plot(times, data_OP1_Te)
            plt.gca().set_xlabel("time [s]")
            plt.gca().set_ylabel("data")
            plt.show()
            pass

for shotnumber in shotnumbersOP2:
    #for OP2
    for divertorUnitOP2 in divertorUnitsOP2:
        for numberOP2 in range(1, 2):#14):
            #called like this time is in seconds from t1
            data_url_OP2_LP_ne = "ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnit}_Probe_{number}_DATASTREAM/V1/1/Plasma_Density".format(divertorUnit=divertorUnitOP2, number=numberOP2)
            data_url_OP2_LP_Te = "ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnit}_Probe_{number}_DATASTREAM/V1/2/Electron_Temperature".format(divertorUnit=divertorUnitOP2, number=numberOP2)
            times, data_OP2_ne = w7xarchive.get_signal_for_program(data_url_OP2_LP_ne, shotnumber)
            times, data_OP2_Te = w7xarchive.get_signal_for_program(data_url_OP2_LP_Te, shotnumber)
            #just to test if reading the data works properly
            plt.figure()
            plt.plot(times, data_OP1_ne)
            plt.plot(times, data_OP1_Te)
            plt.gca().set_xlabel("time [s]")
            plt.gca().set_ylabel("data")
            plt.show()
            pass