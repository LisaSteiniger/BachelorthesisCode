'''This file contains the functions neccessary for pulling data from w7xArchieveDB and finding typical parameters for discharges of certain configurations.
It also works with juice to find information about the discharges (percentage of EIM/FTM/KJM, ...).'''

import w7xarchive

shotnumbers = ['20181010.37']

#for OP1.2
#Langmuir Probes, #numbersOP1 in range [1, 20]
divertorUnitsOP1 = ['lowerTestDivertorUnit', 'upperTestDivertorUnit']
data_url_OP1_LP_ne = r"ArchiveDB/raw/Minerva/Minerva.ElectronDensity.QRP.{divertorUnitOP1}.{numberOP1}/ne_DATASTREAM/V1/0/ne"
data_url_OP1_LP_Te = r"ArchiveDB/raw/Minerva/Minerva.ElectronTemperature.QRP.{divertorUnitOP1}.{numberOP1}/Te_DATASTREAM/V1/0/Te"

#for OP2
#Langmuir Probes, numbersOP2 in range [1, 14]
divertorUnitsOP2 = ['LowerDivertor', 'UpperDivertor']
data_url_OP2_LP_ne = r"ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnitOP2}_Probe_{numberOP2}_DATASTREAM/V1/1/Plasma_Density"
data_url_OP2_LP_Te = r"ArchiveDB/raw/W7XAnalysis/QRP02_Langmuirprobes/{divertorUnitOP2}_Probe_{numberOP2}_DATASTREAM/V1/2/Electron_Temperature"

for shotnumber in shotnumbers:
    #for OP1
    for divertorUnitOP1 in divertorUnitsOP1:
        for numberOP1 in range(1, 21):
            #called like this time is in seconds from t1
            times, data_OP1_ne = w7xarchive.get_signal_for_program(data_url_OP1_LP_ne, shotnumber)
            times, data_OP1_Te = w7xarchive.get_signal_for_program(data_url_OP1_LP_Te, shotnumber)
            pass
    #for OP2
    for divertorUnitOP1 in divertorUnitsOP1:
        for numberOP1 in range(1, 21):
            #called like this time is in seconds from t1
            times, data_OP2_ne = w7xarchive.get_signal_for_program(data_url_OP2_LP_ne, shotnumber)
            times, data_OP2_Te = w7xarchive.get_signal_for_program(data_url_OP2_LP_Te, shotnumber)
            pass