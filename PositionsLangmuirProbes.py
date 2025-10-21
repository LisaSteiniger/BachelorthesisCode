#This file contains the positions of the Langmuir Probes  for OP1.2 and OP.2. 
#They are given as the distance in centimeters from the pumping gap.
#In OP1.2 they are located in divertor module 5 on the horizontal target elements 3 and 4 of upper and lower divertor unit 
#In OP2 they are located in divertor module 5 on the horizontal target elements 2, 3, and 8 of upper and lower divertor unit
#Found in   https://wikis.ipp-hgw.mpg.de/W7X/index.php/QRP_-_Langmuir_Probe_Array_(Divertor),_OP1_flush_mounted,_OP2_pop-up
#           https://w7x-logbook.ipp-hgw.mpg.de/components?id=QRP02#

OP1_ProbeTE3 = ['TE3-01', 'TE3-02', 'TE3-03', 'TE3-04', 'TE3-05', 'TE3-06', 'TE3-07', 'TE3-08', 'TE3-09', 'TE3-10']
OP1_ProbeTE4 = ['TE4-11', 'TE4-12', 'TE4-13', 'TE4-14', 'TE4-15', 'TE4-16', 'TE4-17', 'TE4-18', 'TE4-19', 'TE4-20']
OP1_lowerModuleTE3Distances = [421.38, 396.43, 371.49, 346.55, 321.63, 246.94, 222.09, 197.27, 172.52, 147.84]
OP1_lowerModuleTE4Distances = [436.86, 411.90, 386.95, 362.01, 337.08, 262.35, 237.49, 212.65, 187.85, 163.12]
OP1_upperModuleTE3Distances = [421.40, 396.43, 371.48, 346.55, 321.62, 246.94, 222.10, 197.26, 172.52, 147.85]
OP1_upperModuleTE4Distances = [436.85, 411.90, 386.96, 362.01, 337.09, 262.35, 237.49, 212.65, 187.84, 163.11] 

OP2_ProbeTE2 = ['TE2-01', 'TE2-02', 'TE2-03', 'TE2-04', 'TE2-05', 'TE2-06'] #do not know correct names/numbers
OP2_ProbeTE3 = ['TE3-01', 'TE3-02', 'TE3-03', 'TE3-04', 'TE3-05', 'TE3-06', 'TE3-07', 'TE3-08'] #do not know correct names/numbers
OP2_ProbeTE8 = ['TE8-11', 'TE8-12', 'TE8-13', 'TE8-14', 'TE8-15', 'TE8-16', 'TE8-17', 'TE8-18', 'TE8-19', 'TE8-20'] #do not know correct names/numbers
OP2_TE2Distances = [106.02897, 131.82066, 157.61507, 183.41105, 209.20802, 235.00566] #Are those the right six values as TE2 is said to have Langmuir Probes closer to pumping gap?
OP2_TE3Distances = [325.30727, 351.10524, 376.90349, 402.70196, 428.50063, 454.29944, 480.09839, 505.89745] #Are those the right eight values or are they messed up with TE2?
OP2_TE8Distances = [158.28, 132.78, 117.28, 91.77] 

#Reading from the Archive

#Langmuir Probes provide separate datastream for time (for OP2, W7-X time stamp in ns), plasma density (1e18 1/m^3), electron temperature (eV), and their errors (equivalent units)
#Access OP1.2 through ArchiveDB/raw/Minerva/Minerva.ElectronDensityQRP...   -> get values from pd.dataFrame['values'] and the time stamp from ['dimensions']
#       OP2   through ArchiveDB/raw/W/XAnalysis/QRP02_Langmuirprobes/...    -> get values from pd.dataFrame['values'] and the time stamp from ['dimensions'] or from separate time trace

#IR Cameras provide datastream for maximum temperature of each target module or Temperature distribution over different positions (°C) 
#Access through  ArchiveDB/Test/raw/W7XAnalysis/QRT_IRCAM_new/AEF50_maxT_TMs_DATASTREAM             -> get values from pd.dataFrame['values'] and the time stamp from ['dimensions']
#Access through  ArchiveDB/Test/raw/W7XAnalysis/QRT_IRCAM_new/AEF50_temperature_tar_baf_DATASTREAM  -> get values from pd.dataFrame['values'] and the time stamp from ['dimensions']
#                                                                                                   -> values are nested array, first dimension corresponds to time, second to positions?
#                                                                                                   -> where do I get positions from?
