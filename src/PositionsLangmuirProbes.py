''' This file contains the positions of the Langmuir Probes  for OP1.2 and OP.2. 
    They are given as the distance in meters from the pumping gap or with ending on xyz in x, y, z coordinates.
    In OP1.2 the Langmuir Probes are located in divertor module 5 on the horizontal target elements 3 and 4 of upper and lower divertor unit 
    In OP2 they are located in divertor module 5 on the horizontal target modules 2h07, 3h01, and 8h01 of upper and lower divertor unit
    Found in   https://wikis.ipp-hgw.mpg.de/W7X/index.php/QRP_-_Langmuir_Probe_Array_(Divertor),_OP1_flush_mounted,_OP2_pop-up
               https://w7x-logbook.ipp-hgw.mpg.de/components?id=QRP02#
    
    Additionally, the local incident angle of the magnetic field lines on the target zeta is given at each langmuir probe position in [rad]
    -> measured from the surface of the target towards the surface normal'''

import numpy as np

OP1_ProbeTM3 = ['TE3-01', 'TE3-02', 'TE3-03', 'TE3-04', 'TE3-05', 'TE3-06', 'TE3-07', 'TE3-08', 'TE3-09', 'TE3-10']
OP1_ProbeTM4 = ['TE4-11', 'TE4-12', 'TE4-13', 'TE4-14', 'TE4-15', 'TE4-16', 'TE4-17', 'TE4-18', 'TE4-19', 'TE4-20']
OP1_lowerModuleTM3Distances = [0.42138, 0.39643, 0.37149, 0.34655, 0.32163, 0.24694, 0.22209, 0.19727, 0.17252, 0.14784]
OP1_lowerModuleTM4Distances = [0.43686, 0.41190, 0.38695, 0.36201, 0.33708, 0.26235, 0.23749, 0.21265, 0.18785, 0.16312]
OP1_upperModuleTM3Distances = [0.42140, 0.39643, 0.37148, 0.34655, 0.32162, 0.24694, 0.22210, 0.19726, 0.17252, 0.14785]
OP1_upperModuleTM4Distances = [0.43685, 0.41190, 0.38696, 0.36201, 0.33709, 0.26235, 0.23749, 0.21265, 0.18784, 0.16311] 

OP2_ProbeTM2 = ['TE2-01', 'TE2-02', 'TE2-03', 'TE2-04', 'TE2-05', 'TE2-06'] #do not know correct names/numbers
OP2_ProbeTM3 = ['TE3-01', 'TE3-02', 'TE3-03', 'TE3-04', 'TE3-05', 'TE3-06', 'TE3-07', 'TE3-08'] #do not know correct names/numbers
OP2_ProbeTM8 = ['TE8-11', 'TE8-12', 'TE8-13', 'TE8-14', 'TE8-15', 'TE8-16', 'TE8-17', 'TE8-18', 'TE8-19', 'TE8-20'] #do not know correct names/numbers
OP2_TM2Distances = [0.10602897, 0.13182066, 0.15761507, 0.18341105, 0.20920802, 0.23500566] #Are those the right six values as TM2h07 is said to have Langmuir Probes closer to pumping gap?
OP2_TM3Distances = [0.32530727, 0.35110524, 0.37690349, 0.40270196, 0.42850063, 0.45429944, 0.48009839, 0.50589745] #Are those the right eight values or are they messed up with TM2h07?
OP2_TM8Distances = [0.09177, 0.11728, 0.13278, 0.15828] 

#xyz values for upper divertor, they differ for the lower divertor unit!
OP2_TM2xyz = [[2.52028, -4.64803, 0.99334],
              [2.50815, -4.62569, 0.98894],
              [2.49603, -4.60334, 0.98454],
              [2.4839,  -4.581,   0.98015],
              [2.47178, -4.55865, 0.97575],
              [2.45965, -4.53631, 0.97135]]

OP2_TM3xyz = [[2.64076, -4.88656, 1.03841],
              [2.62866, -4.86418, 1.03412],
              [2.61656, -4.8418,  1.02983],
              [2.60446, -4.81942, 1.02554],
              [2.59236, -4.79704, 1.02125],
              [2.58026, -4.77467, 1.01696],
              [2.56816, -4.75229, 1.01267],
              [2.55605, -4.72991, 1.00838]]

OP2_TM8xyz = [[0.29669, -5.93902, 0.71571],
              [0.29528, -5.91356, 0.71573],
              [0.29442, -5.89808, 0.71575],
              [0.29301, -5.87262, 0.71577]]

OP1_TM3zeta = [np.deg2rad(2.)] * len(OP1_ProbeTM3)
OP1_TM4zeta = [np.deg2rad(2.)] * len(OP1_ProbeTM4)

OP2_TM2zeta = list(map(np.deg2rad, [1.5, 1, 0.5, 0, 0, 0.5]))
OP2_TM3zeta = list(map(np.deg2rad, [2, 2, 2.5, 2.5, 3, 3, 3.5, 4]))
OP2_TM8zeta = [np.deg2rad(2.)] * len(OP2_TM8Distances)