'''This file contains functions to calculate sputtering yields of chemical, physical, and total sputtering. It is applicable to physical sputtering processes with targets made from
graphite (C), Be, Fe, Mo, or W under bombardement with hydrogen (H), deuterium (D), tritium (T) or helium (He). The description of the used variables can be found in variablesDescription.txt'''

import numpy as np
import scipy.constants
import scipy.integrate as integrate
#import pandas as pd

#######################################################################################################################################################################
#initialize common parameter values
e  =  scipy.constants.elementary_charge 
k_B = scipy.constants.Boltzmann
k = k_B/e 

#lines for [Be, C, Fe, Mo, W] and columns with [H, D, T, He, Self-Sputtering, O] (O only known for C), in [eV]
E_TF = np.array([[256, 282, 308, 720, 2208, 0],
                 [415, 447, 479, 1087, 5688, 9298],
                 [2544, 2590, 2635, 5517, 174122, 0], 
                 [4719, 4768, 4817, 9945, 533127, 0],
                 [9871, 9925, 9978, 20376, 1998893, 0]]) 

#[Be, C, Fe, Mo, W], wo kommen die her, Einheit?! -> muss [eV] stimmt aber glaub ich auch   
E_s = np.array([3.38, 7.42, 4.34, 6.83, 8.68])

#Parameters for chemical erosion of C by H-isotopes, [H, D, T] 
Q_y_chem = np.array([0.035, 0.1, 0.12])                                                                  
C_d_chem = np.array([250, 125, 83])                                                                               
E_thd_chem = [15, 15, 15]   #threshold energy for Y_damage                                                                      
E_ths_chem = [2, 1, 1]      #threshold energy for Y_surf                                                                        
E_th_chem=[31, 27, 29]   

#Parameters for net erosion specifically for divertor
lambda_nr, lambda_nl = 1, -1     #nonsense values, just signs are correct

#######################################################################################################################################################################
#Calculation of the Physical Sputtering Yield of Graphite under Bombardement with Hydrogen, Deuterium, Tritium , Helium, and Carbon Particles (self-sputtering)

#calculation sputtering yield from N_emmited and N_incident
def calculatePhysicalSputteringYieldFromN(N_emmitted=1, N_incident=1):
    return N_emmitted/N_incident                                                

#calculate sputtering yield from E_dep and E_s, provide E_s in [eV], d in [angstroem], n_o_times_S_n in [eV/angstroem]   
def calculatePhysicalSputteringYieldFromE(E_s=5, d=5, n_o_times_S_n=0.1*e/(1e-10)):                                                                                  
    E_dep = n_o_times_S_n * d                                                                                                                                                         
    return E_dep/E_s                                                                             

#calculate physical sputtering yield for various target-ion combinations, provide E in [eV]
def calculatePhysicalSputteringYield(targetMaterial='C', incidentParticle='H', E=2, alpha=np.pi/2): 
    #lines for [Be, C, Fe, Mo, W] and columns with [H, D, T, He, Self-Sputtering, O] (O only known for C)
    Q_y_phys = np.array([[0.07, 0.11, 0.14, 0.28, 0.67, 0],
                         [0.05, 0.08, 0.10, 0.2, 0.75, 1.02], 
                         [0.07, 0.12, 0.16, 0.33, 10.44, 0], 
                         [0.05, 0.09, 0.12, 0.24, 16.27, 0], 
                         [0.04, 0.07, 0.1, 0.2, 33.47, 0]])                        
    E_th_phys = np.array([[13, 13, 15, 16, 24, 0],
                          [31, 28, 30, 32, 53, 61.54],
                          [61, 32, 23, 20, 31, 0], 
                          [172, 83, 56, 44, 49, 0],
                          [447, 209, 136, 102, 62, 0]])                    #in [eV]                                                                 
    M1 = np.array([[1.00794, 2.01210175, 3.0160495, 4.002602, 9.01218, 15.9994],
                   [1.00794, 2.01210175, 3.0160495, 4.002602, 12.011, 15.9994], 
                   [1.00794, 2.01210175, 3.0160495, 4.002602, 55.847, 15.9994], 
                   [1.00794, 2.01210175, 3.0160495, 4.002602, 95.94, 15.9994], 
                   [1.00794, 2.01210175, 3.0160495, 4.002602, 183.85, 15.9994]]) #in [u]=[g/mol]  
    Z1 = np.array([[1, 1, 1, 2, 4, 8], 
                   [1, 1, 1, 2, 6, 8], 
                   [1, 1, 1, 2, 26, 8], 
                   [1, 1, 1, 2, 42, 8], 
                   [1, 1, 1, 2, 74, 8]])

    #[Be, C, Fe, Mo, W]
    M2 = np.array([9.01218, 12.011, 55.847, 95.94, 183.85])             #in [u]=[g/mol]     
    Z2 = np.array([4, 6, 26, 42, 74])                           
    n = np.array([2.5, 2.5, 2.5, 2.5, 2.5])                             #wo kommen die her, Einheit?! -> muss [m^{-3}]      
    
    #choose applying parameters by setting the indices
    if targetMaterial=="Be":
        targetIndex = 0
    elif targetMaterial=="C":
        targetIndex = 1
    elif targetMaterial=="Fe":
        targetIndex = 2
    elif targetMaterial=="Mo":
        targetIndex = 3
    elif targetMaterial=="W":
        targetIndex = 4

    if incidentParticle=="H":
        incidentIndex = 0
    elif incidentParticle=="D":
        incidentIndex = 1
    elif incidentParticle=="T":
        incidentIndex = 2
    elif incidentParticle=="He":
        incidentIndex = 3
    elif incidentParticle=="Self":
        incidentIndex = 4
    elif incidentParticle=="O":
        incidentIndex = 5

    if E > E_th_phys[targetIndex, incidentIndex]:
        #calculate sputtering yield for normal incidence angle (0° towards target surface normal)
        epsilon = E/(E_TF[targetIndex, incidentIndex])
        s_n = (0.5 * np.log(1 + 1.2288 * epsilon))/(epsilon + 0.1728 * np.sqrt(epsilon) + 0.008 * epsilon**(0.1504))           
        Y0_phys = (Q_y_phys[targetIndex, incidentIndex] * s_n) * (1 - (E_th_phys[targetIndex, incidentIndex]/E)**(2/3)) * ((1 - (E_th_phys[targetIndex, incidentIndex]/E))**2)                                 

        #calculate sputtering yield for any incidence angle
        a_L = (0.04685/ ((((Z1[targetIndex, incidentIndex])**(2/3)) + ((Z2[targetIndex])**(2/3)))**(0.5))) * 1e-9               #in m         
        gamma = (4 * M1[targetIndex, incidentIndex] * M2[targetIndex])/((M1[targetIndex, incidentIndex] + M2[targetIndex])**2)  
        f_y = np.sqrt(E_s[targetIndex]) * (0.94 - 0.00133 * (M2[targetIndex]/M1[targetIndex, incidentIndex]))                   #E_s in [eV]                                              
        a_max = np.pi/2 - (a_L * n[targetIndex]**(1/3))/(np.sqrt(2 * epsilon * np.sqrt(E_s[targetIndex]/(gamma * E))))        #n in m{-3}, E_s in [eV], a_L in [m]
        Y_phys = (Y0_phys/((np.cos(alpha)**f_y))) * np.exp(f_y * (1 - (1/(np.cos(alpha)))) * np.cos(a_max))
    else:
        Y_phys = 0

    return Y_phys

#calculate physical sputtering yield for self-sputtering of carbon, provide E in [eV]
def calculatePhysicalSputteringYieldEckstein(E=2, alpha=np.pi/4):
    #Fit for Parameter f
    y0_f = 4.55878
    x0_f = 25.5644
    A1_f = 20.17943
    t1_f = 29.8123
    A2_f = 12.08692
    t2_f = 150.66038
    A3_f = 8.99236
    t3_f = 946.68968
    
    f_fit = y0_f + A1_f * np.exp(-(E - x0_f)/t1_f) + A2_f * np.exp(-(E - x0_f)/t2_f) + A3_f * np.exp(-(E - x0_f)/t3_f)
    
    #Fit f0r Parameter b
    y0_b = 1.222
    x0_b = 27.59683
    A1_b = 10.24535
    t1_b = 31.09355
    A2_b = 7.29825
    t2_b = 185.60025
    A3_b = 4.90847
    t3_b = 1040.42162
    
    b_fit = y0_b + A1_b * np.exp(-(E - x0_b)/t1_b) + A2_b * np.exp(-(E - x0_b)/t2_b) + A3_b * np.exp(-(E - x0_b)/t3_b)

    #Fit for Parameter c
    y0_c2 = 0.85257
    x0_c2 = 37.36542
    A1_c2 = -0.10577
    t1_c2 = 346.95644
    A2_c2 = -0.11142
    t2_c2 = 346.94662
    A3_c2 = -0.12915
    t3_c2 = 346.92395
    
    c2_fit = y0_c2 + A1_c2 * np.exp(-(E - x0_c2)/t1_c2) + A2_c2 * np.exp(-(E - x0_c2)/t2_c2) + A3_c2 * np.exp(-(E - x0_c2)/t3_c2)

    #Fit for Parameter Y0
    if E<=0:
        Y02 = 0
    elif 0<E<=40:
        Y02 = -3.318e-4+1.167e-5 * E
    elif 40<E<=50:
        Y02 = -0.00141+3.86e-5 * E
    elif 50<E<=70:
        Y02 = -0.0046+1.0245e-4 * E
    elif 70<E<=100:
        Y02 = -0.01206+2.09e-4 * E
    elif 100<E<=140:
        Y02 = -0.02231+3.115e-4 * E
    elif 140<E<=200:
        Y02 = -0.0256+3.35e-4 * E
    elif 200<E<=300:
        Y02 = -0.019+3.02e-4 * E
    elif 300<E<=500:
        Y02 = 0.005+2.22e-4 * E
    elif 500<E<=1000:
        Y02 = 0.054+1.24e-4 * E
    elif 1000<E<=3000:
        Y02 = 0.1425+3.55e-5 * E
    else:
        Y02 = 0
    
    targetIndex = 1 #index for carbon to access E_s
    
    #calculate self-sputtering yield
    alpha0 = np.pi - np.arccos(np.sqrt(1/(1 + E/E_s[targetIndex])))
    Y_phys = Y02 * (np.cos((alpha * np.pi/(alpha0 * 2))**c2_fit))**(-f_fit) * np.exp(b_fit * (1 - (1/(np.cos((alpha * np.pi/(alpha0 * 2))**c2_fit)))))
    if Y_phys<0:
        Y_phys = 0
    else:
        pass
    
    return Y_phys

#######################################################################################################################################################################
#Calculation of the Chemical Erosion Yield of Graphite under Bombardement with Hydrogen, Deuterium, Tritium

#Chemical erosion yield for low fluxes, provide E in [eV], T_s in [K], flux in [ions/(s m^2)]
def calculateChemicalErosionYield(incidentParticle='H', E=2, T_s=600, flux=1e22):   
    #initialize parameters
     
    c_chem = np.array([1.865, 1.7, 1.535, 1.38, 1.26])                                                                          
    T_s *= k                    #conversion to [eV] 

    if incidentParticle=="H":
        incidentIndex = 0
    if incidentParticle=="D":
        incidentIndex = 1
    if incidentParticle=="T":
        incidentIndex = 2   
    targetIndex = 1             #target is made from graphite, index to access E_TF       

    #calculate Y_chem[i] for selected incident particle (5 subprocesses) and sum them up to get Y_chem
    Y_chem = 0 
    epsilon = E/(E_TF[targetIndex, incidentIndex])                                                                                     
    s_n = (0.5 * np.log(1 + 1.2288 * epsilon))/(epsilon + 0.1728 * np.sqrt(epsilon) + 0.008 * epsilon**(0.1504))
    
    if E<E_thd_chem[incidentIndex]:
        Y_damage = 0
    else:
        Y_damage = Q_y_chem[incidentIndex] * s_n * (1 - ((E_thd_chem[incidentIndex]/E)**(2/3))) * ((1 - (E_thd_chem[incidentIndex]/E))**2)                                

    for i in range(5):                       
        s_chem = (1/(1 + 3 * 1e7 * np.exp(-1.4/T_s))) * ((2 * 1e-32 * flux + np.exp(-c_chem[i]/T_s))/(2 * 1e-32 * flux + (1 + (2 * 1e29 * np.exp(-1.8/T_s))/flux) * np.exp(-c_chem[i]/T_s)))   #T_s in [eV], flux in [ions/(s m^2)]
        Y_therm=(0.0439 * s_chem * np.exp(-c_chem[i]/T_s))/((2 * 1e-32 * flux) + np.exp(-c_chem[i]/T_s))                                                                                       #T_s in [eV], flux in [ions/(s m^2)]
        if E<E_ths_chem[incidentIndex]:
            Y_surf = 0
        else:
            Y_surf = (s_chem * Q_y_chem[incidentIndex] * s_n * (1 - ((E_th_chem[incidentIndex]/E)**(2/3))) * ((1 - (E_th_chem[incidentIndex]/E))**2))/(1 + np.exp((E - 65)/40))                    #E in [eV]
        if i<3:
            Y_chem += (Y_surf + Y_therm * (1 + C_d_chem[incidentIndex] * Y_damage))/4
        else:
            Y_chem += (Y_surf + Y_therm * (1 + C_d_chem[incidentIndex] * Y_damage))/8
    
    return Y_chem

#Chemical erosion yield for low fluxes following Roths Formula, provide E in [eV], T_s in [K], flux in [ions/(s m^2)], k in [eV/K]
def calculateChemicalErosionYieldRoth(incidentParticle='H', E=2, T_s=600, flux=1e22):                                                       
    #index to access correct array entry for e.g. Q_y_chem
    if incidentParticle=="H":
        incidentIndex = 0
    if incidentParticle=="D":
        incidentIndex = 1
    if incidentParticle=="T":
        incidentIndex = 2   
    targetIndex = 1             #target is made from graphite, index to access E_TF       

    if E==0:
        Y_chem = 0
    else:                     
        #calculate physical sputtering as needed to determine ion-induced chem. erosion by bond breaking
        epsilon = E/E_TF[targetIndex, incidentIndex]                                                                                          
        s_n = (0.5 * np.log(1 + 1.2288 * epsilon))/(epsilon + 0.1728 * np.sqrt(epsilon) + 0.008 * epsilon**(0.1504))
        Y_phys = (Q_y_chem[incidentIndex] * s_n) * (1 - (E_th_chem[incidentIndex]/E)**(2/3)) * ((1 - (E_th_chem[incidentIndex]/E))**2)                                          #Y als Funktion von E und dem Winkel alpha=0
        
        if Y_phys<0:
            Y_phys = 0
        else:
            pass
        
        #calculate parameters for chem erosion
        C = 1/(1 + 1e13 * np.exp(-2.45/(k * T_s)))
        c_sp3 = (C * (2 * 1e-32 * flux + np.exp(-1.7/(k * T_s))))/(2 * 1e-32 * flux + ((2 * 1e29/flux) * np.exp(-1.8/(k * T_s))) * np.exp(-1.7/(k * T_s)))
        
        #calculate subprocesses and total chemical erosion yield
        Y_surf = (c_sp3 * (Q_y_chem[incidentIndex] * s_n * (1 - (E_ths_chem[incidentIndex]/E)**(2/3)) * ((1 - (E_ths_chem[incidentIndex]/E))**2)))/(1 + np.exp((E - 90)/50))
        Y_therm = (c_sp3 * (0.033 * np.exp(-1.7/(k * T_s))))/(2 * 1e-32 * flux + np.exp(-1.7/(k * T_s)))
        Y_chem = Y_therm * (1 + C_d_chem[incidentIndex] * Y_phys) + Y_surf
        
    return Y_chem


#Chemical erosion yield for high fluxes, provide flux in [ions/(s m^2)]
def calculateChemicalErosionYieldHighFlux(Y_chem_lowFlux=1e-3, flux=1e22):
    return Y_chem_lowFlux/(1 + (flux/(6 * 1e21))**0.54)

#######################################################################################################################################################################
#Calculation of Total Sputtering Yield for one Type of Incident Ion 

#Energy distribution of ions (thermal energy + acceleration in Langmuir Sheath), E and T_i in [eV], q in [1, 2, 3] for [H, C, O]
def energyDistributionIons(E=2, T_i=1, q=1):
    return 4 * (1/(2 * T_i))**2 * (E - 3 * T_i * q) * np.exp(-(E - 3 * T_i * q)/(T_i))

def calculateTotalErosionYield(incidentParticle, T_i, targetMaterial='C', alpha=np.pi/4, T_s=600, flux=1e22):
    if incidentParticle=="C":
        q_i = 2
    elif incidentParticle=="O":
        q_i = 3
    else:
        q_i = 1
    def Integral(incidentParticle, E, T_i, targetMaterial='C', alpha=np.pi/4, T_s=600, flux=1e22):
        if incidentParticle=="H":
            Y = calculatePhysicalSputteringYield(targetMaterial, 'H', E, alpha) + calculateChemicalErosionYield('H', E, T_s, flux)
        elif incidentParticle=="D":
            Y = calculatePhysicalSputteringYield(targetMaterial, 'D', E, alpha) + calculateChemicalErosionYield('D', E, T_s, flux)
        elif incidentParticle=="T":
            Y = calculatePhysicalSputteringYield(targetMaterial, 'T', E, alpha) + calculateChemicalErosionYield('T', E, T_s, flux)
        elif incidentParticle=="O":
            Y = calculatePhysicalSputteringYield(targetMaterial, 'O', E, alpha) + 0.7
        elif incidentParticle=="C":
            Y = calculatePhysicalSputteringYieldEckstein(E, alpha)   
        return energyDistributionIons(E, T_i, q_i) * Y
    IntegralFunction = lambda E: Integral(incidentParticle, E, T_i, targetMaterial, alpha, T_s, flux)
    IntegralResult = integrate.quad(IntegralFunction, 3 * T_i * q_i, np.inf)

    return IntegralResult[0]

#######################################################################################################################################################################
#Calculation of Fluxes of Incident Ions and Eroded Particles, and Thickness of Eroded Layer 

#Calculation of incident ion flux density from ion velocity
#muss da noch * f mit rein für einzelne Ionenflüsse?
def calculateFluxIncidentIonFromSpeed(c_w, n_e):
    return c_w * n_e  

#Calculation of incident ion flux density from ion velocity, provide temperatures in [K], k_B in [J/K], m_i in [kg], n_e at last closed flux surface in [1/m^3]?
def calculateFluxIncidentIon(T_e, T_i, m_i, n_e, f):
    return np.sqrt(k_B * (T_e + T_i)/m_i) * n_e * f

#Calculation of erosion rate of graphite (redeposition not considered), provide flux in [1/(s*m^2)], n_target in [1/m^3]
#Y and flux must be np.arrays containing sputtering yields and ion fluxes for all relevant erosion processes (phys. sputtering by H, O, C, chem. erosion by H, O)
def calculateErosionRate(Y, flux, n_target):                                                               
    return sum(Y * flux/n_target)

#Calculation of layer thickness of graphite that has been eroded (redeposition not considered), provide flux in [1/(s*m^2)], n_target in [1/m^3], t_discharge in [s]
#Y and flux must be np.arrays containing sputtering yields and ion fluxes for all relevant erosion processes (phys. sputtering by H, O, C, chem. erosion by H, O)
def calculateDeltaErodedLayer(Y, flux, t_discharge, n_target):                                                               
    return sum(Y * flux * t_discharge/n_target)

#Calculation of gross flux of eroded particles (redeposition not considered), provide flux in [1/(s*m^2)], Y and f must be np.arrays for all relevant erosion processes
def calculateFluxErodedParticlesGross(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering):                                                          
    return (flux_electron * np.sum(Y * f))/(1 - P_redeposition * (Y_selfSputtering + 1 - stickingCoefficient))

#Calculation of flux of reposited particles, provide flux in [1/(s*m^2)], Y and f must be np.arrays for all relevant erosion processes
def calculateFluxRedepositedParticles(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering):                                                          
    return (stickingCoefficient * P_redeposition * flux_electron * np.sum(Y * f))/(1 - P_redeposition * (Y_selfSputtering + 1 - stickingCoefficient))

#Calculation of net flux of eroded particles (redeposition considered), provide flux in [1/(s*m^2)], Y and f must be np.arrays for all relevant erosion processes
def calculateFluxErodedParticlesNet(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering):                                                          
    return calculateFluxErodedParticlesGross(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering) - calculateFluxRedepositedParticles(flux_electron, Y, f, P_redeposition, stickingCoefficient, Y_selfSputtering)

#######################################################################################################################################################################
#Calculation of Net Erosion Specifically for Divertor Plates

#Choose applying lambda_n in dependance of y, z, theta
def chooseLambdaN(y, z, theta):
    if y * np.sin(theta) + z * np.cos(theta)<0:
        return lambda_nl
    if y * np.sin(theta) + z * np.cos(theta)>0:
        return lambda_nr
    else:   #dont know what to return
        return 1

#Development of electron density and electron temperature
def developementElectronDensity(n_e_0, y, z, theta, lambda_n):
    lambda_n = chooseLambdaN(y, z, theta)
    return n_e_0 * np.exp(-(y * np.sin(theta) + z * np.cos(theta))/lambda_n)

#Development of electron density and electron temperature
def developementElectronTemperature(T_e_0, y, z, theta, lambda_n):
    lambda_n = chooseLambdaN(y, z, theta)
    return T_e_0 * np.exp(-(y * np.sin(theta) + z * np.cos(theta))/lambda_n)

#Calculation of gross erosion particle flux
def calculateFluxErodedParticlesGrossDivertor():
    pass

#Calculation of deposition particle flux
def calculateFluxRedepositedParticlesDivertor(flux_erodedParticles, y, dy):
    return -flux_erodedParticles(y - dy)

#Calculation of net erosion particle flux
def calculateFluxErodedParticlesNetDivertor(flux_erodedParticles, y, dy):
    return flux_erodedParticles(y) - flux_erodedParticles(y - dy)

#Calculation of an estimate for the layer thickness of eroded material at the strike line region
#Plasma ions exclude carbon impurities!
def calculateDeltaErodedLayerStrikeline(Y_plasmaIons, flux_plasmaIons, t_discharge, n_target, Y_selfsputtering, probability_promptRedeposition, stickingCoefficient, dy, C=3.5): 
    return np.tanh(C * dy/abs(lambda_nl)) * (t_discharge * flux_plasmaIons * Y_plasmaIons * (1 - stickingCoefficient * probability_promptRedeposition))/(n_target * (1 - probability_promptRedeposition * (Y_selfsputtering + 1 - stickingCoefficient)))
