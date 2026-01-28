'''This file is responsible for all plotting related functions'''

import matplotlib.pyplot as plt
import numpy as np
import itertools
import src.ProcessData as process
import src.SputteringYieldFunctions as calc

#######################################################################################################################################################################        
def plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, depositionRate_dt, depositedLayerThickness, dt, timesteps, safe):
    ''' plot sputtering yields, erosion rate, total eroded layer thickness, and electron temperature, surface temperature, density over time for one discharge and one langmuir probe
        all input parameters besides "safe" should be provided as 1D lists/arrays representing measurement/calculation values at one position/langmuir probe over the same number of timesteps
        "safe" is a string determining where to save the plot and under which name, e.g. 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)'''
    #corrected time interval ends for erosion 
    t = []
    for i in range(1, len(timesteps) + 1):
        t.append(np.nansum(np.array(timesteps[:i])))

    #filter out values for which not all input parameters were measured successfully (they were set to 0, thus Y_0 becomes 0)
    filter = np.array([i != 0 and np.isnan(i)==False for i in Y_0])
    
    #if all values are filtered out, no plot is neccessary
    if sum(filter)==0:
        return 'fail'
    
    fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(10, 12), sharex=True)
    
    #first subplot for measurement values
    ax[0].plot(np.array(dt)[filter], np.array(n_e)[filter]*1e-18, '-x', label='$n_e$')
    ax[0].plot(np.array(dt)[filter], np.array(T_e)[filter], '-x', label='$T_e$')
    ax[0].plot(np.array(dt)[filter], np.array(T_s)[filter], '-x', label='$T_s$')
    
    #second subplot for calculated sputtering yields
    ax[1].plot(np.array(dt)[filter], np.array(Y_0)[filter], '-x', label='Y of hydrogen')
    #ax[1].plot(np.array(dt)[filter], np.array(Y_1)[filter], '-x', label='Y of deuterium')
    #ax[1].plot(np.array(dt)[filter], np.array(Y_2)[filter], '-x', label='Y of tritium')
    ax[1].plot(np.array(dt)[filter], np.array(Y_3)[filter], '-x', label='Y of carbon')
    ax[1].plot(np.array(dt)[filter], np.array(Y_4)[filter], '-x', label='Y of oxygen')
    
    #third subplot for resulting erosion rates and layer thicknesses
    #changed1################### replace t by dt
    ax[2].plot(np.array(dt)[filter], np.array(erosionRate_dt)[filter] * 1e9, '-x', label=' $\Delta_{ero}/t$')
    ax[2].plot(np.array(t)[filter], np.array(erodedLayerThickness)[filter] * 1e9, '-x', label=' $\Delta_{ero}$')
    ax[2].plot(np.array(dt)[filter], np.array(depositionRate_dt)[filter] * 1e9, '-x', label=' $\Delta_{dep}/t$')
    ax[2].plot(np.array(t)[filter], np.array(depositedLayerThickness)[filter] * 1e9, '-x', label=' $\Delta_{dep}$')
    ax[2].plot(np.array(t)[filter], (np.array(erodedLayerThickness)[filter] - np.array(depositedLayerThickness)[filter]) * 1e9, '-x', label=' $\Delta_{ero, net}$')
    #changed1###################
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    #ax[0].set_xlabel('Time t from start of the discharge (s)')
    #ax[1].set_xlabel('Time t from start of the discharge (s)')
    ax[2].set_xlabel('Time t from start of the discharge (s)')

    ax[0].set_ylabel('Plasma density $n_e$ (x 1e18 1/m$^3$)\nElectron temperature $T_e$ (eV)\nSurface temperature $T_s$ (K)')
    ax[1].set_ylabel('Sputtering yields Y')
    ax[2].set_ylabel('\nErosion rate $\Delta_{ero}/t$ (nm/s)\nTotal eroded layer thickness  $\Delta_{ero}$ (nm)\nDeposition rate $\Delta_{dep}/t$ (nm/s)\nTotal deposited layer thickness  $\Delta_{dep}$ (nm)\nNet erosion layer thickness  $\Delta_{ero, net}$ (nm)\n')
    
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    ax[2].grid(True)

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()

#######################################################################################################################################################################        
def plotTotalErodedLayerThickness(LP_position: list[int|float], erosion: list[int|float|list[int|float]], deposition: list[int|float|list[int|float]],
                                  erosionError: list[int|float], depositionError: list[int|float],
                                  iota: str, configuration: str, campaign: str, T_default: str, 
                                  extrapolated: bool = False, rates: bool|str =False, safe: str =''):
    ''' This function plots the erosion, deposition, and net erosion at different positions
        positions are given by the installation locations of Langmuir probes given in "LP_position"
        -> depending on "iota" the corresponding locations are chosen from that list 
        -> iota 'low' = LPs on TM2h and TM3h (index 0-13), while 'high' = LPs on TM8h (index 14-17), increasing index = increasing distance from pumping gap
        "erosion" and "deposition" are lists containing the accumulated layer thickness that has been eroded/deposited in one campaign by discharges in a certain configuration
        -> they provide values for all 18 LPs of both divertor units
        -> sorted the same way as LP_position, first lower DU low iota, then lower DU high iota, upper DU low and high iota
        -> "campaign" is either 'OP22', 'OP23', '' (both campaigns), "configuration" the configuration being looked at 
        "T_default" adds information on the treatment of missing surface temperature values (e.g. set to 320K)
        "extrapolated" determines if layer thicknesses are purely from measurement data (False) or have been extrapolated for missing measurement values (True)
        "rates" determines if layer thicknesses or erosion/deposition rates or both ('both') are plotted
        "safe" is where to safe the figure'''
    if configuration == 'all':
        configuration = 'wholeCampaign'
    
    if extrapolated:
        extrapolated = 'extrapolated'
    else:
        extrapolated = ''

    if type(rates) == bool:
        if rates:
            unitFactor = [1e+9]
            y_label = [[['erosion/deposition rates lower divertor unit in (nm/s)'], ['erosion/deposition rates upper divertor unit in (nm/s)']]]
            rates = 'Rates'
        else: 
            unitFactor = [1e+3]
            y_label = [[['total layer thickness lower divertor unit in (mm)'], ['total layer thickness upper divertor unit in (mm)']]]
            rates = 'Layers'

        erosion = [erosion]
        erosionError = [erosionError]
        deposition = [deposition]
        depositionError = [depositionError]

    elif rates == 'both':
        rates = ''
        y_label = [[['total layer thickness lower divertor unit in (mm)'], ['total layer thickness upper divertor unit in (mm)']], [['erosion/deposition rates lower divertor unit in (nm/s)'], ['erosion/deposition rates upper divertor unit in (nm/s)']]]
        unitFactor = [1e+3, 1e+9]


    if iota == 'low':
        LP_startIndex = [0]
        LP_stopIndex = [14]
        erosion_startIndex = [[0, 18]]
        erosion_stopIndex = [[14, 32]]
        columns = 1
        #y_label = [y_label]

    elif iota == 'high':
        LP_startIndex = [14]
        LP_stopIndex = [18]
        erosion_startIndex = [[14, 32]]
        erosion_stopIndex = [[18, 36]]
        columns = 1
        #y_label = [y_label]

    else:
        LP_startIndex = [0, 14]
        LP_stopIndex = [14, 18]
        erosion_startIndex = [[0, 14], [18, 32]]
        erosion_stopIndex = [[14, 18], [32, 36]]
        columns = 2
        y_label_help = [[['Low iota: ' + y_label[0][0][0], 'High iota: ' + y_label[0][0][0]], 
                    ['Low iota: layers ' + y_label[0][1][0], 'High iota: ' + y_label[0][1][0]]]]
        
        if rates == '':
            y_label_help.append([['Low iota: ' + y_label[1][0][0], 'High iota: ' + y_label[1][0][0]], 
                    ['Low iota: layers ' + y_label[1][1][0], 'High iota: ' + y_label[1][1][0]]])
            y_label = y_label_help
        else:
            y_label = y_label_help

    fig, ax = plt.subplots(2, columns, layout='constrained', figsize=(12, 10), sharex='col', sharey=True)
    if rates == '':
        ax00 = ax[0][0].twinx()
        ax01 = ax[1][0].twinx()
        if columns == 2:
            ax02 = ax[0][1].twinx()
            ax03 = ax[1][1].twinx()

    
    for i in range(2):
        for j in range(columns):
            ax[i][j].axhline(0, color='grey')
            ax[i][j].errorbar(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], yerr=erosionError[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]]*unitFactor[0], fmt='r-', label='erosion')
#            ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], 'r', label='erosion')
            ax[i][j].errorbar(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (deposition[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], yerr=depositionError[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]]*unitFactor[0], fmt='b-', label='deposition')
 #           ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (deposition[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], 'b', label='deposition')
            ax[i][j].errorbar(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + deposition[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], yerr=(erosionError[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionError[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], fmt='k-', label='net erosion')
  #          ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + deposition[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0],'k', label='net erosion')
            ax[i][j].legend()
            ax[i][j].set_ylabel(y_label[0][i][j])
            #ax[i][j].tick_params(axis="y", labelcolor='black')

            if rates == '':
                if i == 0 and j == 0:
                    ax1 = ax00
                elif i == 0 and j == 1:
                    ax1 = ax02
                elif i == 1 and j == 0:
                    ax1 = ax01
                elif i == 1 and j == 1:
                    ax1 = ax03

                ax1.errorbar(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[1], 'r:', label='erosion')
                ax1.errorbar(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (deposition[1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[1], 'b:', label='deposition')
                ax1.plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + deposition[1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[1],'k:', label='net erosion')
                ax1.legend()
                ax1.set_ylabel(y_label[1][i][j])
                #ax1.tick_params(axis="y", labelcolor='black')
                

            ax[1][j].set_xlabel('Abstand vom Pumpspalt in (m)')

    if rates == '':
        ax00.sharey(ax01)
        if columns == 2:
            ax02.sharey(ax03)
            min, max = ax00.get_ylim()
            ax02.set_ylim(min, max)
        fig.autofmt_xdate()

    if safe == '':
        safe = 'results/erosionFullCampaign/{campaign}_{Ts}_totalErosion{rates}_{config}{iota}{extrapolated}.png'.format(campaign=campaign, Ts=T_default, rates=rates, iota='_'+iota+'Iota_', config=configuration, extrapolated=extrapolated)
    fig.savefig(safe, bbox_inches='tight')
    plt.show()
    plt.close()

#######################################################################################################################################################################        
#######################################################################################################################################################################        
def plotComparisonErodedLayerThickness(LP_position: list[int|float], erosionList: list[list[int|float|list[int|float]]], depositionList: list[list[int|float|list[int|float]]],
                                  erosionErrorList: list[list[int|float]], depositionErrorList: list[list[int|float]],
                                  preferred: int, configuration: str, campaign: str, T_default: str, 
                                  extrapolated: bool = False, rates: bool =False, safe: str =''):
    ''' This function plots the erosion, deposition, and net erosion at different positions
        positions are given by the installation locations of Langmuir probes given in "LP_position"
        -> depending on "iota" the corresponding locations are chosen from that list 
        -> iota 'low' = LPs on TM2h and TM3h (index 0-13), while 'high' = LPs on TM8h (index 14-17), increasing index = increasing distance from pumping gap
        "erosion" and "deposition" are lists containing the accumulated layer thickness that has been eroded/deposited in one campaign by discharges in a certain configuration
        -> they provide values for all 18 LPs of both divertor units
        -> sorted the same way as LP_position, first lower DU low iota, then lower DU high iota, upper DU low and high iota
        -> "campaign" is either 'OP22', 'OP23', '' (both campaigns), "configuration" the configuration being looked at 
        "T_default" adds information on the treatment of missing surface temperature values (e.g. set to 320K)
        "extrapolated" determines if layer thicknesses are purely from measurement data (False) or have been extrapolated for missing measurement values (True)
        "rates" determines if layer thicknesses or erosion/deposition rates or both ('both') are plotted
        "safe" is where to safe the figure'''
    if configuration == 'all':
        configuration = 'wholeCampaign'
    
    if extrapolated:
        extrapolated = 'extrapolated'
    else:
        extrapolated = ''

    if rates:
        unitFactor = [1e+9]
        y_label = 'Erosions-/Deponierungsrate in (nm/s)'
        rates = 'Rates'
    else: 
        unitFactor = [1e+3]
        y_label = 'Nettoerodierte Schichtdicke in (mm)'
        rates = 'Layers'

    _label = [['Low-iota, obere DU', 'High-iota, obere DU'], ['Low-iota, untere DU', 'High-iota untere DU']]
    LP_startIndex = [0, 14]
    LP_stopIndex = [14, 18]
    erosion_startIndex = [[18, 32], [0, 14]]
    erosion_stopIndex = [[32, 36], [14, 18]]
    columns = 2
    fig, ax = plt.subplots(2, columns, layout='constrained', figsize=(12, 10), sharex='col', sharey=True)


    for i in range(2):
        for j in range(columns):
            ax[i][j].axhline(0, color='grey')
            ax[i][j].errorbar(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosionList[preferred][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[preferred][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], yerr=(erosionErrorList[preferred][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionErrorList[preferred][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], fmt='k.-', label=_label[i][j], capsize=4)
            ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosionList[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], color='grey')
            ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], color='grey')
            #ax[i][j].fill_between(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], alpha=0.2)
            #ax[i][j].fill_between(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], alpha=0.2)
            ax[i][j].fill_between(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosionList[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[0][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], (0 - erosionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + depositionList[-1][erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor[0], alpha=0.2, color='grey')

            ax[i][j].legend()
            ax[i][j].set_ylabel(y_label)

            ax[1][j].set_xlabel('Abstand vom Pumpspalt in (m)')

    if safe == '':
        safe = 'results/erosionFullCampaign/{campaign}_{Ts}_compareErosion{rates}_{config}{extrapolated}.png'.format(campaign=campaign, Ts=T_default, rates=rates, config=configuration, extrapolated=extrapolated)
    fig.savefig(safe, bbox_inches='tight')
    plt.show()
    plt.close()

#######################################################################################################################################################################        
def appendY(Y_H: list[int|float], Y_C: list[int|float], Y_O: list[int|float], erosionRate: list[int|float], depositionRate: list[int|float], 
            Te: int|float, Ts: int|float, ne: int|float, timestep: int|float, alpha: int|float, zeta: int|float, 
            m_i: list[int|float], f_i: list[int|float], ions: list[str], k: int|float, n_target: int|float) -> list[list[int|float]]:
    
    resultParams = process.calculateErosionRelatedQuantitiesOnePosition(np.array([Te]), np.array([Te]), np.array([Ts]), np.array([ne]), timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    Y_0, Y_1, Y_2, Y_3, Y_4, erosionRate_dt, erodedLayerThickness_dt, erodedLayerThickness2, depositionRate_dt, depositedLayerThickness_dt, depositedLayerThickness = resultParams
    
    Y_H.append(Y_0)
    Y_C.append(Y_3)
    Y_O.append(Y_4)
    erosionRate.append(erosionRate_dt)
    depositionRate.append(depositionRate_dt)
    
    return Y_H, Y_C, Y_O, erosionRate, depositionRate
#######################################################################################################################################################################        

def averageValues(varyingQuantity: str, neList: list[int|float], TeList: list[int|float], TsList: list[int|float], alphaList: list[int|float], zetaList: list[int|float], 
                  neIndex: int =2, TeIndex: int =2, TsIndex: int =2, alphaIndex: int =1, zetaIndex: int =1):
    ''' returns some average value for each parameter that is not varied'''
    ne, Te, Ts, alpha, zeta = np.nan, np.nan, np.nan, np.nan, np.nan
    if varyingQuantity != 'ne':
        #ne = neList[neIndex]
        ne = 1e+19
    if varyingQuantity != 'Te':
        #Te = TeList[TeIndex]
        Te = 15
    if varyingQuantity != 'Ts':
        #Ts = TsList[TsIndex]
        Ts = 320
    if varyingQuantity != 'alpha':
        #alpha = alphaList[alphaIndex]
        alpha = 40 * np.pi/180
    if varyingQuantity != 'zeta':
        #zeta = zetaList[zetaIndex]
        zeta = 2 * np.pi/180

    return ne, Te, Ts, alpha, zeta

#######################################################################################################################################################################        
def plotSputteringYieldsInDependence(ne: int|float, Te: int|float, Ts: int|float, alpha: int|float, zeta: int|float, 
                                     m_i: list[int|float], f_i: list[int|float], ions: list[str], k: int|float, n_target: int|float,
                                     safe: str ='results/parameterStudies/sputteringYields.png') -> None:
    
    Y_chem = []
    Y_phys = []

    for m, f, ion in zip(m_i, f_i, ions):
        if ion == 'C':
            q = 2
        elif ion == 'O':
            q = 3
        else:
            q = 1
        energies = np.linspace(3 * Te * q, 1e+5, 1000)
        flux = calc.calculateFluxIncidentIon(zeta, Te, Te, m, ne, f)
        
        Y_ion_chem = []
        Y_ion_phys = []
        for E in energies:
            if ion == 'C':
                Y_ion_chem.append(0)
                Y_ion_phys.append(calc.calculatePhysicalSputteringYieldEckstein(E, alpha))
            elif ion == 'O':
                Y_ion_chem.append(0.7)
                Y_ion_phys.append(calc.calculatePhysicalSputteringYield('C', ion, E, alpha, n_target))
            else:
                Y_ion_chem.append(calc.calculateChemicalErosionYieldRoth(ion, E, Ts, flux))
                Y_ion_phys.append(calc.calculatePhysicalSputteringYield('C', ion, E, alpha, n_target))
            

        Y_chem.append(Y_ion_chem)
        Y_phys.append(Y_ion_phys)

    Y = [np.array(Y_phys), np.array(Y_chem), np.array(Y_phys) + np.array(Y_chem)]
    fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(12, 15), sharex=True)
    for i, Y_type in enumerate(Y):
        for ion, Y_ion in zip(ions, Y_type):
            ax[i].plot(energies, Y_ion, label=ion)
    
    ax[0].set_ylabel('Physical sputtering yield')
    ax[1].set_ylabel('Chemical sputtering yield')
    ax[2].set_ylabel('Total sputtering yield')

    for i in range(3):
        ax[i].set_xlabel('Energy in eV')
        ax[i].legend()
        ax[i].grid(True, which="both")
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
    
    plt.figtext(0.5, 1.01, '$n_e$ = {ne} 1/m$^3$, $T_e$ = {Te} eV, $T_s$ = {Ts} K, $\\alpha$ = {alpha:.3f} rad, $\zeta$ = {zeta:.3f} rad\n$f_H$ = {H}, $f_C$ = {C}, $f_O$ = {O}'.format(ne=ne, Te=Te, Ts=Ts, alpha=alpha, zeta=zeta, H=f_i[0], C=f_i[3], O=f_i[4]), horizontalalignment='center')
    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()

#######################################################################################################################################################################        
def plotEnergyDistribution(Te: int|float, ions: list[str],
                           safe: str ='results/parameterStudies/energyDistribution.png') -> None:
    fig, ax = plt.subplots(1, 1, layout='constrained', sharex=True)

    for ion in ions:
        if ion == 'C':
            q = 2
        elif ion == 'O':
            q = 3
        else:
            q = 1
        energies = np.linspace(3 * Te * q, 1e+5, 1000)
        energyDist = [calc.energyDistributionIons(x, Te, q) for x in energies]        
        
        import scipy.integrate as integrate
        IntegralFunction = lambda E: calc.energyDistributionIons(E, Te, q)
        IntegralResult = integrate.quad(IntegralFunction, 3 * Te * q, np.inf)
        print(IntegralResult[0])
        ax.plot(energies, energyDist, label=f'{ion}')
    
        ax.set_ylabel('Wahrscheinlichkeit')
        ax.set_xlabel('Energie in (eV)')
        ax.legend()
        ax.grid('minor')
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    plt.figtext(0.5, 1.01, '$T_e$ = {Te} eV'.format(Te=Te), horizontalalignment='center')
    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()

#######################################################################################################################################################################        
def parameterStudy(neList: list[int|float], TeList: list[int|float], TsList: list[int|float], alphaList: list[int|float], zetaList: list[int|float], 
                   m_i: list[int|float], f_i: list[int|float], ions: list[str], k: int|float, n_target: int|float,
                   timestep: list[int|float] =[50000], safe: str ='results/parameterStudies/overviewPlot.png') -> None:
    ''' parameter study for varying ne, Te, Ts, alpha and zeta returning sputtering yields and rates'''
    fig, ax = plt.subplots(5, 2, layout='constrained', figsize=(12, 15))

    #varying zeta
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('zeta', neList, TeList, TsList, alphaList, zetaList)    
    ne, Te, Ts, alpha = averages[0:-1]
    for zeta in zetaList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[4][0].plot(list(map(np.rad2deg, zetaList)), list(itertools.chain.from_iterable(Y_H)), label='Wasserstoff')
    ax[4][0].plot(list(map(np.rad2deg, zetaList)), list(itertools.chain.from_iterable(Y_C)), label='Kohlenstoff')
    ax[4][0].plot(list(map(np.rad2deg, zetaList)), list(itertools.chain.from_iterable(Y_O)), label='Sauerstoff')
    ax[4][1].plot(list(map(np.rad2deg, zetaList)), 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[4][1].plot(list(map(np.rad2deg, zetaList)), np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deponierung')

    for i in range(2):
        ax[4][i].set_xlabel('Einfallswinkel der Magnetfeldlinien $\zeta$ in (°)')

    #varying alpha
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('alpha', neList, TeList, TsList, alphaList, zetaList)    
    ne, Te, Ts = averages[0:-2]
    zeta = averages[-1]
    for alpha in alphaList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[3][0].plot(list(map(np.rad2deg, alphaList)), list(itertools.chain.from_iterable(Y_H)), label='Wasserstoff')
    ax[3][0].plot(list(map(np.rad2deg, alphaList)), list(itertools.chain.from_iterable(Y_C)), label='Kohlenstoff')
    ax[3][0].plot(list(map(np.rad2deg, alphaList)), list(itertools.chain.from_iterable(Y_O)), label='Sauerstoff')
    ax[3][1].plot(list(map(np.rad2deg, alphaList)), 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[3][1].plot(list(map(np.rad2deg, alphaList)), np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deponierung')

    for i in range(2):
        ax[3][i].set_xlabel('Ioneneinfallswinkel $\\alpha$ in (°)')

    #varying ne
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('ne', neList, TeList, TsList, alphaList, zetaList)    
    Te, Ts, alpha, zeta = averages[1:]
    for ne in neList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[0][0].plot(neList, list(itertools.chain.from_iterable(Y_H)), label='Wasserstoff')
    ax[0][0].plot(neList, list(itertools.chain.from_iterable(Y_C)), label='Kohlenstoff')
    ax[0][0].plot(neList, list(itertools.chain.from_iterable(Y_O)), label='Sauerstoff')
    ax[0][1].plot(neList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[0][1].plot(neList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deponierung')

    for i in range(2):
        ax[0][i].set_xlabel('Elektronendichte $n_e$ in (m$^{-3}$)')

    #varying Te
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('Te', neList, TeList, TsList, alphaList, zetaList)    
    ne = averages[0]
    Ts, alpha, zeta = averages[2:]
    for Te in TeList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[1][0].plot(TeList, list(itertools.chain.from_iterable(Y_H)), label='Wasserstoff')
    ax[1][0].plot(TeList, list(itertools.chain.from_iterable(Y_C)), label='Kohlenstoff')
    ax[1][0].plot(TeList, list(itertools.chain.from_iterable(Y_O)), label='Sauerstoff')
    ax[1][1].plot(TeList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[1][1].plot(TeList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deponierung')

    for i in range(2):
        ax[1][i].set_xlabel('Elektronentemperatur $T_e$ in (eV)')

    #varying Ts
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('Ts', neList, TeList, TsList, alphaList, zetaList)    
    ne, Te = averages[0:2]
    alpha, zeta = averages[3:]
    for Ts in TsList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[2][0].plot(TsList, list(itertools.chain.from_iterable(Y_H)), label='Wasserstoff')
    #ax[2][0].plot(TsList, list(itertools.chain.from_iterable(Y_C)), label='Kohlenstoff')
    #ax[2][0].plot(TsList, list(itertools.chain.from_iterable(Y_O)), label='Sauerstoff')
    ax[2][1].plot(TsList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[2][1].plot(TsList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deponierung')

    for i in range(2):
        ax[2][i].set_xlabel('Oberflächentemperatur $T_s$ in (K)')

    for i in range(5):
        ax[i][0].set_yscale('log')
        ax[i][0].set_ylabel('Zerstäubungsausbeute Y')
        ax[i][1].set_ylabel('Erosions-/Deponierungsrate in (nm/s)')
        ax[i][0].legend()
        ax[i][1].legend()
    
    ne, Te, Ts, alpha, zeta = averageValues('', neList, TeList, TsList, alphaList, zetaList)
    plt.figtext(0.5, 1.01, 'If not varied: $n_e$ = {ne} 1/m$^3$, $T_e$ = {Te} eV, $T_s$ = {Ts} K, $\\alpha$ = {alpha:.3f} rad, $\zeta$ = {zeta:.3f} rad\n$f_H$ = {H}, $f_C$ = {C}, $f_O$ = {O}'.format(ne=ne, Te=Te, Ts=Ts, alpha=alpha, zeta=zeta, H=f_i[0], C=f_i[3], O=f_i[4]), horizontalalignment='center')
    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()