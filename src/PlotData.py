'''This file is responsible for all plotting related functions'''

import matplotlib.pyplot as plt
import numpy as np

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

def plotTotalErodedLayerThickness(LP_position: list[int|float], erosion: list[int|float], deposition: list[int|float],
                                  iota: str, divertorUnit: str, configuration: str, campaign: str, T_default: str, 
                                  extrapolated: bool = False, rates: bool =False):
    ''' This function plots the erosion, deposition, and net erosion at different positions
        positions are given by the installation locations of Langmuir probes given in "LP_position"
        -> depending on "iota" the corresponding locations are chosen from that list 
        -> iota 'low' = LPs on TM2h and TM3h (index 0-13), while 'high' = LPs on TM8h (index 14-17), increasing index = increasing distance from pumping gap
        "erosion" and "deposition" are lists containing the accumulated layer thickness that has been eroded/deposited in one campaign by discharges in a certain configuration
        -> they provide values for all 18 LPs of both divertor units
        -> sorted the same way as LP_position, first lower DU low iota, then lower DU high iota, upper DU low and high iota
        -> "campaign" is either 'OP22', 'OP23', '' (both campaigns), "configuration" the configuration being looked at 
        divertor unit is determined by "divertorUnit" = 'lower' or 'upper'
        "T_default" adds information on the treatment of missing surface temperature values (e.g. set to 320K)
        "extrapolated" determines if layer thicknesses are purely from measurement data (False) or have been extrapolated for missing measurement values (True)
        "rates" determines if layer thicknesses or erosion/deposition rates are plotted'''
    if configuration == 'all':
        configuration = 'wholeCampaign'
    
    if extrapolated:
        extrapolated = 'extrapolated'
    else:
        extrapolated = ''

    if rates:
        unitFactor = 1e+9
        y_label = 'erosion/deposition rates in (nm/s)'
        rates = 'Rates'
    else: 
        unitFactor = 1e+3
        y_label = 'total layer thickness (mm)'
        rates = ''

    if iota == 'low':
        LP_startIndex = 0
        LP_stopIndex = 14
        if divertorUnit == 'lower':
            erosion_startIndex = 0
            erosion_stopIndex = 14
        elif divertorUnit == 'upper':
            erosion_startIndex = 18
            erosion_stopIndex = 32
    elif iota == 'high':
        LP_startIndex = 14
        LP_stopIndex = 18
        if divertorUnit == 'lower':
            erosion_startIndex = 14
            erosion_stopIndex = 18
        elif divertorUnit == 'upper':
            erosion_startIndex = 32
            erosion_stopIndex = 36
    
    plt.plot(LP_position[LP_startIndex:LP_stopIndex], (0 - erosion[erosion_startIndex:erosion_stopIndex])*unitFactor, 'r', label='erosion')
    plt.plot(LP_position[LP_startIndex:LP_stopIndex], (deposition[erosion_startIndex:erosion_stopIndex])*unitFactor, 'b', label='deposition')
    plt.plot(LP_position[LP_startIndex:LP_stopIndex], (0 - erosion[erosion_startIndex:erosion_stopIndex] + deposition[erosion_startIndex:erosion_stopIndex])*unitFactor,'k', label='net erosion')
    plt.legend()
    plt.xlabel('distance from pumping gap (m)')
    plt.ylabel(y_label)
    plt.savefig('results/erosionFullCampaign/{campaign}_{Ts}_totalErosion{rates}_{config}_{iota}Iota_{DU}DU{extrapolated}.png'.format(campaign=campaign, Ts=T_default, rates=rates, iota=iota, DU=divertorUnit, config=configuration, extrapolated=extrapolated), bbox_inches='tight')
    plt.show()
    plt.close()