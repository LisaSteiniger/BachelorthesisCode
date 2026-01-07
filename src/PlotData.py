'''This file is responsible for all plotting related functions'''

import matplotlib.pyplot as plt
import numpy as np
import itertools
import src.ProcessData as process

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
                                  iota: str, configuration: str, campaign: str, T_default: str, 
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
        "rates" determines if layer thicknesses or erosion/deposition rates are plotted
        "safe" is where to safe the figure'''
    if configuration == 'all':
        configuration = 'wholeCampaign'
    
    if extrapolated:
        extrapolated = 'extrapolated'
    else:
        extrapolated = ''

    if rates:
        unitFactor = 1e+9
        y_label = ['erosion/deposition rates lower divertor unit in (nm/s)', 'erosion/deposition rates upper divertor unit in (nm/s)']
        rates = 'Rates'
    else: 
        unitFactor = 1e+3
        y_label = ['total layer thickness lower divertor unit in (mm)', 'total layer thickness upper divertor unit in (mm)']
        rates = ''

    if iota == 'low':
        LP_startIndex = [0]
        LP_stopIndex = [14]
        erosion_startIndex = [[0, 18]]
        erosion_stopIndex = [[14, 32]]
        columns = 1
        y_label = [y_label]

    elif iota == 'high':
        LP_startIndex = [14]
        LP_stopIndex = [18]
        erosion_startIndex = [[14, 32]]
        erosion_stopIndex = [[18, 36]]
        columns = 1
        y_label = [y_label]

    else:
        LP_startIndex = [0, 14]
        LP_stopIndex = [14, 18]
        erosion_startIndex = [[0, 14], [18, 32]]
        erosion_stopIndex = [[14, 18], [32, 36]]
        columns = 2
        y_label = [['Low iota: ' + y_label[0], 'High iota: ' + y_label[0]], ['Low iota: ' + y_label[1], 'High iota: ' + y_label[1]]]

    fig, ax = plt.subplots(2, columns, layout='constrained', figsize=(12, 10), sharex='col', sharey='row')
    
    for i in range(2):
        for j in range(columns):
            ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor, 'r', label='erosion')
            ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (deposition[erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor, 'b', label='deposition')
            ax[i][j].plot(LP_position[LP_startIndex[j]:LP_stopIndex[j]], (0 - erosion[erosion_startIndex[i][j]:erosion_stopIndex[i][j]] + deposition[erosion_startIndex[i][j]:erosion_stopIndex[i][j]])*unitFactor,'k', label='net erosion')
            ax[i][j].legend()
            ax[i][j].axhline(0, color='grey')
            ax[i][j].set_ylabel(y_label[i][j])

            ax[1][j].set_xlabel('distance from pumping gap (m)')
    if safe == '':
        safe = 'results/erosionFullCampaign/{campaign}_{Ts}_totalErosion{rates}_{config}{iota}{extrapolated}.png'.format(campaign=campaign, Ts=T_default, rates=rates, iota='_'+iota+'Iota_', config=configuration, extrapolated=extrapolated)
    fig.savefig(safe, bbox_inches='tight')
    plt.show()
    plt.close()

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
    ax[4][0].plot(zetaList, list(itertools.chain.from_iterable(Y_H)), label='Hydrogen')
    ax[4][0].plot(zetaList, list(itertools.chain.from_iterable(Y_C)), label='Carbon')
    ax[4][0].plot(zetaList, list(itertools.chain.from_iterable(Y_O)), label='Oxygen')
    ax[4][1].plot(zetaList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[4][1].plot(zetaList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deposition')

    for i in range(2):
        ax[4][i].set_xlabel('Incident angle of magnetic field lines $\zeta$ in (rad)')

    #varying alpha
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('alpha', neList, TeList, TsList, alphaList, zetaList)    
    ne, Te, Ts = averages[0:-2]
    zeta = averages[-1]
    for alpha in alphaList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[3][0].plot(alphaList, list(itertools.chain.from_iterable(Y_H)), label='Hydrogen')
    ax[3][0].plot(alphaList, list(itertools.chain.from_iterable(Y_C)), label='Carbon')
    ax[3][0].plot(alphaList, list(itertools.chain.from_iterable(Y_O)), label='Oxygen')
    ax[3][1].plot(alphaList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[3][1].plot(alphaList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deposition')

    for i in range(2):
        ax[3][i].set_xlabel('Ion incident angle $\\alpha$ in (rad)')

    #varying ne
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('ne', neList, TeList, TsList, alphaList, zetaList)    
    Te, Ts, alpha, zeta = averages[1:]
    for ne in neList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[0][0].plot(neList, list(itertools.chain.from_iterable(Y_H)), label='Hydrogen')
    ax[0][0].plot(neList, list(itertools.chain.from_iterable(Y_C)), label='Carbon')
    ax[0][0].plot(neList, list(itertools.chain.from_iterable(Y_O)), label='Oxygen')
    ax[0][1].plot(neList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[0][1].plot(neList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deposition')

    for i in range(2):
        ax[0][i].set_xlabel('Electron density $n_e$ in (m$^{-3}$)')

    #varying Te
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('Te', neList, TeList, TsList, alphaList, zetaList)    
    ne = averages[0]
    Ts, alpha, zeta = averages[2:]
    for Te in TeList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[1][0].plot(TeList, list(itertools.chain.from_iterable(Y_H)), label='Hydrogen')
    ax[1][0].plot(TeList, list(itertools.chain.from_iterable(Y_C)), label='Carbon')
    ax[1][0].plot(TeList, list(itertools.chain.from_iterable(Y_O)), label='Oxygen')
    ax[1][1].plot(TeList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[1][1].plot(TeList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deposition')

    for i in range(2):
        ax[1][i].set_xlabel('Electron temperature $T_e$ in (eV)')

    #varying Ts
    Y_H, Y_C, Y_O, erosionRate, depositionRate = [], [], [], [], []
    averages = averageValues('Ts', neList, TeList, TsList, alphaList, zetaList)    
    ne, Te = averages[0:2]
    alpha, zeta = averages[3:]
    for Ts in TsList:
        Y_H, Y_C, Y_O, erosionRate, depositionRate = appendY(Y_H, Y_C, Y_O, erosionRate, depositionRate, Te, Ts, ne, timestep, alpha, zeta, m_i, f_i, ions, k, n_target)
    ax[2][0].plot(TsList, list(itertools.chain.from_iterable(Y_H)), label='Hydrogen')
    ax[2][0].plot(TsList, list(itertools.chain.from_iterable(Y_C)), label='Carbon')
    ax[2][0].plot(TsList, list(itertools.chain.from_iterable(Y_O)), label='Oxygen')
    ax[2][1].plot(TsList, 0 - np.array(list(itertools.chain.from_iterable(erosionRate)))*1e+9, label='Erosion')
    ax[2][1].plot(TsList, np.array(list(itertools.chain.from_iterable(depositionRate)))*1e+9, label='Deposition')

    for i in range(2):
        ax[2][i].set_xlabel('Surface temperature $T_s$ in (K)')

    for i in range(5):
        ax[i][0].set_yscale('log')
        ax[i][0].set_ylabel('Sputtering yield Y')
        ax[i][1].set_ylabel('Erosion/Deposition rate in (nm/s)')
        ax[i][0].legend()
        ax[i][1].legend()
    
    ne, Te, Ts, alpha, zeta = averageValues('', neList, TeList, TsList, alphaList, zetaList)
    plt.figtext(0.5, 1.01, 'If not varied: $n_e$ = {ne} 1/m$^3$, $T_e$ = {Te} eV, $T_s$ = {Ts} K, $\\alpha$ = {alpha:.3f} rad, $\zeta$ = {zeta:.3f} rad\n$f_H$ = {H}, $f_C$ = {C}, $f_O$ = {O}'.format(ne=ne, Te=Te, Ts=Ts, alpha=alpha, zeta=zeta, H=f_i[0], C=f_i[3], O=f_i[4]), horizontalalignment='center')
    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()