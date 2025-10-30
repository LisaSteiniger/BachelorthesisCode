'''This file is responsible for all plotting related functions'''

import matplotlib.pyplot as plt
import numpy as np

def plotOverview(n_e, T_e, T_s, Y_0, Y_3, Y_4, erosionRate_dt, erodedLayerThickness, dt, safe):
    ''' plot sputtering yields, erosion rate, total eroded layer thickness, and electron temperature, surface temperature, density over time for one discharge and one langmuir probe
        all input parameters besides "safe" should be provided as 1D lists/arrays representing measurement/calculation values at one position/langmuir probe over the same number of timesteps
        "safe" is a string determining where to save the plot and under which name, e.g. 'results/calculationTables/results_{discharge}.csv'.format(discharge=discharge)'''
    #filter out values for which not all input parameters were measured successfully (they were set to 0, thus Y_0 becomes 0)
    filter = np.array([i != 0 and np.isnan(i)==False for i in Y_0])
    
    #if all values are filtered out, no plot is neccessary
    if sum(filter)==0:
        return 'fail'
    
    fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(10, 12), sharex=True)
    
    #first subplot for measurement values
    ax[0].plot(np.array(dt)[filter], np.array(n_e)[filter], '-x', label='$n_e$')
    ax[0].plot(np.array(dt)[filter], np.array(T_e)[filter], '-x', label='$T_e$')
    ax[0].plot(np.array(dt)[filter], np.array(T_s)[filter], '-x', label='$T_s$')
    
    #second subplot for calculated sputtering yields
    ax[1].plot(np.array(dt)[filter], np.array(Y_0)[filter], '-x', label='Y of hydrogen')
    #ax[1].plot(np.array(dt)[filter], np.array(Y_1)[filter], '-x', label='Y of deuterium')
    #ax[1].plot(np.array(dt)[filter], np.array(Y_2)[filter], '-x', label='Y of tritium')
    ax[1].plot(np.array(dt)[filter], np.array(Y_3)[filter], '-x', label='Y of carbon')
    ax[1].plot(np.array(dt)[filter], np.array(Y_4)[filter], '-x', label='Y of oxygen')
    
    #third subplot for resulting erosion rates and layer thicknesses
    ax[2].plot(np.array(dt)[filter], np.array(erosionRate_dt)[filter] * 1e9, '-x', label=' $\Delta_{ero}/t$')
    ax[2].plot(np.array(dt)[filter], np.array(erodedLayerThickness)[filter] * 1e9, '-x', label=' $\Delta_{ero}$')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    #ax[0].set_xlabel('Time t from start of the discharge (s)')
    #ax[1].set_xlabel('Time t from start of the discharge (s)')
    ax[2].set_xlabel('Time t from start of the discharge (s)')

    ax[0].set_ylabel('Plasma density $n_e$ (x 1e19 1/m$^3$)\nElectron temperature $T_e$ (eV)\nSurface temperature $T_s$ (K)')
    ax[1].set_ylabel('Sputtering yields Y')
    ax[2].set_ylabel('\nErosion rate $\Delta_{ero}/t$ (nm/s)\nTotal eroded layer thickness  $\Delta_{ero}$ (nm)')
    
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    ax[2].grid(True)

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(safe, bbox_inches='tight')
    fig.show()
    plt.close()
