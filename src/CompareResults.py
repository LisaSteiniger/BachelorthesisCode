import pandas as pd
import numpy as np

def compareMarkus(safe, tableOverview):
    #compares results calculated by this program with results calculated from Markus Kandlers Program (modified version)
    dataMarkus = pd.read_csv('{safe}_Markus.csv'.format(safe=safe[:-4]), sep=';')
        
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['{key}_Markus'.format(key=key)] = dataMarkus[key]
    
    for key in dataMarkus.keys():
        if key != 'Unnamed: 0':
            tableOverview['Delta_{key}'.format(key=key)] = np.round((tableOverview[key] - dataMarkus[key])/tableOverview[key], 10)
