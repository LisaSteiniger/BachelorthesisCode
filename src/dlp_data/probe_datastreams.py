import numpy as np

probesdb = {
    '50201': ["/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/2/CH2", 1],
    '50203': ["/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/3/CH3", 1],
    '50205': ["/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/4/CH4", 1],
    '50207': ["/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/5/CH5", 1],
    '50209': {
        'default': ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/0/CH0", 2],
        '20241125.000': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/0/CH0", 2],
        '20250225.037': ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/0/CH0", 2],  
        '20250226.001': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/0/CH0", 2],
        '20250226.020': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/7/CH7", 2],
        },
    '50211': ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/1/CH1", 2],
    '50211': {
        'default': ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/1/CH1", 2],
        '20250225.037': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/5/CH5", 2],
        '20250226': ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/1/CH1", 2],
        },
    '50218' : ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/4/CH4", 3],
    '50220' : ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/5/CH5", 3],
    '50222': {
        'default': ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/6/CH6", 3],
        '20241204.021': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/1/CH1", 3],
            },
    '50224' : ["/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/7/CH7", 3],

    '50226' : ["/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/2/CH2", 4],
    '50228' : ["/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/3/CH3", 4],
    '50230' : ["/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/4/CH4", 4],
    '50232' : ["/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/5/CH5", 4],

    '50246' : ["/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/0/CH0", 5],
    '50248' : ["/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/1/CH1", 5],
    '50249' : ["/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/2/CH2", 5],
    '50251' : ["/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/3/CH3", 5],

    '51201' : ["/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/6/CH6", 6],
    '51203' : ["/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/7/CH7", 6],
    '51205': {
        'default': ["/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/0/CH0", 6],
        '20250225.025': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/4/CH4", 6],
            },
    '51207': {
        'default': ["/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/1/CH1", 6],
        '20250225.025': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/3/CH3", 6],
            },

    '51209' : ["/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/4/CH4", 7],
    '51211' : ["/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/5/CH5", 7],

    '51218' : ["/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/0/CH0", 8],
    '51220' : ["/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/1/CH1", 8],
    '51222': {
        'default': ["/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/2/CH2", 8],
        '20241204': ["/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/2/CH2", 8],
            },            
    '51224' : ["/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/3/CH3", 8],

    '51226' : ["/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/6/CH6", 9],
    '51228' : ["/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/7/CH7", 9],
    '51230' : ["/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/0/CH0", 9],
    '51232' : ["/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/1/CH1", 9],

    '51246' : ["/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/4/CH4", 10],
    '51248' : ["/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/5/CH5", 10],
    '51249' : ["/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/6/CH6", 10],
    '51251' : ["/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/7/CH7", 10],
        }

probelist_low_iota = np.array(['50201', '50203', '50205', '50207', '50209', '50211', '50218',
                            '50220', '50222', '50224', '50226', '50228', '50230', '50232',
                            '51201', '51203', '51205', '51207', '51209', '51211', '51218',
                            '51220', '51222', '51224', '51226', '51228', '51230', '51232'])

probelist_2h = np.array(['50201', '50203', '50205', '50207', '50209', '50211', 
                        '51201', '51203', '51205', '51207', '51209', '51211'])

probelist_8h_complete = np.array(['50246', '50248', '50249', '50251', 
                         '51246', '51248', '51249', '51251'])
probelist_8h_reduced = np.array(['50249',
                         '51246', '51248', '51249', '51251'])

'''
20250226.035 -> BB6 ref was changed to ADC 8 ch0, but bb7 ref was supposed to be changed 
20250226.042 -> the ADC channels were "fixed", bb6 ref back to its nominal position and bb7 ref to ADC 8 ch0
'''

referenceprobedb = {1: "/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/1/CH1",
                    2: ["/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/7/CH7", "/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/6/CH6"],
                    3: "/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/3/CH3",
                    4: "/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/1/CH1",
                    5: "/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/7/CH7",
                    6: "/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/5/CH5",
                    7: ["/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/3/CH3",  "/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/0/CH0"], # 20250225.038 onwards
                    8: ["/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/7/CH7", "/raw/W7X/ControlStation.72003/ACQ3-1_DATASTREAM/5/CH5"],
                    9: "/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/5/CH5",
                    10: "/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/3/CH3",
                }

inputvoltagedb = {1: "/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/0/CH0",
                    2: "/raw/W7X/ControlStation.72002/ACQ0-1_DATASTREAM/6/CH6",
                    3: "/raw/W7X/ControlStation.72002/ACQ1-1_DATASTREAM/2/CH2",
                    4: "/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/0/CH0",
                    5: "/raw/W7X/ControlStation.72002/ACQ2-1_DATASTREAM/6/CH6",
                    6: "/raw/W7X/ControlStation.72002/ACQ3-1_DATASTREAM/4/CH4",
                    7: "/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/2/CH2",
                    8: "/raw/W7X/ControlStation.72003/ACQ0-1_DATASTREAM/6/CH6",
                    9: "/raw/W7X/ControlStation.72003/ACQ1-1_DATASTREAM/4/CH4",
                    10: "/raw/W7X/ControlStation.72003/ACQ2-1_DATASTREAM/2/CH2",
                }