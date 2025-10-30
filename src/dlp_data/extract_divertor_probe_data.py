import re
import numpy as np
import os

xdrive_directory = "//x-drive/Diagnostic-logbooks/QRP-LangmuirProbes/QRP02-Divertor Langmuir Probes/Analysis/OP2/"

"""
use function fetch_xdrive_data to get all the analyzed data for a discharge:
data_lower, data_upper = fetch_xdrive_data(shot)
data_lower, data_upper are dictionaries with probe numbers 0..17 as keys, 
containing data objects for example: probe 1 lower divertor density is stored in data_lower[0].ne
"""

def fetch_xdrive_data(shot):
    """
    shot: acceptable format yyyymmdd.ss or  yyyymmdd.sss or "yyyymmdd.sss"
    """
    exception_lower, exception_upper = [], []#I added this line
    shot = uniform_shot_number(shot)
    data_lower = {}
    data_upper = {}
    probelist = [str(i) for i in np.append(probes_lower, probes_upper)]
    for name in probelist:
        if int(name) > 51000:
            idx = probes_upper.index(int(name))
            try:
                data_upper[idx] = get_parameters_probe(name, shot)
            except Exception as e:
                print(e)
                exception_upper.append(int(name))#I added this line
        else:
            idx = probes_lower.index(int(name))
            try:
                data_lower[idx] = get_parameters_probe(name, shot)
            except Exception as e:
                print(e)
                exception_lower.append(int(name))#I added this line
    return data_lower, data_upper, exception_lower, exception_upper #I added exception

def get_parameters_probe(probename, shot, return_dict = False):
    """
    Returns:
        obj: attribute container, e.g. obj.ne, obj.Te and so on
    """
    params = ['time','ne','Te','Vp','Vf','Js','sTe','sne','sVp','sVf', 'I_SOL']
    labels_list = [
        'time [s]',
        r'$n_e$ [10$^{18}$m$^{-3}$]',
        r'$T_e$ [eV]',
        r'$V_p$ [V]',
        r'$V_f$ [V]',
        r'$J_{sat}$ [A/m$^2$]',
        r'$\sigma_{T_e}$ [eV]',
        r'$\sigma_{n_e}$ [10$^{18}$m$^{-3}$]',
        r'$\sigma_{V_p}$ [V]',
        r'$\sigma_{V_f}$ [V]',
        r'$I_{0}$ [A]',
        
    ]

    A = np.asarray(get_all_data_probe(probename, shot))
    if A.ndim != 2 or A.shape[1] != len(params):
        raise ValueError(f"data shape {A.shape} does not match {len(params)} parameters")

    def split_label(s):
        m = re.match(r'^(.*?)\s*\[(.*)\]\s*$', s)
        return (m.group(1).strip(), m.group(2).strip()) if m else (s.strip(), "")

    symbols, units = {}, {}
    for p, lab in zip(params, labels_list):
        sym, un = split_label(lab)
        symbols[p], units[p] = sym, un
    units['R [m]'] = 'm'
    symbols['R [m]'] = 'R'

    data_dict = {p: A[:, i] for i, p in enumerate(params)}
    if int(probename) > 51000:
        idx = probes_upper.index(int(probename))
    else:
        idx = probes_lower.index(int(probename))
    data_dict['R'] = R_values[idx]
    class _Obj: pass
    obj = _Obj()
    for p, v in data_dict.items():
        setattr(obj, p, v)

    obj.symbols = symbols
    obj.units = units
    obj.R = R_values[idx]
    if return_dict:
        return obj, data_dict
    else:
        return obj

def get_all_data_probe(probename, shot):
    directory = f"{xdrive_directory}/{shot}/"
    file = directory + f"{shot}_probe_{probename}.txt"
    data = np.loadtxt(file, skiprows=1)
    return data

def uniform_shot_number(shot):
    if isinstance(shot, (float, np.floating)):
        shot = f"{shot:.3f}"

    # Case 2: string length 11, insert '0' at position 10
    shot_str = str(shot)
    if len(shot_str) == 11:
        shot_str = shot_str[:10] + "0" + shot_str[10:]
    return shot_str





probes_lower =     [50201, 50203, 50205, 50207, 50209, 50211, 
                    50218, 50220, 50222, 50224, 50226, 50228, 50230, 50232, 
                    50246, 50248, 50249, 50251] 

probes_upper =     [51201, 51203, 51205, 51207, 51209, 51211, 
                    51218, 51220, 51222, 51224, 51226, 51228, 51230, 51232, 
                    51246, 51248, 51249, 51251] 

R_values =  [5.1602, 5.1857, 5.2111, 5.2365, 5.2619, 5.2873, 
            5.3764, 5.4018, 5.4273, 5.4527, 5.4781, 5.5036, 5.529, 5.5545, 
            5.8799, 5.9054, 5.9209, 5.9464]



