#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:20:32 2025

@author: thir
"""
import numpy as np
import matplotlib.pyplot as plt
from src.archivedb.archivedb.signals import get_signal
from src.archivedb.archivedb.parlogs import get_parameters_box
from src.archivedb.archivedb.programs import get_program_from_to
from src.archivedb.archivedb.versions import get_last_version_for_program

def readImpurityConcentrationFromCXRS(shot):
    # change those for different impurities
    signal_path_C = "ArchiveDB/raw/W7XAnalysis/QSK_n_imp/n_C6_fit_DATASTREAM/"
    parlog_path_C = "ArchiveDB/raw/W7XAnalysis/QSK_n_imp/n_C6_fit_PARLOG/"
    signal_path_O = "ArchiveDB/raw/W7XAnalysis/QSK_n_imp/n_C6_fit_DATASTREAM/"#O8_fit_DATASTREAM/"
    parlog_path_O = "ArchiveDB/raw/W7XAnalysis/QSK_n_imp/n_C6_fit_PARLOG/"#O8_fit_PARLOG/"

    #find the latest version
    version_C = get_last_version_for_program(signal_path_C, shot)
    version_O = get_last_version_for_program(signal_path_O, shot)

    # add this to the paths
    signal_path_C += version_C + "/"
    parlog_path_C += version_C + "/"
    signal_path_O += version_O + "/"
    parlog_path_O += version_O + "/"

    #fetch the fitted profiles
    time_C, values_C = get_signal(signal_path_C, *get_program_from_to(shot), enforceDataType=True, timeout=10)
    time_C = (np.array(time_C) - get_program_from_to(shot)[0]) / 1e9 - 61
    time_O, values_O = get_signal(signal_path_O, *get_program_from_to(shot), enforceDataType=True, timeout=10)
    time_O = (np.array(time_O) - get_program_from_to(shot)[0]) / 1e9 - 61

    # also check the parlog fetching
    parlog_C = get_parameters_box(parlog_path_C, *get_program_from_to(shot), timeout=10)
    parlog_O = get_parameters_box(parlog_path_O, *get_program_from_to(shot), timeout=10)

    # gather the data in a sensible format
    # identify the parlog for the different time instances
    for impurity, parlog_imp, values_imp, time_imp in zip(['C', 'O'], [parlog_C["values"][0], parlog_O["values"][0]], [values_C, values_O], [time_C, time_O]):
        idx = 0
        parlog_handles = []
        for parlog_handle in parlog_imp:
            if not parlog_handle.startswith("t="):
                continue
            if (
                float(parlog_handle.split("=")[1].split("_")[0]) > time_imp[idx]
                or float(parlog_handle.split("=")[1].split("_")[1]) < time_imp[idx]
            ):
                continue
            # the surviving one is the correct parlog
            parlog_handles.append(parlog_handle)
            idx += 1


        # gather the rho locations of the fits
        rho = np.zeros_like(values_imp[:, 0])
        for rho_idx, entry in enumerate(parlog_imp["chanDescs"]):
            rho[rho_idx] = float(parlog_imp["chanDescs"][entry]["name"])

        # and the density values
        data_rho = np.zeros((len(parlog_imp[parlog_handles[0]]["rho"]), len(time_imp)))
        data_values = np.zeros((len(parlog_imp[parlog_handles[0]]["rho"]), len(time_imp)))
        data_errors = np.zeros((len(parlog_imp[parlog_handles[0]]["rho"]), len(time_imp)))
        data_mask = np.zeros((len(parlog_imp[parlog_handles[0]]["rho"]), len(time_imp)), dtype=bool)
        for rho_idx, entry in enumerate(parlog_imp[parlog_handles[0]]["rho"]):
            for t_idx, parlog_handle in enumerate(parlog_handles):
                data_rho[rho_idx, t_idx] = float(parlog_imp[parlog_handle]["rho"][entry])
                data_values[rho_idx, t_idx] = float(parlog_imp[parlog_handle]["values"][entry])
                data_errors[rho_idx, t_idx] = float(parlog_imp[parlog_handle]["errors"][entry])
                # BUG data_mask[rho_idx, t_idx] = float(parlog["values"][0][parlog_handle]["sensible"][entry])
                
        # HACK
        data_mask[...] = True

        # plot the data we fetched - only for a few instances though
        mult = int(np.floor(len(parlog_handles) / 2) - 1)
        for i, color in enumerate(["black", "orangered", "purple"]):
            # profiles
            plt.plot(rho, values_imp[:, i * mult], label=f"t (s) = {time_C[i * mult]:.2f}", color=color)
            # data points
            plt.errorbar(
                data_rho[data_mask[:, i * mult], i * mult],
                data_values[data_mask[:, i * mult], i * mult],
                yerr=data_errors[data_mask[:, i * mult], i * mult],
                ls="",
                marker="x",
                color=color,
            )


        plt.legend()
        plt.xlabel(r"$\rho$")
        if impurity == 'C':
            plt.ylabel(r"n$_C$ in m$^{-3}$")
        elif impurity == 'O':
            plt.ylabel(r"n$_O$ in m$^{-3}$")
        plt.savefig('results/impurities/CXRS_{shot}{impurity}.png'.format(shot=shot, impurity=impurity))
        plt.show()
        plt.close()
