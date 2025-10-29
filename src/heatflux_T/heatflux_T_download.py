# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:27:55 2021

@author: ygao
"""
import numpy as np
import time
import sys
import scipy.signal
import multiprocess
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.fftpack import rfft, irfft, fftfreq

import w7xarchive
from src.heatflux_T.heatflux_T_plotTools import plot_Frame, plot_Profile, plot_Finger
from src.heatflux_T.shutter_closed_PID import shutterclosed
import warnings
warnings.filterwarnings("ignore")


use_cache = True
protocol = 'cbor'
# if download frames too slow, put protocol to 'json'
# use_cache may create large files locally due to the new feathure of w7xarchive, put False.


portlistdict = {
    'OP1.2' : ['AEF10', 'AEF11', 'AEF20', 'AEF21', 'AEF30', 'AEF31', 'AEF40', 'AEF41', 'AEF51'], 
    'OP2' :   ['AEK30', 'AEK31', 'AEA30', 'AEA31', 'AEF10', 'AEF11', 'AEF20', 'AEF21', \
               'AEF40', 'AEF41', 'AEF50', 'AEF51']
    }

def get_signal_ts_te(url, interval):
    
    time1, data1 = w7xarchive.get_signal(url, interval[0], interval[1], \
            timeout = 300, retries = 5, protocol = protocol);
    return [time1, data1]
            
class cal_constants (object):
    def __init__(self, PID = '20180920.009', group = 'QRT_IRCAM_new', \
                 project = 'W7XAnalysis', view = 'raw', database = 'Test'):
        '''
        
        To initialize all the constants required for heat flux or T processes, 
        by downloading from the archive.
        Notice that "use_cache" is used for fast reloading, see details in w7xarchive.
        
        Parameters
        ----------
        PID : string, optional
            program number. The default is '20180920.009'.
        group : string, optional
            group in the archive. The default is 'QRT_IRCAM_new'.
        project : string, optional
            project in the archive. The default is 'W7XAnalysis'.
        view : string, optional
            view in the archive. The default is 'raw'.
        database : string, optional
            database in the archive. The default is 'Test'.

        Returns
        -------
        None.

        '''
        
        stream = 'MergedCoor'                   
        values = w7xarchive.get_parameters_box_for_program(\
        database + '/' + view + '/' + project + '/' + group + '/' + stream, PID, use_last_version = True, \
                use_cache = use_cache)['values'][0]
        data_timestamp = values['data_timestamp']
        coornames = np.array([v['name'] for k, v in values['chanDescs'].items()])
        self._stacknames = np.array([k for k,v in values['AreaStacks'].items()])
        self._areastacks = np.array([v for k,v in values['AreaStacks'].items()])
        self._TMnames = np.array([k for k,v in values['TMcontains'].items()])
        TMcontainstemp = np.array([v for k,v in values['TMcontains'].items()])
        TMcontains = []
        for i in TMcontainstemp:
            TMcontains.append(np.array([int(x) for x in i.split(' ')]))
        self._TMcontains = np.array(TMcontains, dtype = 'object')
        
        self._MergedProfileIndices = np.array([int(i) for i in values['MergedProfileIndices'].split(' ')])
        coordatas = w7xarchive.get_signal([database, view, project, group, stream], \
            data_timestamp, data_timestamp, use_last_version = True, \
                use_cache = use_cache)[1]
        for coorname, coordata in list(zip(coornames, coordatas)):
            setattr(self, '_' + coorname, coordata)
        
        # MergedlineNO, finger coordinate and S are never going to be used in a merged format
        # so split them already here.
        
        MergedlineNO = np.array([int(i) for i in values['MergedlineNO'].split(' ')])
        indices = np.where(MergedlineNO == 0)[0]
        stacklineNOs = []
        for i in range(len(indices) - 1):
            stacklineNOs.append(MergedlineNO[indices[i]:indices[i+1]])
        stacklineNOs.append(MergedlineNO[indices[-1]:])
        self._stacklineNOs = np.array(stacklineNOs, dtype = 'object')  
        
        self._stackrowcounts, self._stackS = \
            splitMergedlocationalltoStacks(self._S, self._MergedProfileIndices)
        self._stackft1 = splitMergedlocationalltoStacks(self._ft1, self._MergedProfileIndices)[1]
        self._stackft2 = splitMergedlocationalltoStacks(self._ft2, self._MergedProfileIndices)[1]
        del self._S, self._ft1, self._ft2
            
class heatflux_T_process (cal_constants):
    
    def __init__(self, PID = '20180920.009', port = 'AEF10', group = 'QRT_IRCAM_new', \
                 project = 'W7XAnalysis', view = 'raw', database = 'Test'):
        '''
        

        Parameters
        ----------
        PID : string, optional
            program number. The default is '20180920.009'.
        port: string, optional
            port name. The default is 'AEF10'.
        group : string, optional
            group in the archive. The default is 'QRT_IRCAM_new'.
        project : string, optional
            project in the archive. The default is 'W7XAnalysis'.
        view : string, optional
            view in the archive. The default is 'raw'.
        database : string, optional
            database in the archive. The default is 'Test'.

        Returns
        -------
        None.

        '''
        super().__init__(PID, group, project, view, database)
        if 'W7X' + PID in shutterclosed.keys():
            if port in shutterclosed['W7X' + PID]:
                sys.exit("shutter closed")
        self.PID = PID
        self.port = port

        self._archiveComp = [database, view, project, group]  
        
        self._strikelinewidth_TMs = None
        self._wetted_area_TMs = None
        self._maxHF_TMs = None
        self._meanT_TMs = None
        self._maxT_TMs = None
        
        self._strikelinewidth_stacks = None
        self._wetted_area_stacks = None
        self._maxHF_stacks = None
        self._meanT_stacks = None
        self._maxT_stacks = None
        self._meanHF_stacks = None
        self._integralpower_stacks = None
        
        self._strikelinewidth_TMs_v2 = None
        self._wetted_area_TMs_v2 = None
        self._maxHF_TMs_v2 = None
        self._meanHF_TMs_v2 = None
        self._integralpower_TMs_v2 = None        
        self._loc_TMs_v2 = None
        
        self._strikelinewidth_stacks_v2 = None
        self._wetted_area_stacks_v2 = None
        self._maxHF_stacks_v2 = None
        self._meanHF_stacks_v2 = None
        self._integralpower_stacks_v2 = None
        self._loc_stacks_v2 = None
        
        # _meanHF_TMs and _integralpower_TMs as well as availtimes got here already.
        self.__get_meanHF_TMs()
               
    def get_Frames (self, ts, te, T = False, average = True, cpu_count = 4):
        
        '''
        Get temperature or heat flux frames given the start and end time.
        Notice that the resulting attributes self._datas is protected, because 
        they are merged data directly from the archive with difficult ordering.
        It is recommended to use get_TMs to get the sorted and/or processed data.
        
        parallel reading implemented.

        Parameters
        ----------
        ts : float
            start time relative to t1 in second.
        te : float
            end time relative to t1 in second.
        T : boolean, optional
            to get temperature or heat flux. The default is False.
        average : boolean, optional
            if True, to average the data between ts or te. The default is True.
        cpu_count : integer, optinal
            how many cpu to be used for downloading in parallel. The default is 4.
            
        Returns
        -------
        None.
        
        '''              
            
        ts = max(0, ts)
        te = min(self.availtimes[-1], te)
        tsstamp = self.t1 + int(ts * 1e9)
        testamp = self.t1 + int(te * 1e9)
        
        if T:
            stream_channel_name = self.port + '_temperature_tar_baf_DATASTREAM'
        else:
            stream_channel_name = self.port + '_heat_flux_tar_baf_DATASTREAM'
        
        namelist = np.hstack([self._archiveComp, stream_channel_name, '0', stream_channel_name])
        url = w7xarchive.versionize_url(namelist, tsstamp, testamp, use_last_version = True)
        
        # use cpu_count intervals normally, with time step limited in [0.1, 4] s.
        # the interval duration can not be longer than 4 s, due to the archhive 
        # only allow data size smaller than 2 GB in one call.
        # For very long discharge, the RAM may be full, try use shorter ts, te.
        step = np.min([np.max([int(0.1 * 1e9), int((testamp - tsstamp) / cpu_count)]), int(4 * 1e9)]) 
        # notice that np.arange works stably only when step is an integer.
        split_time = np.hstack([np.arange(tsstamp, testamp, step), testamp + 1])
        intervals = []
        for i in range(len(split_time) - 1):
            intervals.append([split_time[i], split_time[i+1] - 1])
        processNO = len(intervals)
        
        print (str(processNO) + ' Processes Reading ' + self.PID + '_' + self.port + \
               ' from ' + str(ts) + 's' + ' upto ' + str(te) + 's')    
        st = time.time()
        
        processes = [None] * processNO
        
        if cpu_count == 1:
            # avoid using mutiprocess if only one CPU is required.
            for i in range(processNO):
                processes[i] = get_signal_ts_te(url, intervals[i])
        else:
            #pool = multiprocess.Pool(multiprocess.cpu_count()) # use all available processors
            pool = multiprocess.Pool(cpu_count)
            processes = [None] * processNO
            for i in range(processNO):
                processes[i] = pool.apply_async(get_signal_ts_te, args = (url, intervals[i]))

            pool.close()
            pool.join()
        
        timestamps, datas = [], []
        for i in range(processNO):
            if cpu_count == 1:
                temp = processes[i]
                timestamps.append(temp[0])
                datas.append(temp[1])
            else:
                if processes[i].successful():
                    # the last interval could have no data frame, then skip.
                    temp = processes[i].get()
                    timestamps.append(temp[0])
                    datas.append(temp[1])
            
        readingtime = time.time() - st
        print ('Processes finished after ' + str(round(readingtime, 2)) + 's')
        
        timestamps = np.hstack(timestamps)
        datas = np.vstack(datas)            
        
        times = (timestamps - self.t1) / 1e9
                 
        if average:
            # although it is only one frame, we make a list out of it
            # for consistency of datas dimension.
            datas = np.array([np.mean(datas, axis = 0)])
            
        self._datas = datas
        self.datatimestamps = timestamps
        self.datatimes = times
        self.T = T
        self.average = average
        
        formattime = time.time() - st
        print ('Get Frames after ' + str(round(formattime, 2)) + 's')
        
#%% Get the frame data and relavant plot in 2D target coordinate, i.e. tt1, tt2.
           
    def get_TMs (self, TMnames = ['targets'], clipthreshold = 0, clear_negative = False):
        
        '''
        This function sort and process the downloaded merged data in self_datas, 
        from get_Frames function.

        Parameters
        ----------
        TMnames : list of strings, optional
            data in which target modules you want.
            can be selected from:
            ['TM1h', 'TM2h', 'TM3h', 'TM4h', 'TM5h', 'TM6h', 'TM7h', 'TM8h',
            'TM9h', 'TM1v', 'TM2v', 'TM3v', 'lowiota', 'middleiota', 'highiota',
            'vertical', 'horizontal', 'targets', 'baffles', 'inner', 'outer']
            and self._stacknames, e.g. ['1lh_l_00', '32000001'].
            The default is ['targets'].            
        clipthreshold : float, optional
            absoulte value equal or smaller than it will be set to 0. The default is 0, 
            meanning no clipping.
        clear_negative : boolean, optional
            clip negative value or not. The default is False, meanning not clipping.

        Returns
        -------
        None.

        '''
        if not hasattr(self, '_datas'):
            raise RuntimeError('Please run get_Frames first')
            
        if hasattr(self, 'dataformat'):
            if 'Profile' in self.dataformat:
                # delete attributes belong to get_Profiles, if you run it before. Does not matter.
                try:
                    del self.stackS
                    del self.stackft1
                    del self.stackft2
                    del self.stacklineNO
                except:
                    pass
            
        propertylist = ['_datas', '_tt1', '_tt2', '_x', '_y', '_z']
        for prop in propertylist:
            datatemp = deepcopy(getattr(self, prop))
            
            if prop == '_datas':
                ndim3 = True
                # datatemp[abs(datatemp) <= clipthreshold] = 0
                # invalid value warnings because of nan value, although the result is correct.
                # use below to aviod warnings.
                mask = ~np.isnan(datatemp)
                valuemask = (abs(datatemp[mask]) <= clipthreshold)
                mask[mask] = valuemask
                datatemp[mask] = 0

                if clear_negative:
                    mask = ~np.isnan(datatemp)
                    valuemask = (datatemp[mask] < 0)
                    mask[mask] = valuemask
                    datatemp[mask] = 0
            else:
                ndim3 = False
                    
            datastacks = splitMergedlocationalltoStacks(datatemp, self._MergedProfileIndices, ndim3 = ndim3)[1]
            data_TMs, stackindices = [], []
            for TMname in TMnames:
                if TMname + '_Contains' in self._TMnames:
                    stackindex = self._TMcontains[self._TMnames == TMname + '_Contains'][0]
                elif TMname in self._stacknames:
                    stackindex = np.where(self._stacknames == TMname)[0]
                else:
                    raise NameError(TMname + '_is not the one from either self._TMnames or self._stacknames')
                stackindices.extend(stackindex)
                
            stackindices = np.array(list(set(stackindices)))
            datanew = [datastacks[x] for x in stackindices]
            datanew = np.hstack(datanew)
            data_TMs.append(datanew)
            data_TMs = np.hstack(data_TMs)
            setattr(self, prop.split('_')[1], data_TMs)
        self.clipthreshold = clipthreshold
        self.clear_negative = clear_negative
        self.TMnames = TMnames
        self.title = self.PID.replace('.', '_') + '_' + self.port + '_' + str(int(np.rint(self.datatimes[0] * 1e3))) + '_to_' + str(int(np.rint(self.datatimes[-1] * 1e3))) + '_ms'
        self.dataformat = 'Target_module_scattered_data'

    def plot_Frames (self, savefolder = None, colorbar = None, xlim = None, ylim = None):
        if not hasattr(self, 'datas'):
            raise RuntimeError ('please run get_TMs or get_Profiles first')
        if len(self.datas) == 1:
            plot_Frame(self.tt1, self.tt2, self.datas[0], self.title, self.T, \
                       savefolder = savefolder, colorbar = colorbar, xlim = xlim, ylim = ylim)
        else:
            for data, t in list(zip(self.datas, self.datatimes)):
                title = self.PID.replace('.', '_') + '_' + self.port + '_' + \
                    str(int(np.rint(t * 1e3))) + '_ms'
                plot_Frame(self.tt1, self.tt2, data, title, self.T, \
                       savefolder = savefolder, colorbar = colorbar, xlim = xlim, ylim = ylim)
                    
#%% Get the profile data and relavant plot in finger coordinate, i.e. ft1, ft2, S

    def get_Profiles (self, stackname = '1lh_l_11', clipthreshold = 0, clear_negative = False, \
                      AverageNearby = None):
        
        if not stackname in self._stacknames:
            raise NameError(stackname + '_is not the one from self._stacknames')
        stackindex = np.where(self._stacknames == stackname)[0][0]
        self.get_TMs([stackname], clipthreshold, clear_negative)
        stackrowcount = self._stackrowcounts[stackindex]
        stacklineNO = self._stacklineNOs[stackindex]
        
        propertylist = ['datas', '_stackS', '_stackft1', '_stackft2', 'x', 'y', 'z', 'tt1', 'tt2']
        for prop in propertylist:
            datatemp = deepcopy(getattr(self, prop))
            if prop == 'datas':
                ndim3 = True
            else:
                ndim3 = False
            if '_stack' in prop:
                datatemp = datatemp[stackindex]
                
            linedatas = splitStacktoLines(datatemp, stackrowcount, ndim3 = ndim3)
            sortedlinedatas = [P for (NO, P) in sorted(zip(stacklineNO, linedatas))]
            setattr(self, prop.split('_')[-1], sortedlinedatas)
        
        self.stacklineNO = np.array(sorted(stacklineNO))
        # notice that the lineNOs is increased from 0, with increased phi direction for
        # h, v, i, but for outer baffle it is with decreased phi direction.
        
        # change the order of datas to [time, profiles, S]
        totalframes = len(self.datas[0])
        datasnew = []
        
        for t in range(totalframes):
            datapert = []
            for data in self.datas:
                datapert.append(data[t])
            datasnew.append(datapert)
        self.datas = np.array(datasnew, dtype = 'object')
        ############################################
        
        # change the S = 0 to always from pumping gap,
        # notice that the S was from oppsite direction for vertical and baffles.
        # add cases for OP2
        if stackname[:3] == '320' or stackname[2] == 'v' or stackname[3] == 'v' or stackname[:2] == 'BM':
            stackSnew = []
            for S in self.stackS:
                Snew = S[-1] - S
                stackSnew.append(Snew)
            self.stackS = stackSnew
        ############################################

        if AverageNearby is not None:
            middle_line_index = len(stacklineNO) // 2
            middle_S = self.stackS[middle_line_index]
            
            # to avoid negative index
            start_line_index = middle_line_index - AverageNearby
            if start_line_index < 0:
                start_line_index = 0

            end_line_index = middle_line_index + AverageNearby
            
            Sall = self.stackS[start_line_index : (end_line_index + 1) ]
            datasall = self.datas[:, start_line_index : (end_line_index + 1)]
            
            interpdatas_all = []
            for dataall in datasall:
                interpdata_all = []
                for S, data in list(zip(Sall, dataall)): 
                    if S[0] > S[1]:
                        # decreasing order need reverse before interpolation.
                        interpdata_all.append(np.interp(middle_S, S[::-1], data[::-1]))
                    else:
                        interpdata_all.append(np.interp(middle_S, S, data))

                interpdata_all = np.mean(interpdata_all, axis = 0)
                interpdatas_all.append(interpdata_all)                        
            
            self.datas = np.array(interpdatas_all)
            
            for prop in propertylist[1:]:
                prop = prop.split('_')[-1]
                data = getattr(self, prop)
                datanew = data[middle_line_index]
                setattr(self, prop, datanew)
            
        self.dataformat = 'Profile_data_' + str(AverageNearby)
    
    def get_Stacks_ave_Profiles (self, stacknames, clipthreshold = 0, clear_negative = False, \
                      AverageNearby = 100):
        # notice that assuming AverageNearby >= 1, so self.datas in [time, S], self.stackS in [S]
        Ss, datas, Smaxs = [], [], []
        for stackname in stacknames:
            self.get_Profiles(stackname, clipthreshold, clear_negative, \
                      AverageNearby)
            Ss.append(self.stackS)
            Smaxs.append(self.stackS.max())
            datas.append(self.datas)
        Smaxs = np.array(Smaxs)
        maxindex = np.argmax(Smaxs)
        Snorm = Ss[maxindex]
        
        # change the order of datas to [time, profiles, S]
        totalframes = len(datas[0])
        datasnew = []
        
        for t in range(totalframes):
            datapert = []
            for data in datas:
                datapert.append(data[t])
            datasnew.append(datapert)
        datas = np.array(datasnew)
        
        ######### interpolate to Snorm #######
        datasnew = []
        for datapert in datas:
            datanew = []
            for S, data in list(zip(Ss, datapert)):
                if S[0] > S[1]:
                    # decreasing order need reverse before interpolation
                    # notice that there is a gap in vertical target, so from top is recommended.
                    Srev = S[0] - S 
                    Snormrev = Snorm[0] - Snorm
                    datanorm = np.interp(Snormrev, Srev, data, left = np.nan, right = np.nan)
                else:
                    datanorm = np.interp(Snorm, S, data, left = np.nan, right = np.nan)
                datanew.append(datanorm)
            datanew = np.nanmean(datanew, axis = 0)
            datasnew.append(datanew)
        self.datas = np.array(datasnew)
        self.stackS = Snorm
            
    def plot_Fingers (self, savefolder = None, colorbar = None, xlim = None, ylim = None):
        # notice that for outer baffle the finger coordinate here is flipped in Z (y axis) 
        # and phi (x axis) direction with respect to target coordinate.
        if 'Profile' not in self.dataformat:
            raise RuntimeError('Please run get_Profiles first')
            
        if len(self.datas) == 1:
            # only one time
            title = self.title + '_' + self.TMnames[0] + '_Finger'
            plot_Finger(self.stackft1, self.stackft2, self.datas[0], title, self.T, \
                          savefolder, colorbar, xlim, ylim)
        else:
            for data, t in list(zip(self.datas, self.datatimes)):
                title = self.PID.replace('.', '_') + '_' + self.port + '_' + \
                    str(int(np.rint(t * 1e3))) + '_ms_' + self.TMnames[0] + '_Finger'
                plot_Finger(self.stackft1, self.stackft2, data, title, self.T, \
                       savefolder, colorbar, xlim, ylim)       
                    
    def plot_Profiles (self, savefolder = None, xlim = None, ylim = None):
        
        if 'Profile' not in self.dataformat:
            raise RuntimeError('Please run get_Profiles first')
            
        if len(self.datas) == 1:
            # only one time
            title = self.title + '_' + self.TMnames[0]
            plot_Profile(self.stackS, self.datas[0], title, self.T, \
                          savefolder, xlim, ylim)
        else:
            for data, t in list(zip(self.datas, self.datatimes)):
                title = self.PID.replace('.', '_') + '_' + self.port + '_' + \
                    str(int(np.rint(t * 1e3))) + '_ms_' + self.TMnames[0]
                plot_Profile(self.stackS, data, title, self.T, \
                       savefolder, xlim, ylim)
                    
#%% get TMs time trace    
    def __get_meanHF_TMs (self):    
        self.availtimestamps, meanHF_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_meanHF_TMs_DATASTREAM'), self.PID, use_last_version = True, relative_time = False)
        self.t1 = w7xarchive.get_program_t1(self.PID)
        self.availtimes = (self.availtimestamps - self.t1) * 1e-9
        self._meanHF_TMs, self._integralpower_TMs = {}, {}
        for TMname, meanHF, TMcontain in list(zip(self._TMnames, meanHF_TMs, self._TMcontains)):
            self._meanHF_TMs[TMname.split('_Contains')[0]] = meanHF
            self._integralpower_TMs[TMname.split('_Contains')[0]] = meanHF * self._areastacks[TMcontain].sum()    
            
    def __get_strikewidth_TMs (self):
        self.availtimes, strikelinewidth_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_strikelinewidth_TMs_DATASTREAM'), self.PID, use_last_version = True)             
        self._strikelinewidth_TMs = {}
        for TMname, strikelinewidth, TMcontain in list(zip(self._TMnames, strikelinewidth_TMs, self._TMcontains)):
            self._strikelinewidth_TMs[TMname.split('_Contains')[0]] = strikelinewidth

    def __get_wetted_area_TMs (self):
        self.availtimes, wetted_area_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_wetted_area_TMs_DATASTREAM'), self.PID, use_last_version = True)                
        self._wetted_area_TMs = {}
        for TMname, wetted_area, TMcontain in list(zip(self._TMnames, wetted_area_TMs, self._TMcontains)):
            self._wetted_area_TMs[TMname.split('_Contains')[0]] = wetted_area
            
    def __get_maxHF_TMs (self):
        self.availtimes, maxHF_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_maxHF_TMs_DATASTREAM'), self.PID, use_last_version = True)                
        self._maxHF_TMs = {}
        for TMname, maxHF_TM, TMcontain in list(zip(self._TMnames, maxHF_TMs, self._TMcontains)):
            self._maxHF_TMs[TMname.split('_Contains')[0]] = maxHF_TM

    def __get_meanT_TMs (self):
        self.availtimes, meanT_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_meanT_TMs_DATASTREAM'), self.PID, use_last_version = True)               
        self._meanT_TMs = {}
        for TMname, meanT_TM, TMcontain in list(zip(self._TMnames, meanT_TMs, self._TMcontains)):
            self._meanT_TMs[TMname.split('_Contains')[0]] = meanT_TM
    
    def __get_maxT_TMs (self):
        self.availtimes, maxT_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_maxT_TMs_DATASTREAM'), self.PID, use_last_version = True)               
        self._maxT_TMs = {}
        for TMname, maxT_TM, TMcontain in list(zip(self._TMnames, maxT_TMs, self._TMcontains)):
            self._maxT_TMs[TMname.split('_Contains')[0]] = maxT_TM
            
#%% get stacks time trace            
    
    def __get_meanHF_stacks (self):    
        self.availtimes, meanHF_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_meanHF_stacks_DATASTREAM'), self.PID, use_last_version = True)
        self._meanHF_stacks, self._integralpower_stacks = {}, {}
        for stackname, meanHF, area in list(zip(self._stacknames, meanHF_stacks, self._areastacks)):
            self._meanHF_stacks[stackname] = meanHF
            self._integralpower_stacks[stackname] = meanHF * area   
            
    def __get_strikewidth_stacks (self):
        self.availtimes, strikelinewidth_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_strikelinewidth_stacks_DATASTREAM'), self.PID, use_last_version = True)             
        self._strikelinewidth_stacks = {}
        for stackname, strikelinewidth in list(zip(self._stacknames, strikelinewidth_stacks)):
            self._strikelinewidth_stacks[stackname] = strikelinewidth

    def __get_wetted_area_stacks (self):
        self.availtimes, wetted_area_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_wetted_area_stacks_DATASTREAM'), self.PID, use_last_version = True)                
        self._wetted_area_stacks = {}
        for stackname, wetted_area in list(zip(self._stacknames, wetted_area_stacks)):
            self._wetted_area_stacks[stackname] = wetted_area
            
    def __get_maxHF_stacks (self):
        self.availtimes, maxHF_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_maxHF_stacks_DATASTREAM'), self.PID, use_last_version = True)                
        self._maxHF_stacks = {}
        for stackname, maxHF_TM in list(zip(self._stacknames, maxHF_stacks)):
            self._maxHF_stacks[stackname] = maxHF_TM

    def __get_meanT_stacks (self):
        self.availtimes, meanT_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_meanT_stacks_DATASTREAM'), self.PID, use_last_version = True)               
        self._meanT_stacks = {}
        for stackname, meanT_TM in list(zip(self._stacknames, meanT_stacks)):
            self._meanT_stacks[stackname] = meanT_TM
    
    def __get_maxT_stacks (self):
        self.availtimes, maxT_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_maxT_stacks_DATASTREAM'), self.PID, use_last_version = True)               
        self._maxT_stacks = {}
        for stackname, maxT_TM in list(zip(self._stacknames, maxT_stacks)):
            self._maxT_stacks[stackname] = maxT_TM
            
#%% TMs properties            
    @property
    def meanHF_TMs (self):
      if self._meanHF_TMs is None:
        self.__get_meanHF_TMs()
      return self._meanHF_TMs
  
    @property
    def integralpower_TMs (self):
      if self._integralpower_TMs is None:
        self.__get_meanHF_TMs()
      return self._integralpower_TMs 
  
    @property
    def strikelinewidth_TMs (self):
      if self._strikelinewidth_TMs is None:
        self.__get_strikewidth_TMs()
      return self._strikelinewidth_TMs    

    @property
    def wetted_area_TMs (self):
      if self._wetted_area_TMs is None:
        self.__get_wetted_area_TMs()
      return self._wetted_area_TMs   

    @property
    def maxHF_TMs (self):
      if self._maxHF_TMs is None:
        self.__get_maxHF_TMs()
      return self._maxHF_TMs

    @property
    def meanT_TMs (self):
      if self._meanT_TMs is None:
        self.__get_meanT_TMs()
      return self._meanT_TMs 

    @property
    def maxT_TMs (self):
      if self._maxT_TMs is None:
        self.__get_maxT_TMs()
      return self._maxT_TMs 

#%% stacks properties
    @property
    def meanHF_stacks (self):
      if self._meanHF_stacks is None:
        self.__get_meanHF_stacks()
      return self._meanHF_stacks
  
    @property
    def integralpower_stacks (self):
      if self._integralpower_stacks is None:
        self.__get_meanHF_stacks()
      return self._integralpower_stacks 
  
    @property
    def strikelinewidth_stacks (self):
      if self._strikelinewidth_stacks is None:
        self.__get_strikewidth_stacks()
      return self._strikelinewidth_stacks    

    @property
    def wetted_area_stacks (self):
      if self._wetted_area_stacks is None:
        self.__get_wetted_area_stacks()
      return self._wetted_area_stacks   

    @property
    def maxHF_stacks (self):
      if self._maxHF_stacks is None:
        self.__get_maxHF_stacks()
      return self._maxHF_stacks

    @property
    def meanT_stacks (self):
      if self._meanT_stacks is None:
        self.__get_meanT_stacks()
      return self._meanT_stacks 

    @property
    def maxT_stacks (self):
      if self._maxT_stacks is None:
        self.__get_maxT_stacks()
      return self._maxT_stacks 

#%% We generated v2_meta, which may be more physics relavant, because all the results
# are only using middle 3 lines of the heat flux data, thus excluding leading edges.
# Baffles are not calculated.      
# strike line locations are added.
                    
#%% get TMs time trace    
    def __get_meanHF_TMs_v2 (self):    
        self.availtimestamps, meanHF_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_meanHF_TMs_v2_DATASTREAM'), self.PID, use_last_version = True, relative_time = False)
        self.t1 = w7xarchive.get_program_t1(self.PID)
        self.availtimes = (self.availtimestamps - self.t1) * 1e-9
        self._meanHF_TMs_v2, self._integralpower_TMs_v2 = {}, {}
        for TMname, meanHF, TMcontain in list(zip(self._TMnames, meanHF_TMs, self._TMcontains)):
            self._meanHF_TMs_v2[TMname.split('_Contains')[0]] = meanHF
            self._integralpower_TMs_v2[TMname.split('_Contains')[0]] = meanHF * self._areastacks[TMcontain].sum()    
            
    def __get_strikewidth_TMs_v2 (self):
        self.availtimes, strikelinewidth_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_strikelinewidth_TMs_v2_DATASTREAM'), self.PID, use_last_version = True)             
        self._strikelinewidth_TMs_v2 = {}
        for TMname, strikelinewidth, TMcontain in list(zip(self._TMnames, strikelinewidth_TMs, self._TMcontains)):
            self._strikelinewidth_TMs_v2[TMname.split('_Contains')[0]] = strikelinewidth

    def __get_wetted_area_TMs_v2 (self):
        self.availtimes, wetted_area_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_wetted_area_TMs_v2_DATASTREAM'), self.PID, use_last_version = True)                
        self._wetted_area_TMs_v2 = {}
        for TMname, wetted_area, TMcontain in list(zip(self._TMnames, wetted_area_TMs, self._TMcontains)):
            self._wetted_area_TMs_v2[TMname.split('_Contains')[0]] = wetted_area
            
    def __get_maxHF_TMs_v2 (self):
        self.availtimes, maxHF_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_maxHF_TMs_v2_DATASTREAM'), self.PID, use_last_version = True)                
        self._maxHF_TMs_v2 = {}
        for TMname, maxHF_TM, TMcontain in list(zip(self._TMnames, maxHF_TMs, self._TMcontains)):
            self._maxHF_TMs_v2[TMname.split('_Contains')[0]] = maxHF_TM
            
    def __get_loc_TMs_v2 (self):
        self.availtimes, loc_TMs = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_loc_TMs_v2_DATASTREAM'), self.PID, use_last_version = True)                
        self._loc_TMs_v2 = {}
        for TMname, loc_TM, TMcontain in list(zip(self._TMnames, loc_TMs, self._TMcontains)):
            self._loc_TMs_v2[TMname.split('_Contains')[0]] = loc_TM
            
#%% get stacks time trace            
    
    def __get_meanHF_stacks_v2 (self):    
        self.availtimes, meanHF_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_meanHF_stacks_v2_DATASTREAM'), self.PID, use_last_version = True)
        self._meanHF_stacks_v2, self._integralpower_stacks_v2 = {}, {}
        for stackname, meanHF, area in list(zip(self._stacknames, meanHF_stacks, self._areastacks)):
            self._meanHF_stacks_v2[stackname] = meanHF
            self._integralpower_stacks_v2[stackname] = meanHF * area   
            
    def __get_strikewidth_stacks_v2 (self):
        self.availtimes, strikelinewidth_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_strikelinewidth_stacks_v2_DATASTREAM'), self.PID, use_last_version = True)             
        self._strikelinewidth_stacks_v2 = {}
        for stackname, strikelinewidth in list(zip(self._stacknames, strikelinewidth_stacks)):
            self._strikelinewidth_stacks_v2[stackname] = strikelinewidth

    def __get_wetted_area_stacks_v2 (self):
        self.availtimes, wetted_area_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_wetted_area_stacks_v2_DATASTREAM'), self.PID, use_last_version = True)                
        self._wetted_area_stacks_v2 = {}
        for stackname, wetted_area in list(zip(self._stacknames, wetted_area_stacks)):
            self._wetted_area_stacks_v2[stackname] = wetted_area
            
    def __get_maxHF_stacks_v2 (self):
        self.availtimes, maxHF_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_maxHF_stacks_v2_DATASTREAM'), self.PID, use_last_version = True)                
        self._maxHF_stacks_v2 = {}
        for stackname, maxHF_TM in list(zip(self._stacknames, maxHF_stacks)):
            self._maxHF_stacks_v2[stackname] = maxHF_TM
            
    def __get_loc_stacks_v2 (self):
        self.availtimes, loc_stacks = w7xarchive.get_signal_for_program(np.append(self._archiveComp, self.port + '_loc_stacks_v2_DATASTREAM'), self.PID, use_last_version = True)                
        self._loc_stacks_v2 = {}
        for stackname, loc_stack in list(zip(self._stacknames, loc_stacks)):
            self._loc_stacks_v2[stackname] = loc_stack
            
#%% TMs properties            
    @property
    def meanHF_TMs_v2 (self):
      if self._meanHF_TMs_v2 is None:
        self.__get_meanHF_TMs_v2()
      return self._meanHF_TMs_v2
  
    @property
    def integralpower_TMs_v2 (self):
      if self._integralpower_TMs_v2 is None:
        self.__get_meanHF_TMs_v2()
      return self._integralpower_TMs_v2 
  
    @property
    def strikelinewidth_TMs_v2 (self):
      if self._strikelinewidth_TMs_v2 is None:
        self.__get_strikewidth_TMs_v2()
      return self._strikelinewidth_TMs_v2   

    @property
    def wetted_area_TMs_v2 (self):
      if self._wetted_area_TMs_v2 is None:
        self.__get_wetted_area_TMs_v2()
      return self._wetted_area_TMs_v2   

    @property
    def maxHF_TMs_v2 (self):
      if self._maxHF_TMs_v2 is None:
        self.__get_maxHF_TMs_v2()
      return self._maxHF_TMs_v2
  
    @property
    def loc_TMs_v2 (self):
      if self._loc_TMs_v2 is None:
        self.__get_loc_TMs_v2()
      return self._loc_TMs_v2
  
#%% stacks properties
    @property
    def meanHF_stacks_v2 (self):
      if self._meanHF_stacks_v2 is None:
        self.__get_meanHF_stacks_v2()
      return self._meanHF_stacks_v2
  
    @property
    def integralpower_stacks_v2 (self):
      if self._integralpower_stacks_v2 is None:
        self.__get_meanHF_stacks_v2()
      return self._integralpower_stacks_v2 
  
    @property
    def strikelinewidth_stacks_v2 (self):
      if self._strikelinewidth_stacks_v2 is None:
        self.__get_strikewidth_stacks_v2()
      return self._strikelinewidth_stacks_v2   

    @property
    def wetted_area_stacks_v2 (self):
      if self._wetted_area_stacks_v2 is None:
        self.__get_wetted_area_stacks_v2()
      return self._wetted_area_stacks_v2   

    @property
    def maxHF_stacks_v2 (self):
      if self._maxHF_stacks_v2 is None:
        self.__get_maxHF_stacks_v2()
      return self._maxHF_stacks_v2

    @property
    def loc_stacks_v2 (self):
      if self._loc_stacks_v2 is None:
        self.__get_loc_stacks_v2()
      return self._loc_stacks_v2
  
#%%   
    def fft_filter_single(self, attri_name = 'integralpower_TMs_v2', keyname = 'targets', \
                          frequencythreshold = 0.5, AlsoSGfit = True, window_length = 11, polyorder = 1, \
                          shift_offset = True):
        data = getattr(self, attri_name)[keyname]
        datanew = fftfiltering(self.availtimes, data, frequencythreshold, False, \
                               AlsoSGfit, window_length, polyorder, shift_offset)[1]
        return datanew
            
#%% internal functions for splitting merged data
def splitMergedlocationalltoStacks (locationall2D_Tar, rowcountarrayTar, ndim3 = False):
    '''
    Internal function only to work with 'MergedProfileIndices'
    to split the Merged data to stacks.

    Parameters
    ----------
    locationall2D_Tar : 1D or multi-dimension array
        Merged data to be splitted. 
        If the data have time information, the first axes is time.
    rowcountarrayTar : 1D array or list
        Merged indices for splitting.
    ndim3 : boolean, optional
        if the data has time dimension or not. The default is False.

    Returns
    -------
    stackrowcount : list
        all the line indices each splitted stack contains.
    stacklocations : list
        splitted data.

    '''
    
    row0indices = np.where(rowcountarrayTar == 0)[0]
    
    stackrowcount = []
    stacklocations = []
    for i in range(len(row0indices) - 1):
        stackrowcount.append(rowcountarrayTar[row0indices[i] : row0indices[i+1]])
    stackrowcount.append(rowcountarrayTar[row0indices[-1] : ])   
    
    indicestotal = [stackrowcount[0]]
    accumulateindex = 0
    for i in range(1, len(stackrowcount)):
        accumulateindex += stackrowcount[i-1][-1]
        indicestotal.append(stackrowcount[i] + accumulateindex)
    
    for i in indicestotal:
        if ndim3 == False:
            stacklocations.append(locationall2D_Tar[i[0] : i[-1]])
        else:
            stacklocations.append(locationall2D_Tar[:, i[0] : i[-1]])
    return stackrowcount, stacklocations

def splitStacktoLines (locationall, rowcountarray, ndim3 = False):
    '''
    Internal functions only to work with stackrowcount to split merged stack
    data to line profiles.

    Parameters
    ----------
    locationall : 1D or muti-dimenstion array
        data to be splitted.
        If the data have time information, the first axes is time.
    rowcountarray : list or 1D array
        DESCRIPTION.
    ndim3 : boolean, optional
        if the data has time dimension or not. The default is False.

    Returns
    -------
    locations : list
        list of splitted data.

    '''
    
    locationall = np.array(locationall)
    #if not locationall.flatten().any():  
    if locationall.ndim == 0:
        locations = False
    else:
        locations = []
        for i in range(len(rowcountarray) - 1):
            if ndim3 == False:
                locations.append(np.array(locationall[rowcountarray[i]:rowcountarray[i+1]]))
            else:
                locations.append(np.array(locationall[:, rowcountarray[i]:rowcountarray[i+1]]))
        
    return locations

#%% in OP2.1 we have strong noise in the IR camreas.
# we use FFT filtering to clear the noise.
def fftfiltering (x, yo, frequencythreshold = 0.5, plot = False, \
                  AlsoSGfit = True, window_length = 11, polyorder = 1, shift_offset = True):
    # frequencythreshold means only the frequency in between the +/-threshold is kept.
    y = deepcopy(yo)
    if np.where(np.isnan(y))[0].size != 0:
        for i in np.where(np.isnan(y)) :
            y[i] = (y[i-1] + y[i+1]) / 2
    frequencythreshold = abs(frequencythreshold)
    print (y)
    W = fftfreq( y.size, d = np.mean(np.diff(x)) )
    f_signal = rfft(y)
    cut_f_signal = deepcopy(f_signal)
    cut_f_signal[(W < -frequencythreshold)] = 0
    cut_f_signal[(W > frequencythreshold)] = 0
    cut_signal = irfft(cut_f_signal)
    
    if AlsoSGfit:
        SGsignal = SGfit(cut_signal, window_length, polyorder)
    else:
        SGsignal = deepcopy(cut_signal)
    
    if shift_offset:
        offset = y[-90:].mean()
        SGsignal -= offset
        cut_signal -= offset
        y -= offset
        
    if plot == True:
        #FontSize = 25
        #matplotlib.rcParams.update( { 'font.size': FontSize } )   
        #plt.rcParams["figure.figsize"] = (18,9)
        plt.figure()
        plt.subplot(221)
        plt.plot(x,y)
        plt.grid('on')
        plt.subplot(222)
        plt.plot(W,f_signal)
        plt.grid('on')
        plt.subplot(223)
        plt.plot(W,cut_f_signal)
        plt.grid('on')
        plt.subplot(224)
        plt.plot(x,y, 'o')
        plt.plot(x, cut_signal)
        plt.plot(x, SGsignal)
        plt.grid('on')
    return x, SGsignal

def SGfit(signal, window_length = 11, polyorder = 1):
    # 201, 3
    return scipy.signal.savgol_filter(signal, window_length, polyorder)