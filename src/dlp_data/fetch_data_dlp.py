import w7xarchive
import probe_datastreams as streams
from scipy.optimize import curve_fit
import numpy as np
import ast
from urllib.error import HTTPError
import urllib.request
import json
import datetime
import probe_geometry as pg

reference_datastreams = streams.referenceprobedb
probe_datastreams = streams.probesdb

class Probes:
    def __init__(self, probe, shot='20250225.055', source=None, piece_fetching=True):
        datasource = 'ArchiveDB'
        self.probename = probe
        self.shot = shot
        self.piece_fetching = piece_fetching
        self.source = source or datasource
        self._set_defaults()
        self._set_data_sources()
        self._load_probe_metadata()
        self._post_init_conditions()
        self.get_timestamps()
        self.get_logs()

    def __repr__(self):
        return f"<Probes(probename='{self.probename}', shot='{self.shot}')>"

    def __str__(self):
        return f"Langmuir Probe '{self.probename}' for shot {self.shot}"

    def _set_defaults(self):
        self.scale_linear = 3.0517578125E-4
        self.scale_offset = 0
        self.resistance = 50  # ohms
        self.rate = 500000
        self.defekt = False

    def _set_data_sources(self):
        self.get_probe_source()
        self.source_input_voltage = self.source + streams.inputvoltagedb[self.bridge_probe_index]
        self.get_reference_source()
        self.offset_voltage_factor = pg.offset_factor[self.index]

    def _load_probe_metadata(self):
        self.area = pg.probe_area[self.index] * 1e-6
        self.psi = pg.probe_psi[self.index]
        self.theta = get_theta_probe(self.psi)
        self.gain = 2 / 5 if float(self.shot) > 20240000 else 1

    def _post_init_conditions(self):
        if float(self.shot) > 20241114.018 and self.probename == '50246':
            self.defekt = True

    def code_settings(self):
        settings_path = f"{self.source}/raw/W7X/ControlStation.72001/CONTROL-0_PARLOG/file/code"
        self.code = w7xarchive.get_parameters_box_for_program(settings_path, self.shot)

    def get_logs(self):
        self.parlog = parlog(self.shot)
        self.parlog.code_settings()
        get_drive(self)
        get_active_time(self, self.parlog)
        self.number_of_ivs = len(self.pop_up_time)

    def get_probe_source(self):
        shot_float = float(self.shot)
        probe_data = probe_datastreams.get(self.probename, {})

        if isinstance(probe_data, dict):
            thresholds = sorted((float(k), k) for k in probe_data if k != 'default')
            selected_key = max((key for val, key in thresholds if shot_float >= val), default='default')
            source_info = probe_data.get(selected_key, probe_data['default'])
        else:
            source_info = probe_data

        self.source_probe_voltage = self.source + source_info[0]
        self.bridge_probe_index = source_info[1]
        self.index = list(probe_datastreams.keys()).index(self.probename)

    def get_reference_source(self):
        entry = reference_datastreams[self.bridge_probe_index]
        if isinstance(entry, list):
            index = -1 if float(self.shot) > 20250226 else 0
            entry = entry[index]
        self.source_reference_voltage = self.source + entry

    def get_timestamps(self):
        try:
            url = f"http://archive-webapi.ipp-hgw.mpg.de/programs.json?from={self.shot}"
            response = url_request(url)
            triggers = response['programs'][0]['trigger']
            self.time_range = np.asarray([triggers['1'][0], triggers['4'][0]])
        except IndexError:
            self.time_range = w7xarchive.get_program_from_to(self.shot)

    def _process_voltage(self, raw_voltage, calibration_key):
        voltage = (raw_voltage * self.scale_linear + self.scale_offset) * self.gain
        slope, offset = pg.calibration_parameters[self.index]
        return (voltage + offset) * slope

    def get_input_voltage(self):
        if self.piece_fetching:
            self.time, raw = get_signal_piecewise(self.source_input_voltage, self)
        else:
            tm, raw = w7xarchive.get_signal(self.source_input_voltage, *self.time_range)
            self.time = time_array(tm)
        self.input_voltage = self._process_voltage(raw, 'input')

    def get_reference_voltage(self):
        if self.piece_fetching:
            self.time, raw = get_signal_piecewise(self.source_reference_voltage, self)
        else:
            tm, raw = w7xarchive.get_signal(self.source_reference_voltage, *self.time_range)
            self.time = time_array(tm)
        self.reference_voltage = self._process_voltage(raw, 'reference')

    def get_probe_voltage(self):
        if self.piece_fetching:
            self.time, raw = get_signal_piecewise(self.source_probe_voltage, self)
        else:
            tm, raw = w7xarchive.get_signal(self.source_probe_voltage, *self.time_range)
            self.time = time_array(tm)
        self.probe_voltage = self._process_voltage(raw, 'probe')

    def get_current(self):
        l = min(len(self.probe_voltage), len(self.reference_voltage))
        self.reference_voltage = self.reference_voltage[:l]
        self.probe_voltage = self.probe_voltage[:l]

        if getattr(self, "termination", True):
            current = (self.reference_voltage - self.probe_voltage) / self.resistance
        else:
            f1 = pg.gain_difference[self.index]
            current = (self.reference_voltage - self.probe_voltage) / self.resistance
            current = (self.probe_voltage * (1 / f1 - 1) + (self.resistance * current / f1)) / self.resistance
            current -= self.offset_voltage_factor

        self.probe_current = current
        self.R, self.Offset = fit_resistor_mismatch(self)
        self.probe_current -= self.R * self.probe_voltage + self.Offset
        self.pop_up_time = self.pop_up_time[:self.number_of_ivs]

    def get_data(self):
        self.get_reference_voltage()
        self.get_probe_voltage()
        self.get_current()

def url_request(url_str):
    request = urllib.request.Request(url_str)
    request.add_header('Content-Type', 'application/json')
    url = urllib.request.urlopen(request)
    response = json.loads(url.read().decode(url.info().get_param('charset') or 'utf-8'))
    url.close()
    return response

def get_signal_piecewise(source, probe):
    time = []
    signal = []
    probe.number_of_ivs = 0
    for i, j in enumerate(probe.pop_up_time):
        try:
            t, s = get_signal_time_range(source, probe.shot, time_range = [j-1, j-0.9])
            time = np.append(time, t)
            signal = np.append(signal, s)
            probe.number_of_ivs +=1
        except Exception as e:
            print(f"{i, j}, Discharge probably shorter than anticipated.", e)
            probe.pop_up_time = probe.pop_up_time[:i]
            break
    return np.array(time), np.array(signal)

def get_signal_time_range(signal, prog, time_range = [0.0, 1.0]):
    date, pid = prog.split('.')
    pglist = w7xarchive.get_program_list_for_day('-'.join([date[0:4], date[4:6], date[6:8]]))
    progdata = [pr for pr in pglist if pr['id'] == prog]
    if len(progdata) != 1:
        raise LookupError('program {p} not found'.format(p=prog))
    progdata = progdata[0]
    triggers = progdata['trigger']
    time_range = [triggers['1'][0] + int(t * 1e9) for t in time_range]
    tm, sig = w7xarchive.get_signal(signal, *time_range)
    time = 1e-9 * (tm - int(triggers['1'][0]))    
    return time, sig

def time_array(tm):
    date_str_end = datetime.datetime.utcfromtimestamp(tm[-1]/1e9)
    date_str_st = datetime.datetime.utcfromtimestamp(tm[0]/1e9)
    time2 = ((date_str_end.hour*3600 + date_str_end.minute*60 + date_str_end.second)*1e6 + date_str_end.microsecond)
    time1 = ((date_str_st.hour*3600 + date_str_st.minute*60 + date_str_st.second)*1e6 + date_str_st.microsecond)
    time = np.linspace(time1,  time2, len(tm))
    time = (time - time[0])/1e6
    return time

def get_theta_probe(psi):
    degtorad = np.pi/180
    x = ((1.96/np.cos(psi*degtorad))/2)*np.cos(np.arcsin(2*(1/2)/1.96))
    return np.arctan((1/2)/x)/degtorad

def fit_resistor_mismatch(probe):
    def linear(x, a, b):
        return a * x + b

    x = probe.probe_voltage[:1000]
    y = probe.probe_current[:1000]

    popt, _ = curve_fit(linear, x, y)
    return popt[0], popt[1]

def get_active_time(probe, parlog):
    probe.pop_up_time = extractparlog(parlog, 'pop_up_time_s')[probe.drive_index]
    probe.hold_time = extractparlog(parlog, 'hold_time_s')[probe.drive_index]
    probe.insertion_time = extractparlog(parlog, 'insertion_time_s')[probe.drive_index]
    probe.retraction_time = extractparlog(parlog, 'retraction_time_s')[probe.drive_index]

def get_drive(probe):
    probe.drive_index = drive_index(probe.probename)
    probe.drive_name = list(pg.motion_drives.keys())[probe.drive_index]
    
def drive_index(probename):
    for i, j in enumerate(probe_datastreams.keys()):
        if j == probename:
            return int(i/2)

class parlog:
    def __init__(self, shot = '20230312.001'):
        self.shot = shot

    def code_settings(self):
        settings_source = "ArchiveDB/raw/W7X/ControlStation.72001/CONTROL-0_PARLOG/file/code"
        self.code = w7xarchive.get_parameters_box_for_program(settings_source, self.shot)           

def extractparlog(log, key):
    dictkeys=[]
    dictvalues = []
    lines = []
    for i in log.code['values']:
        j = i.split('\ndrives.')
        lines.extend(j)
    lines = lines[1:]
    for i in range(len(lines)-1):
        dictkeys.append(lines[i].split(' = ')[0].split('.'))
        dictvalues.append(lines[i].split(' = ')[1])
    if key == 'static':
        return dictvalues.pop(162)
    index = []
    for i in range(len(dictvalues)):
        try:
            if dictkeys[i][1] == key:
                index.append(i)
        except(IndexError):
            pass
    keys = []
    values=[]
    for i in index:
        values.append(ast.literal_eval(dictvalues[i]))
        keys.append(dictkeys[i][1])
    return values
