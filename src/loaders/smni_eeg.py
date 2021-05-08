import os
import urllib
import tarfile
import gzip
import shutil
import numpy as np
from . import io_tools
import collections

_smni_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/eeg_full.tar'

_BAD_PERSONS = [
    'co2c1000367', # bad recordings
    'co2a0000425', # low number of trials
    'co2c0000391' # low number of trials
]

_mistake_key = 'error'
_conditions = {
    'S1 obj': 0, # single object shown
    'S2 match': 1, # second object matching
    'S2 nomatch': 2, # second object not matching
    _mistake_key: -1
}

_classes = {
    'a': 0, # alcocholic
    'c': 1  # control
}

#https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-i$
def unsortedSet(s):
    tmp = collections.OrderedDict.fromkeys(s)
    return list(tmp.keys())

def downloadSMNI(save_data_dirname=None, remove_archive=False):
    save_data_filename = 'eeg_full.tar'
    if save_data_dirname is None:
        save_data_dirname = ''
    if not os.path.exists(save_data_dirname):
        os.makedirs(save_data_dirname, exist_ok=True)
    path = os.path.join(save_data_dirname, save_data_filename)
    urllib.request.urlretrieve(_smni_url, path)
    #print(path)
    with tarfile.open(path) as f:
        f.extractall(save_data_dirname)
    if remove_archive:
        os.remove(os.path.join(save_data_dirname, save_data_filename))
    fnms = os.listdir(save_data_dirname)
    #print(fnms)
    #fnms = list(filter(lambda x: x != subdirname, fnms))
    for fnm in fnms:
        if not fnm.endswith('.tar.gz'):
            continue
        subdirname = fnm.replace('.tar.gz', '')
        subpath = os.path.join(save_data_dirname, subdirname)
        #os.mkdir(subpath)
        with tarfile.open(os.path.join(save_data_dirname, fnm), 'r:gz') as f: #, ) as f:
            f.extractall(save_data_dirname)
        os.remove(os.path.join(save_data_dirname, fnm))
        subfnms = os.listdir(subpath)
        for subfnm in subfnms:
            subsubpath = os.path.join(subpath, subfnm)
            subsubpath_new = os.path.join(subpath, subfnm.replace('.', '_').replace('_gz', '.gz'))
            with gzip.open(subsubpath, 'r') as f:
                with open(subsubpath.replace('.gz', ''), 'wb') as g:
                    shutil.copyfileobj(f, g)
            os.remove(subsubpath)
            
def loadOneFile(path):
    filename = path.split('/')[-1]
    label = _classes[filename[3]]
    trial, channels, timepoints, voltage = [], [], [], []
    chanDict, trialDict = {}, {}
    condition = _conditions[_mistake_key]
    with open(path, 'r') as f:
        title = f.readline()
        description = f.readline()
        mV = f.readline()
        condition_line = f.readline()
        for key in _conditions:
            if key in condition_line:
                condition = _conditions[key]
                break
        line = '#'
        while len(line) > 0:
            if not line.startswith('#'):
                values = line.split(' ')
                trial.append(int(values[0]))
                channels.append(str(values[1]))
                timepoints.append(int(values[2]))
                voltage.append(float(values[3]))
            elif len(line) > 1:
                line = line.split(' ')
                chanDict[line[1]] = int(line[3])
            line = f.readline()
    trial = unsortedSet(trial)
    assert len(trial) == 1
    channels = unsortedSet(channels)
    timepoints = unsortedSet(timepoints)
    shape = list(map(len, [channels, timepoints]))
    #voltage = np.array(voltage)
    voltage = np.reshape(voltage, shape, order='F')
    df = {}
    df['data'] = voltage
    df['channels'] = chanDict
    df['label'] = label
    df['trial'] = trial[0]
    df['condition'] = condition
    return df

def loadPerson(path):
    fnms = os.listdir(path)
    fnms.sort()
    data = []
    #channels = None
    conditions, trials = [], []
    for i in range(len(fnms)):
        subpath = os.path.join(path, fnms[i])
        tmp = loadOneFile(subpath)
        conditions.append(tmp['condition'])
        trials.append(tmp['trial'])
        data.append(tmp['data'])
        if i == 0:
            channels = tmp['channels']
            label = tmp['label']
        else:
            '''
            assert channels.keys() == tmp['channels'].keys(), \
                f"{subpath}: channels are not consistent with previous data"
            assert data[-1].shape[0] == tmp['data'].shape[0], \
                f"{subpath}: timepoints are not consistent with previous data"
            common_trials = set(trials.keys()).intersection(set(tmp['trials'].keys()))
            assert  len(common_trials) == set(), \
                f"{subpath}: non-unique trials ({common_trials:s})"
            #permutationArray = mapperDataChannels(channels, tmp['channels'])
            data.append(tmp['data'][:, permutationArray, :])
            trial_offset = len(trials.keys())
            for key in tmp['trials'].keys():
                trials[key] = tmp['trials'][key] + trial_offset
            '''
            assert channels == tmp['channels']
            assert label == tmp['label']
    assert len(trials) == len(set(trials))
    person = {}
    person['data'] = np.array(data)
    assert person['data'].ndim > 1
    person['trials'] = trials
    person['channels'] = channels
    person['label'] = label
    person['conditions'] = conditions
    return person

def loadDirectory(path, return_df=False, save_path=None):
    subdirs = os.listdir(path)
    subdirs = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), subdirs))
    subdirs = list(sorted(subdirs, key=io_tools.stringSplitByNumbers))
    df = []
    for i in range(len(subdirs)):
        subdir = subdirs[i]
        if subdir in _BAD_PERSONS:
            print(f"\r Filtered: {subdir}", end='')
            continue
        print(f'\r {subdir}: {(i+1)/len(subdirs)*100.:.3f}%', end='')
        df.append(loadPerson(os.path.join(path, subdir)))
    if save_path is not None:
        np.savez_compressed(save_path, df=df)
    if return_df:
        return df