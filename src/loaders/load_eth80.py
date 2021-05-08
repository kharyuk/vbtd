import numpy as _np
import urllib as _urllib
import tarfile as _tarfile
import os as _os
import copy as _copy
import imageio as _imageio
from skimage.transform import resize as _resize
# local imports
import io_tools as _io_work




def downloadETH80(save_data_dirname=''):
    url = 'http://datasets.d2.mpi-inf.mpg.de/eth80/eth80-cropped-close128.tgz'
    save_data_filename = 'eth80-cropped-close128.tgz'
    _urllib.urlretrieve(url, save_data_dirname+save_data_filename)
    with _tarfile.open(save_data_dirname+save_data_filename, 'r:gz') as f:
        f.extractall(save_data_dirname)
        
def buildNpzETH80(data_filename, data_dirname='', resizeValue=None):
    Nobjects = 10 # fixed values
    Nclasses = 8 # fixed value
    data_dirname_new = data_dirname+data_filename.split('.')[0] + '/'
    all_filenames = _os.listdir(data_dirname_new)
    all_filenames = sorted(all_filenames, key=_io_work.stringSplitByNumbers)
    data = []
    classes = []
    for i in xrange(Nclasses):
        locDirnames = all_filenames[i*Nobjects:(i+1)*Nobjects]
        class_data = []
        className = filter(lambda x: not x.isdigit(), locDirnames[0])
        classes.append(className)
        for j in xrange(Nobjects):
            fnms = _os.listdir(data_dirname_new+locDirnames[j])
            fnms = filter(lambda x: x.endswith('.png'), fnms)
            fnms = sorted(fnms, key=_io_work.stringSplitByNumbers)
            object_data = []
            for k in xrange(len(fnms)):
                local_dirname_lv2 = data_dirname_new+locDirnames[j] + '/'
                tmp = _imageio.imread(local_dirname_lv2+fnms[k])
                if resizeValue is not None:
                    newShape = [resizeValue]*2
                    tmp = _resize(tmp, newShape, mode='constant')
                object_data.append(tmp.copy())
            class_data.append(_copy.deepcopy(object_data))
        data.append(_copy.deepcopy(class_data))
    data = _np.array(data)
    if data.max() > 1.:
        data = data / 255.
    return data, classes
    
if __name__ == '__main__':
    pass
