import numpy as np

import os
import requests
#import tqdm
import re
import copy

import PIL.Image as Image
import urllib.request
import tarfile
from . import io_tools

def downloadETH80(save_data_dirname=None):
    save_data_filename = 'eth80-cropped-close128.tgz'
    if save_data_dirname is None:
        save_data_filename = ''
    if not os.path.exists(save_data_dirname):
        os.makedirs(save_data_dirname, exist_ok=True)
    path = os.path.join(save_data_dirname, save_data_filename)
    url = 'http://datasets.d2.mpi-inf.mpg.de/eth80/eth80-cropped-close128.tgz'
    urllib.request.urlretrieve(url, path)
    with tarfile.open(path, 'r:gz') as f:
        f.extractall(save_data_dirname)
    os.remove(path)
    
def eth80_dataset(data_dirname, image_shape=32):
    Nobjects = 10 # fixed values
    Nclasses = 8 # fixed value
    subdirnames = os.listdir(data_dirname)
    subdirnames = sorted(subdirnames, key=io_tools.stringSplitByNumbers)
    classes = {}
    data, labels = [], []
    for i in range(Nclasses):
        class_dirnames = subdirnames[i*Nobjects : (i+1)*Nobjects]
        class_name = re.findall('[a-z]+', class_dirnames[0])[0]
        classes[class_name] = i
        labels += [i]*Nobjects
        for j in range(Nobjects):
            current_path = os.path.join(data_dirname, class_dirnames[j])
            filenames = os.listdir(current_path)
            filenames = list(
                filter(lambda x: os.path.isfile(os.path.join(current_path, x)), filenames)
            )
            filenames = sorted(filenames, key=io_tools.stringSplitByNumbers)
            object_data = []
            for k in range(len(filenames)):
                image = Image.open(os.path.join(current_path, filenames[k]))
                if image_shape is not None:
                    if isinstance(image_shape, int):
                        image = image.resize([image_shape]*2, Image.ANTIALIAS)
                    else:
                        image = image.resize(image_shape, Image.ANTIALIAS)
                object_data.append(np.array(image))
                del image
            data.append(object_data)
    data = np.array(data)
    return data, labels, classes
