MAX_NUM_THREADS = 4
likelihoods = ['bernoulli', 'normal']
guide_types = ['ppca', 'delta', 'diag_normal']
term_types = ['cpd', 'lro', 'tucker-core', 'tucker-factor', 'tt', 'qtt']

# random seeds
sklearn_random_state = 577
kmeans_random_state = 577

# metrics
ami_average = 'arithmetic'

# plot config
max_rank = 5

import argparse
########################### parsing command line arguments ###################
parser = argparse.ArgumentParser(description='VBTD experiment')
parser.add_argument(
    '--random_seed',
    action='store',
    dest='random_seed',
    type=int,
    default=0
)
parser.add_argument(
    '--num_threads',
    action='store',
    dest='num_threads',
    type=int,
    default=MAX_NUM_THREADS
)
parser.add_argument(
    '--likelihood',
    action='store',
    dest='likelihood',
    type=str
)
parser.add_argument(
    '--guide_type',
    action='store',
    dest='guide_type',
    type=str
)
parser.add_argument(
    '--isolate_group',
    action='store',
    dest='isolate_group',
    type=str,
    default='None'
)
parser.add_argument(
    '--orthogonolize_terms',
    action='store',
    dest='orthogonolize_terms',
    type=str,
    default='None'
)
parser.add_argument(
    '--normalize',
    action='store',
    dest='normalize',
    type=int,
    default=0
)
parser.add_argument(
    '--isotropic',
    action='store',
    dest='isotropic',
    type=int,
    default=0
)
parser.add_argument(
    '--highlight_peak',
    action='store',
    dest='highlight_peak',
    type=float,
    default=0
)
parser.add_argument(
    '--term_type',
    action='store',
    dest='term_type',
    type=str
)
parser.add_argument(
    '--group_term_type',
    action='store',
    dest='group_term_type',
    type=str,
    default=''
)
parser.add_argument(
    '--bias_type',
    action='store',
    dest='bias_type',
    type=str,
    default=''
)
parser.add_argument(
    '--group_bias_type',
    action='store',
    dest='group_bias_type',
    type=str,
    default=''
)
parser.add_argument(
    '--lr',
    action='store',
    dest='lr',
    type=float,
    default=0.005
)
parser.add_argument(
    '--num_epoch',
    action='store',
    dest='num_epoch',
    help='Number of epochs',
    type=int,
    default=100
)
parser.add_argument(
    '--device',
    action='store',
    dest='device',
    type=str,
    default='cpu'
)
parser.add_argument(
    '--batch_size',
    action='store',
    dest='batch_size',
    type=int
)
parser.add_argument(
    '--data_path',
    action='store',
    dest='data_path',
    type=str
)
parser.add_argument(
    '--results_path',
    action='store',
    dest='results_path',
    type=str
)
parser.add_argument(
    '--matrix_normalize',
    action='store',
    dest='matrix_normalize',
    type=int,
    default=1
)


input_args = parser.parse_args()

if input_args.highlight_peak == 0:
    input_args.highlight_peak = None
if len(input_args.isolate_group) > 0:
    input_args.isolate_group = eval(input_args.isolate_group)
else:
    input_args.isolate_group = None
if len(input_args.orthogonolize_terms) > 0:
    input_args.orthogonolize_terms = eval(input_args.orthogonolize_terms)
else:
    input_args.orthogonolize_terms = None
if len(input_args.group_term_type) == 0:
    input_args.group_term_type = None
if len(input_args.bias_type) == 0:
    input_args.bias_type = None
if len(input_args.group_bias_type) == 0:
    input_args.group_bias_type = None

    
input_args.normalize = bool(input_args.normalize)
input_args.isotropic = bool(input_args.isotropic)
input_args.matrix_normalize = bool(input_args.matrix_normalize)
assert input_args.likelihood in likelihoods
assert input_args.guide_type in guide_types
assert input_args.term_type in term_types
if input_args.group_term_type is not None:
    assert input_args.group_term_type in term_types
    
print('\t Current config')
    
print(f'Random seed: {input_args.random_seed}')
print(f'Data path: {input_args.data_path}')
print(f'Results path: {input_args.results_path}')
    
print(f'Batch size: {input_args.batch_size}')
print(f'Number of epochs: {input_args.num_epoch}')
print(f'Learning rate: {input_args.lr}')
print(f'Device: {input_args.device}')
print(f'Number of threads: {input_args.num_threads}')
    
print(f'Likelihood: {input_args.likelihood}')
print(f'Isotropic: {input_args.isotropic}')
print(f'Variational distribution: {input_args.guide_type}')
print(f'Isolate group: {str(input_args.isolate_group)}')
print(f'Orthogonolize terms: {str(input_args.orthogonolize_terms)}')
print(f'Normalize: {input_args.normalize}')
print(f'Highlight peak: {input_args.highlight_peak}')
print(f'Term type: {input_args.term_type}')
print(f'Group term type: {input_args.group_term_type}')
print(f'Term bias type: {input_args.bias_type}')
print(f'Group term bias type: {input_args.group_bias_type}')
print(f'Matrix normalize: {input_args.matrix_normalize}')
####################################################################################


import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
#print(mkl_rt.mkl_get_max_threads())
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(input_args.num_threads)
print(f'Number of threads was limited to {mkl_get_max_threads()}.')


import os
import logging

#import matplotlib.pyplot as plt
#from IPython.display import clear_output

import numpy as np

import pyro
import pyro.optim

import torch

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score

import vbtd_tools
import plot_tools
import vbtd
import learning_tools

import numpy_tools
import loaders.eth80 as eth80
#import loaders.smni_eeg as smni_eeg

# performance scores
usv_scores = {
    'ami': lambda x, y: adjusted_mutual_info_score(x, y, average_method=ami_average),
    'ari': adjusted_rand_score,
    'fmi': fowlkes_mallows_score    
}
optim_config = {'lr': input_args.lr}


# data_eth80_dirname = '/home/hariyuki/data/eth80/'
# load ETH80
# data_eth80_path = os.path.join(data_eth80_dirname, 'eth80-cropped-close128')

image_shape = 32
Nclasses, Nobjects = 8, 10
Npixels = int(round(image_shape**2))

data, labels, classes = eth80.eth80_dataset(input_args.data_path, image_shape)
classes_reverse_dict = dict((y, x) for x, y in classes.items())
Nsamples, Nangles, _, _, Ncolors = data.shape
data = numpy_tools.reshape_np(data, [Nangles, -1, Ncolors], order='F', use_batch=True)
permutation = [0, 2, 1, 3]
data = np.transpose(data, permutation)

n = [Npixels, Nangles, Ncolors]

tmp = numpy_tools.flatten_np(data)
if input_args.likelihood == 'bernoulli':
    tmp = tmp / 255
elif input_args.likelihood == 'normal':
    if input_args.matrix_normalize:
        tmp_mean = np.mean(tmp, axis=1, keepdims=True)
        tmp_std = np.std(tmp, axis=1, keepdims=True)
        tmp = (tmp - tmp_mean) / tmp_std
    else:
        axes = tuple(range(2, len(data.shape)))
        tmp_mean = np.mean(data, axis=axes, keepdims=True)
        tmp_std = np.std(data, axis=axes, keepdims=True)
        tmp = (data - tmp_mean) / tmp_std
else:
    raise ValueError

torch_data = torch.from_numpy(numpy_tools.reshape_np(tmp, n)).to(dtype=torch.float32)  

cp_term_config = {
    'type': 'cp',
    'R': 5
}
lro_term_config = {
    'type': 'lro',
    'L': [3],
    'P': 2
}
tucker_term_config1 = {
    'type': 'tucker',
    'r': [3, 3, 3],
}
tucker_term_config2 = {
    'type': 'tucker',
    'r': [3, 3, 3],
    'r0': 3
}
tt_config = {
    'type': 'tt',
    'r': [1]+[3]*(len(n)),
}

term_config_dict = {
    'cpd': cp_term_config,
    'lro': lro_term_config,
    'tucker-core': tucker_term_config1,
    'tucker-factor': tucker_term_config2,
    'tt': tt_config,
    'qtt': None
}



btd_terms_list = [
    {
        'tensor_config': term_config_dict[input_args.term_type],
        'bias': False if input_args.bias_type is None else True,
        'bias_config': None if input_args.bias_type is None else (
            term_config_dict[input_args.bias_type]
        )
    }
]*Nclasses

if input_args.group_term_type is None:
    gm_config = None
else:
    gm_config = {
        'tensor_config': term_config_dict[input_args.group_term_type],
        'bias': False if input_args.group_bias_type is None else True,
        'bias_config': None if input_args.group_bias_type is None else (
            term_config_dict[input_args.group_bias_type]
        )
    }
    
terms_config = {
    'isotropic': input_args.isotropic,
    'btd_terms': btd_terms_list,
    'btd_iterms': None,
}
group_term_config = None if gm_config is None else {
    'isotropic': input_args.isotropic,
    'btd_term': gm_config,
    'btd_iterm': None,
}
source_mode = 0

pyro.set_rng_seed(input_args.random_seed)
pyro.clear_param_store()
vbtd_model = vbtd.VariationalBTD(
    n,
    likelihood=input_args.likelihood,
    terms_config=terms_config,
    group_term_config=group_term_config,
    source_mode=source_mode
)
if input_args.guide_type in vbtd_model._auto_guide_types:
    vbtd_model.generate_auto_guide(input_args.guide_type)

#torch_data = torch_data.to(device=device, dtype=torch.float)
#vbtd_model = vbtd_model.to(device=device, dtype=torch.float)
#if normalize:
#    vbtd_model.normalize_parameters()

optim = pyro.optim.Adam(optim_config)

log_filename = 'vbtd'
log_filename += f"_terms-isotropic={terms_config['isotropic']:d}"
log_filename += f"_likelihood={input_args.likelihood}"
log_filename += f"_guide-type={input_args.guide_type}"
log_filename += f"_isolate-group={input_args.isolate_group}"
log_filename += f"_normalize={input_args.normalize}"
log_filename += f"_orthogonolize-terms={input_args.orthogonolize_terms}"
log_filename += f"_highlight-peak={input_args.highlight_peak}"
log_filename += f"_term-type={input_args.term_type}"
log_filename += f"_group-term-type={input_args.group_term_type}"
log_filename += f"_bias-type={input_args.bias_type}"
log_filename += f"_group-bias-type={input_args.group_bias_type}"
log_filename += f"_matrix-normalize={input_args.matrix_normalize}"

save_filename = f'{log_filename}.npz'
final_save_filename = f'final_{log_filename}.npz'
log_filename = f'{log_filename}.log'

results = learning_tools.svi_train_unsupervised_mixture_model(
    torch_data,
    mixture_model=vbtd_model,
    optimizer=optim,
    batch_size=input_args.batch_size,
    labels=labels,
    model_field='model_with',
    guide_field='guide_with',
    device=input_args.device,
    save_results_path=os.path.join(
        input_args.results_path,
        save_filename
    ),
    save_model_path=None,
    best_model_metric_name='ami',
    monitor_norms=True,
    usv_scores=usv_scores,
    display_plots=False,
    max_plate_nesting=1,
    maxitnum=input_args.num_epoch,
    highlight_peak=input_args.highlight_peak,#10*Nclasses
    log_filename=log_filename,
    normalize=input_args.normalize,
    isolate_group=input_args.isolate_group,
    orthogonolize_terms=input_args.orthogonolize_terms,
    expert=True
)
np.savez_compressed(
    os.path.join(
        input_args.results_path,
        final_save_filename
    ),
    results=results
)