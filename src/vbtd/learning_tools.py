import os
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
from IPython.display import clear_output

import numpy as np

import pyro
#import pyro.optim

import torch

import vbtd_tools
import plot_tools
#import vbtd

#from importlib import reload
#reload(vbtd_tools);
#reload(plot_tools);
#reload(vbtd);

#from sklearn.metrics import adjusted_mutual_info_score
#from sklearn.metrics import adjusted_rand_score
#from sklearn.metrics import fowlkes_mallows_score

import numpy_tools
#import loaders.eth80 as eth80
#import loaders.smni_eeg as smni_eeg


def svi_train_unsupervised_mixture_model(
    data,
    mixture_model,
    optimizer,
    batch_size=None,
    labels=None,
    model_field='model',
    guide_field='guide',
    device='cpu',
    save_results_path=None,
    save_model_path=None,
    best_model_metric_name=None,
    monitor_norms=False,
    usv_scores=None,
    display_plots=False,
    max_plate_nesting=1,
    maxitnum=1,
    highlight_peak=None,
    log_filename=None,
    normalize=False,
    isolate_group=None,
    orthogonolize_terms=None,
    expert=True,
    verbose=True
):          
    if log_filename is not None:
        logging.basicConfig(
            filename=log_filename,
            filemode='w+'
        )
        logger = logging.getLogger('Learning_log')
        logger.info(f'Starting learning session: {maxitnum+1} epochs\n')
    else:
        logger = None
    
    if usv_scores is not None:
        assert labels is not None
        scores_prior_lists, scores_posterior_lists = {}, {}
        for key in usv_scores:
            scores_prior_lists[key], scores_posterior_lists[key] = [], []
            
    if save_model_path is not None:
        if best_model_metric_name is not None:
            assert best_model_metric_name in usv_scores
            
    mixture_model = mixture_model.to(device)
    data = data.to(device)
    
    model = getattr(mixture_model, model_field)
    guide = getattr(mixture_model, guide_field)
    
    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    svi = pyro.infer.SVI(model, guide, optimizer, loss=elbo)
    
    sample_size = len(data)
    if batch_size is None:
        batch_size = sample_size
    
    # Register hooks to monitor gradient norms.
    if monitor_norms:
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    svi_losses = []
    max_pi_score = -np.inf

    for i in range(maxitnum):
        sind = np.random.permutation(sample_size)
        ind = 0
        current_loss_value = 0.
        local_i = 0
        while (ind < sample_size):
            local_i += 1
            cind = torch.from_numpy(
                sind[ind:ind+batch_size]
            ).to(device=device, dtype=torch.long)
            ind += batch_size
            loss = svi.step(
                data,
                labels=None,
                subsample_size=len(cind),
                subsample=cind,
                xi_greedy=None,
                #xi_greedy=(1. - (i+1)/maxitnum),#**2.,
                highlight_peak=highlight_peak,
                normalize=normalize,
                isolate_group=isolate_group,
                orthogonolize_terms=orthogonolize_terms,
                expert=expert
            )
            #print(pps['ppca_sigmas_p'])
            #vbtd.xi = #(i//10)/(1. + i//10)
            current_loss_value += float(loss)
        svi_losses.append(np.mean(current_loss_value))
        if display_plots:
            clear_output()
        if (labels is not None) and (usv_scores is not None):
            scores_prior, scores_posterior = vbtd_tools.measure_usv_performance(
                mixture_model,
                guide,
                data,
                labels,
                scores=usv_scores
            )
            prior_string = 'Prior:\t '
            posterior_string = 'Posterior:\t '
            for key in usv_scores:
                scores_prior_lists[key].append(scores_prior[key])
                prior_string += f'{key}={scores_prior[key]:.3f}\t'
                scores_posterior_lists[key].append(scores_posterior[key])
                posterior_string += f'{key}={scores_posterior[key]:.3f}\t'
            if verbose:
                print(f'{prior_string}\n{posterior_string}')
            if logger is not None:
                logger.warning(f'{prior_string}\n{posterior_string}')
            if best_model_metric_name is not None:
                if max_pi_score < scores_prior[best_model_metric_name]:
                    print('updating factors')
                    source_factors = vbtd_tools.get_pixel_factors(
                        mixture_model, mode=mixture_model.source_mode
                    )
                    max_pi_score = scores_prior[best_model_metric_name]
            else:
                source_factors = vbtd_tools.get_pixel_factors(
                    mixture_model, mode=mixture_model.source_mode
                )
            if display_plots:
                fig, ax = plot_tools.plot_performance_evolution(
                    svi_losses,
                    scores_prior_lists,
                    scores_posterior_lists,
                    colors=None
                )
            plt.show()
            if save_results_path is not None:
                np.savez_compressed(
                    save_results_path,
                    scores_prior_lists=scores_prior_lists,
                    scores_posterior_lists=scores_posterior_lists,
                    source_factors=source_factors,
                    svi_losses=svi_losses
                )
        else:
            if save_results_path is not None:
                np.savez_compressed(
                    save_results_path,
                    scores_prior_lists=None,
                    scores_posterior_lists=None,
                    source_factors=source_factors,
                    svi_losses=svi_losses
                )
        if display_plots:
            fig, ax = plot_tools.plot_pixel_factors(
                mixture_model,
                [image_shape, image_shape],
                max_rank=max_rank,
                mode=mixture_model.source_mode
            )
            plt.show()
        if verbose:
            print(f'\rSVI loss={svi_losses[-1]:.3e}')
        if logger is not None:
            logger.warning(f'SVI loss={svi_losses[-1]:.3e}')
        #cur_param = dict(pyro.get_param_store().named_parameters())
        #for x in interesting_params:
        #    print(x, cur_param[x])
        
    mixture_model = mixture_model.to('cpu')
    data = data.to('cpu')
    result = [svi_losses]
    if (labels is not None) and (usv_scores is not None):
        result += [scores_prior_lists, scores_posterior_lists]
    if monitor_norms:
        result += [gradient_norms]
    
    return result

def svi_train_smni_unsupervised_mixture_model(
    data,
    mixture_model,
    optimizer,
    batch_size=None,
    labels=None,
    model_field='model',
    guide_field='guide',
    device='cpu',
    save_results_path=None,
    save_model_path=None,
    best_model_metric_name=None,
    monitor_norms=False,
    usv_scores=None,
    display_plots=False,
    max_plate_nesting=1,
    maxitnum=1,
    highlight_peak=None,
    log_filename=None,
    normalize=False,
    isolate_group=None,
    orthogonolize_terms=None,
    expert=True,
    verbose=True
):          
    if log_filename is not None:
        logging.basicConfig(
            filename=log_filename,
            filemode='w+'
        )
        logger = logging.getLogger('Learning_log')
        logger.info(f'Starting learning session: {maxitnum+1} epochs\n')
    else:
        logger = None
    
    if usv_scores is not None:
        assert labels is not None
        scores_prior_lists, scores_posterior_lists = {}, {}
        for key in usv_scores:
            scores_prior_lists[key], scores_posterior_lists[key] = [], []
            
    if save_model_path is not None:
        if best_model_metric_name is not None:
            assert best_model_metric_name in usv_scores
            
    mixture_model = mixture_model.to(device)
    data = data.to(device)
    
    model = getattr(mixture_model, model_field)
    guide = getattr(mixture_model, guide_field)
    
    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    svi = pyro.infer.SVI(model, guide, optimizer, loss=elbo)
    
    sample_size = len(data)
    if batch_size is None:
        batch_size = sample_size
    
    # Register hooks to monitor gradient norms.
    if monitor_norms:
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    svi_losses = []
    max_pi_score = -np.inf

    for i in range(maxitnum):
        sind = np.random.permutation(sample_size)
        ind = 0
        current_loss_value = 0.
        local_i = 0
        while (ind < sample_size):
            local_i += 1
            cind = torch.from_numpy(
                sind[ind:ind+batch_size]
            ).to(device=device, dtype=torch.long)
            ind += batch_size
            loss = svi.step(
                data,
                labels=None,
                subsample_size=len(cind),
                subsample=cind,
                xi_greedy=None,
                #xi_greedy=(1. - (i+1)/maxitnum),#**2.,
                highlight_peak=highlight_peak,
                normalize=normalize,
                isolate_group=isolate_group,
                orthogonolize_terms=orthogonolize_terms,
                expert=expert
            )
            #print(pps['ppca_sigmas_p'])
            #vbtd.xi = #(i//10)/(1. + i//10)
            current_loss_value += float(loss)
        svi_losses.append(np.mean(current_loss_value))
        if display_plots:
            clear_output()
        if (labels is not None) and (usv_scores is not None):
            scores_prior, scores_posterior = vbtd_tools.measure_usv_performance(
                mixture_model,
                guide,
                data,
                labels,
                scores=usv_scores
            )
            prior_string = 'Prior:\t '
            posterior_string = 'Posterior:\t '
            for key in usv_scores:
                scores_prior_lists[key].append(scores_prior[key])
                prior_string += f'{key}={scores_prior[key]:.3f}\t'
                scores_posterior_lists[key].append(scores_posterior[key])
                posterior_string += f'{key}={scores_posterior[key]:.3f}\t'
            if verbose:
                print(f'{prior_string}\n{posterior_string}')
            if logger is not None:
                logger.warning(f'{prior_string}\n{posterior_string}')
            if best_model_metric_name is not None:
                if max_pi_score < scores_prior[best_model_metric_name]:
                    print('updating factors')
                    source_factors = vbtd_tools.get_pixel_factors(
                        mixture_model, mode=mixture_model.source_mode
                    )
                    max_pi_score = scores_prior[best_model_metric_name]
            else:
                source_factors = vbtd_tools.get_pixel_factors(
                    mixture_model, mode=mixture_model.source_mode
                )
            if display_plots:
                fig, ax = plot_tools.plot_performance_evolution(
                    svi_losses,
                    scores_prior_lists,
                    scores_posterior_lists,
                    colors=None
                )
            plt.show()
            if save_results_path is not None:
                np.savez_compressed(
                    save_results_path,
                    scores_prior_lists=scores_prior_lists,
                    scores_posterior_lists=scores_posterior_lists,
                    source_factors=source_factors,
                    svi_losses=svi_losses
                )
        else:
            if save_results_path is not None:
                np.savez_compressed(
                    save_results_path,
                    scores_prior_lists=None,
                    scores_posterior_lists=None,
                    source_factors=source_factors,
                    svi_losses=svi_losses
                )
        if display_plots:
            fig, ax = plot_tools.plot_pixel_factors(
                mixture_model,
                [image_shape, image_shape],
                max_rank=max_rank,
                mode=mixture_model.source_mode
            )
            plt.show()
        if verbose:
            print(f'\rSVI loss={svi_losses[-1]:.3e}')
        if logger is not None:
            logger.warning(f'SVI loss={svi_losses[-1]:.3e}')
        #cur_param = dict(pyro.get_param_store().named_parameters())
        #for x in interesting_params:
        #    print(x, cur_param[x])
        
    mixture_model = mixture_model.to('cpu')
    data = data.to('cpu')
    result = [svi_losses]
    if (labels is not None) and (usv_scores is not None):
        result += [scores_prior_lists, scores_posterior_lists]
    if monitor_norms:
        result += [gradient_norms]
    
    return result