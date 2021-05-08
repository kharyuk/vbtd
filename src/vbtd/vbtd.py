import torch
import torch.nn as nn
from torch.distributions import constraints

import numpy as np

import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import config_enumerate, infer_discrete
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal

import torch_utils
import tensorial


_likelihoods = [
    'normal', 'bernoulli'
]

_default_terms_config = {
    'isotropic': True,
    'btd_terms': [tensorial._default_cp_term_config],
    'btd_iterms': None,
}

_default_group_term_config = {
    'isotropic': True,
    'btd_term': tensorial._default_cp_term_config,
    'btd_iterm': None,
}

class VariationalBTD(nn.Module):
    _pvariables = ['ppca_gm_sigma', 'ppca_sigmas', 'zk_sigma_', 'zg_sigma']
    _auto_guide_types = ['delta', 'diag_normal']
    _likelihoods = ['normal', 'bernoulli']
    def __init__(
        self,
        n,
        likelihood='normal',
        terms_config=None,
        group_term_config=None,
        source_mode=0
    ):
        super(VariationalBTD, self).__init__()
        
        assert likelihood in _likelihoods
        self.likelihood = likelihood
        
        assert terms_config is not None
        self.terms_isotropic = terms_config['isotropic']
        
        self.n = n
        self.d = len(n)
        self.K = len(terms_config['btd_terms'])
        self.output_dim = int(round(np.prod(self.n)))
        self.source_mode = source_mode
        
        # \sum T_k(\theta_k) <=> \sum_k W_k z_k
        self.terms = nn.ModuleList()
        self.hidden_dims = []
        for k in range(self.K):
            self.terms.append(tensorial.TensorizedLinear(n, **terms_config['btd_terms'][k]))
            self.hidden_dims.append(self.terms[k].linear_mapping.hidden_dim())
        self.hidden_dims = np.array(self.hidden_dims)
        
        # inverse: x -> z_1, \ldots, z_K
        if terms_config['btd_iterms'] is not None:
            self.iterms = nn.ModuleList()
            for k in range(self.K):
                self.iterms.append(tensorial.TensorizedLinear(n, **terms_config['btd_iterms'][k]))
        else:
            self.register_parameter('iterms', None)
        
        # W_g z_g and inverse transformation; in simple case there is only one term
        if group_term_config is not None:
            self.group_term = tensorial.TensorizedLinear(n, **group_term_config['btd_term'])
            self.group_hidden_dim = self.group_term.linear_mapping.hidden_dim()
            self.group_isotropic = group_term_config['isotropic']
            if group_term_config['btd_iterm'] is not None:
                self.group_iterm = tensorial.TensorizedLinear(n, **group_term_config['btd_iterm'])
            else:
                self.register_parameter('group_iterm', None)
        else:
            self.register_parameter('group_term', None)
            self.register_parameter('group_iterm', None)
        
    
    def isolate_group_factors(self, mode):
        assert self.group_term is not None
        if isinstance(mode, int):
            mode = [mode]
        for m in mode:
            if isinstance(self.group_term.linear_mapping, tensorial.TTTensor):
                tmp = self.group_term.linear_mapping.cores[m].permute([1, 0, 2])
                tmp = torch_utils.reshape_torch(tmp, [tmp.shape[0], -1], order='F', use_batch=False)
                u, _, _ = torch_utils.fast_svd_torch(tmp)
            else:
                u, _, _ = torch_utils.fast_svd_torch(self.group_term.linear_mapping.factors[m])
            for k in range(self.K):
                if isinstance(self.terms[k].linear_mapping, tensorial.TTTensor):
                    tmp = self.terms[k].linear_mapping.cores[m].permute([1, 0, 2])
                    tmp_shape = list(tmp.shape)
                    tmp = torch_utils.reshape_torch(tmp, [tmp_shape[0], -1], order='F', use_batch=False)
                    tmp -= torch.mm(u, torch.mm(u.t(), tmp))
                    tmp = torch_utils.reshape_torch(tmp, tmp_shape, order='F', use_batch=False)
                    self.terms[k].linear_mapping.cores[m].data = tmp.permute([1, 0, 2])
                else:
                    self.terms[k].linear_mapping.factors[m].data = (
                        self.terms[k].linear_mapping.factors[m] -
                        torch.mm(u, torch.mm(u.t(), self.terms[k].linear_mapping.factors[m]))
                    )
                
    def orthogonolize_k_factors(self, mode):
        if isinstance(mode, int):
            mode = [mode]
        for m in mode:
            for k in range(self.K):
                if isinstance(self.terms[k].linear_mapping, tensorial.TTTensor):
                    tmp = self.terms[k].linear_mapping.cores[m].permute([1, 0, 2])
                    tmp = torch_utils.reshape_torch(tmp, [tmp.shape[0], -1], order='F', use_batch=False)
                    uk, _, _ = torch_utils.fast_svd_torch(tmp)
                else:
                    uk, _, _ = torch_utils.fast_svd_torch(self.terms[k].linear_mapping.factors[m])
                for l in range(k+1, self.K):
                    if isinstance(self.terms[l].linear_mapping, tensorial.TTTensor):
                        tmp = self.terms[l].linear_mapping.cores[m].permute([1, 0, 2])
                        tmp_shape = list(tmp.shape)
                        tmp = torch_utils.reshape_torch(tmp, [tmp_shape[0], -1], order='F', use_batch=False)
                        tmp -= torch.mm(uk, torch.mm(uk.t(), tmp))
                        tmp = torch_utils.reshape_torch(tmp, tmp_shape, order='F', use_batch=False)
                        self.terms[l].linear_mapping.cores[m].data = tmp.permute([1, 0, 2])
                    else:
                        self.terms[l].linear_mapping.factors[m].data = (
                            self.terms[l].linear_mapping.factors[m] -
                            torch.mm(uk, torch.mm(uk.t(), self.terms[l].linear_mapping.factors[m]))
                        )
        
    def measure_principal_angles(self, input_batch, mode, fast=True):
        batch_size = input_batch.shape[0]
        mu_k = []
        for k in range(self.K):
            tmp = self.terms[k].get_bias(tensorize_output=False, use_batch=True)
            if tmp is None:
                tmp = input_batch.new_zeros(1, self.output_dim)
            mu_k.append( tmp )
        
        mu_k = torch.cat(mu_k)
        
        if self.group_term is not None:
            mu_g = self.group_term.get_bias(tensorize_output=False, use_batch=True)
            if mu_g is None:
                mu_g = input_batch.new_zeros(1, self.output_dim)
            projected_batch = self.group_term.multi_project(
                input_batch, remove_bias=False, tensorize=False
            )
            projected_mu = mu_k + mu_g
            projected_mu -= self.group_term.multi_project(
                torch_utils.reshape_torch(projected_mu, self.n),
                remove_bias=False,
                tensorize=False
            )
        #'''
        input_batch = torch_utils.flatten_torch(input_batch)
        output_angles = input_batch.new_zeros([batch_size, self.K])
        if fast:
            chi = 1.2
        for k in range(self.K):
            if fast:
                Uk, _, _ = torch_utils.fast_svd_torch(
                    #self.terms[k].linear_mapping.factors[mode]
                    self.terms[k].get_sources(self.source_mode),
                    chi=chi
                )
            else:
                #Uk, _, _ = torch.svd(self.terms[k].linear_mapping.factors[mode])
                Uk, _, _ = torch.svd(
                    self.terms[k].get_sources(self.source_mode)
                )
                Uk = Uk[:, :self.terms[k].linear_mapping.r[mode]]
            Uk = Uk.t()
            '''
            if self.group_term is not None:
                current_batch = input_batch - projected_batch - projected_mu[k:k+1, :]
            else:
                current_batch = input_batch - mu_k[k:k+1, :]
            '''
            current_batch = torch_utils.reshape_torch(input_batch, self.n) # current
            current_batch = torch_utils.swapaxes_torch(current_batch, 1, mode+1)
            current_batch = torch_utils.reshape_torch(current_batch, [self.n[mode], -1])
            tmp_r = current_batch.shape[-1]
            for i in range(batch_size):
                if fast:
                    u, _, _ = torch_utils.fast_svd_torch(current_batch[i], chi=chi)
                else:
                    u, _, _ = torch.svd(current_batch[i])
                _, s, _ = torch.svd(Uk.mm(u[:, :tmp_r]))
                output_angles[i, k] = s[0]
        return output_angles
            
    def normalize_parameters(self):
        if self.group_term is not None:
            self.group_term.linear_mapping.normalize()
        for k in range(self.K):
            self.terms[k].linear_mapping.normalize()
        
    def transform(self, z_batch, angles):
        batch_size = z_batch.shape[1] # !!
        output = z_batch.new_zeros([self.K, batch_size, self.output_dim])
        for k in range(self.K):
            output[k] = self.terms[k](
                z_batch[k, :, :self.hidden_dims[k]] * angles[:, k].view(-1, 1)
            )
        return output
    
    def itransform(self, x_batch, ppca_sigmas):
        batch_size = x_batch.shape[0]
        max_hidden_dim = np.max(self.hidden_dims)   
        output_mean = x_batch.new_zeros([self.K, batch_size, max_hidden_dim])
        output_cov = x_batch.new_zeros([self.K, max_hidden_dim, max_hidden_dim])
        for k in range(self.K):
            z_mean, z_cov = (
                self.terms[k].get_posterior_gaussian_mean_covariance(
                    x_batch, noise_sigma=ppca_sigmas[k]
                )
            )
            output_mean[k] += z_mean            
            output_cov[k] += z_cov[0]
        output_cov = output_cov.unsqueeze(1)
        return output_mean, output_cov
        
        
    def predict(
        self,
        input_batch,
        probas=False,
        guide=None,
        temperature=0.,
        fast=True,
        expert=False,
        model_method_name='model_with'
    ):
        batch_size = input_batch.shape[0]
        output_angles = self.measure_principal_angles(input_batch, mode=self.source_mode, fast=fast)
        output_angles = output_angles.softmax(dim=-1)
        if not probas:
            output_angles = output_angles.argmax(dim=-1)
        if guide is not None:
            #guide_trace = poutine.trace(self.guide_for).get_trace(input_batch, None, None, expert=expert)
            guide_trace = poutine.trace(guide).get_trace(input_batch)
            trained_model = poutine.replay(getattr(self, model_method_name), trace=guide_trace)
            inferred_model = infer_discrete(
                trained_model, temperature=temperature, first_available_dim=-2 #######################
            )
            trace = poutine.trace(inferred_model).get_trace(input_batch)
            return output_angles, trace.nodes["assignments"]["value"]
        return output_angles
    
    def module_ppca_means_sigmas_weights(
        self,
        input_batch,
        epsilon,
        expert=False,
        highlight_peak=None
    ):
        # zk = zk_mean ##+ eps*alpha_k, eps \sim N(0, I)
        # ppca_mean = Wk zk, ppca_sigma = sigma_k^2
        max_hidden_dim = max(self.hidden_dims)
        batch_size = input_batch.shape[0]
        if expert:
            pi = self.measure_principal_angles(input_batch, mode=self.source_mode)
            if highlight_peak is not None:
                pi = highlight_peak*pi
            pi = pi.softmax(dim=-1)
        else:
            pi = pyro.sample(
                'pi',
                dist.Dirichlet(input_batch.new_ones(self.K)/self.K)
            )
            if highlight_peak is not None:
                pi = highlight_peak*pi.log()#dim=-1)
                pi = pi.softmax(dim=-1)
                
        #zk_mean = input_batch.new_zeros([batch_size, self.K, max_hidden_dim])
        ppca_means = input_batch.new_zeros([self.K, batch_size, self.output_dim])
        if self.likelihood == 'normal':
            with pyro.plate('ppca_sigma_plate', self.K):
                '''
                ppca_sigmas_p = pyro.param(
                    f'ppca_sigmas_p',
                    input_batch.new_ones(self.K),
                    #constraint=constraints.interval(1e-3, 2.)
                    constraint=constraints.positive
                )'''
                if self.terms_isotropic:
                    ppca_sigmas = pyro.sample(
                        f'ppca_sigmas',
                        #dist.Delta(ppca_sigmas_p)#.independent(1)
                        #dist.LogNormal(0, ppca_sigmas_p)#.independent(1)
                        dist.LogNormal(0, input_batch.new_ones(self.K, 1)).independent(1)
                        #dist.Delta(input_batch.new_ones(self.K, 1)).independent(1)
                    )
                else:
                    ppca_sigmas_list = []
                    ppca_sigmas = input_batch.new_ones(self.K, 1)
                    for i in range(self.d):
                        ppca_sigmas_list.append(
                            pyro.sample(
                                f'ppca_sigmas_{i}',
                                #dist.Delta(ppca_sigmas_p)#.independent(1)
                                #dist.LogNormal(0, ppca_sigmas_p)#.independent(1)
                                dist.LogNormal(0, input_batch.new_ones(self.K, self.n[i])).independent(1)
                                #dist.Delta(input_batch.new_ones(self.K, self.n[i])).independent(1)
                            )
                        )
                        ppca_sigmas = torch_utils.krp_cw_torch(ppca_sigmas_list[i], ppca_sigmas, column=False)
                    
        '''
        alpha_p = pyro.param(
            f'alpha_p',
            input_batch.new_ones([self.K, max_hidden_dim]),
            constraint=constraints.positive
        )''' 
        with pyro.plate('alpha_plate', self.K):
            alpha = pyro.sample(
                f'alpha',
                #dist.LogNormal(0, alpha_p).independent(1)
                #dist.Delta(alpha_p).independent(1)
                dist.LogNormal(0, input_batch.new_ones([self.K, max_hidden_dim])).independent(1)
                #dist.Delta(input_batch.new_ones([self.K, max_hidden_dim])).independent(1)
            )#'''
        #alpha = input_batch.new_ones([self.K, max_hidden_dim])
        for k in range(self.K):
            #zk_mean[:, k, :self.hidden_dims[k]] = self.terms[k].linear_mapping.inverse_batch(input_batch, fast=True)
            #zk_mean[:, k, :self.hidden_dims[k]] += epsilon[:, :self.hidden_dims[k]]*alpha[k:k+1, :self.hidden_dims[k]]
            if self.iterms is None:
                #zk_mean = self.terms[k].linear_mapping.inverse_batch(input_batch)
                ppca_means[k, :, :] += self.terms[k].multi_project(input_batch, remove_bias=False, tensorize=False)
            else:
                #zk_mean = self.iterms[k](torch_utils.flatten_torch(input_batch), T=True)
                ppca_means[k, :, :] += self.terms[k](self.iterms[k](torch_utils.flatten_torch(input_batch), T=True))
            
            ppca_means[k, :, :] += self.terms[k](
                epsilon[:, :self.hidden_dims[k]]*alpha[k:k+1, :self.hidden_dims[k]]
            )
        if self.likelihood == 'bernoulli':
            return pi, ppca_means.sigmoid()
        if self.likelihood == 'normal':
            return pi, ppca_means, ppca_sigmas
        raise ValueError
        
    def module_ppca_means_sigmas_weights_guide(
        self,
        input_batch,
        epsilon,
        expert=False,
        xi_greedy=None,
        highlight_peak=None
    ):
        # zk = zk_mean ##+ eps*alpha_k, eps \sim N(0, I)
        # ppca_mean = Wk zk, ppca_sigma = sigma_k^2
        max_hidden_dim = max(self.hidden_dims)
        batch_size = input_batch.shape[0]
        
        gamma = input_batch.new_zeros(batch_size, self.K)
        if expert:
            output_angles = self.measure_principal_angles(input_batch, mode=self.source_mode)
        else:
            pi_p = pyro.param(
                'pi_p',
                input_batch.new_ones(self.K)/self.K,
                constraint=constraints.positive
            )
            output_angles = pyro.sample(
                'pi',
                dist.Dirichlet(pi_p),
            ).log()
        if highlight_peak is not None:
            output_angles = highlight_peak*output_angles
        output_angles = output_angles.log_softmax(dim=-1)
        
        if xi_greedy is not None:
            phi = dist.LogNormal(
                input_batch.new_zeros([batch_size, self.K]),
                input_batch.new_ones([batch_size, self.K])
            ).to_event(1).sample()
            output_angles = (1-xi_greedy)*output_angles + xi_greedy*phi
            
        #output_angles /= output_angles.sum(dim=-1, keepdim=True)
        #output_angles = output_angles.log()
        ppca_means = input_batch.new_zeros([self.K, batch_size, self.output_dim])
        if self.likelihood == 'normal':
            with pyro.plate('ppca_sigma_plate', self.K):
                if self.terms_isotropic:
                    ppca_sigmas_p = pyro.param(
                        f'ppca_sigmas_p',
                        input_batch.new_ones(self.K, 1),
                        #constraint=constraints.interval(1e-6, 10.)
                        constraint=constraints.positive
                    )
                    ppca_sigmas = pyro.sample(
                        f'ppca_sigmas',
                        dist.Delta(ppca_sigmas_p).independent(1)
                        #dist.LogNormal(0, ppca_sigmas_p)#.independent(1)
                    )
                else:
                    ppca_sigmas = input_batch.new_ones(self.K, 1)
                    ppca_sigmas_list = []
                    for i in range(self.d):
                        ppca_sigmas_p = pyro.param(
                            f'ppca_sigmas_{i}_p',
                            input_batch.new_ones(self.K, self.n[i]),
                            #constraint=constraints.interval(1e-6, 10.)
                            constraint=constraints.positive
                        )
                        ppca_sigmas_list.append(
                            pyro.sample(
                                f'ppca_sigmas_{i}',
                                dist.Delta(ppca_sigmas_p).independent(1)
                                #dist.LogNormal(0, ppca_sigmas_p)#.independent(1)
                            )
                        )
                        ppca_sigmas = torch_utils.krp_cw_torch(ppca_sigmas_list[i], ppca_sigmas, column=False)
                #'''
        else:
            ppca_sigmas = None
            ppca_sigmas_list = [input_batch.new_ones(self.K, self.n[i]) for i in range(self.d)]
        with pyro.plate('alpha_plate', self.K):
            alpha_p = pyro.param(
                f'alpha_p',
                input_batch.new_ones([self.K, max_hidden_dim]),
                constraint=constraints.positive
                #constraint=constraints.interval(1e-6, 10.)
            )
            alpha = pyro.sample(
                f'alpha',
                dist.Delta(alpha_p).independent(1)
                #dist.LogNormal(0, alpha_p).independent(1)
            )#'''
            #alpha = input_batch.new_ones([self.K, max_hidden_dim])
        
        
        for k in range(self.K):
            if self.iterms is None:
                z_mu = self.terms[k].linear_mapping.inverse_batch(input_batch)
            else:
                z_mu = self.iterms[k](torch_utils.flatten_torch(input_batch), T=True) 
            if self.terms_isotropic:
                zk_mean, zk_cov = self.terms[k].get_posterior_gaussian_mean_covariance(
                    input_batch,
                    noise_sigma=ppca_sigmas[k] if ppca_sigmas is not None else 1,
                    z_mu=z_mu,
                    z_sigma=alpha[k]
                )
            else:
                zk_mean, zk_cov = self.terms[k].get_posterior_gaussian_mean_covariance(
                    input_batch, noise_sigma=[x[k] for x in ppca_sigmas_list], z_mu=z_mu, z_sigma=alpha[k]
                )
            ppca_means[k, :, :] = self.terms[k](
                zk_mean + epsilon[:, :self.hidden_dims[k]].mm(
                    zk_cov.view(self.hidden_dims[k], self.hidden_dims[k])
                )
            )
            if self.likelihood == 'bernoulli':
                gamma[:, k] = dist.Bernoulli(
                    ppca_means[k].sigmoid(),
                    validate_args=False
                ).to_event(1).log_prob(torch_utils.flatten_torch(input_batch))
            elif self.likelihood == 'normal':
                gamma[:, k] = dist.Normal(
                    loc=ppca_means[k],
                    scale=ppca_sigmas[k]
                ).to_event(1).log_prob(torch_utils.flatten_torch(input_batch))
            else:
                raise ValueError
        gamma = (output_angles + gamma).softmax(dim=-1)
        #gamma_l = 0.999
        #gamma = gamma_l*gamma+(1.-gamma_l)*np.ones([1, self.K])/self.K
        '''
        pps = pyro.get_param_store()
        Nk = gamma.sum(dim=0)
        tmp1 = input_batch.new_zeros([self.K, max_hidden_dim])
        tmp2 = input_batch.new_zeros(self.K)
        for k in range(self.K):
            tmp1[k] = (
                gamma[:, k:k+1]*(
                    zk_mean + epsilon[:, :self.hidden_dims[k]].mm(
                        zk_cov.view(self.hidden_dims[k], self.hidden_dims[k])
                    )**2.
                )
            ).sum(dim=0) / Nk[k]
            tmp2[k] = (
                (gamma[:, k]*torch.norm(torch_utils.flatten_torch(input_batch) - ppca_means[k], dim=1)**2.).sum()
            ) / Nk[k]
        pname = 'alpha_p'
        pps.replace_param(pname, tmp1, pps[pname])
        pname = 'ppca_sigmas_p'
        pps.replace_param(pname, tmp2, pps[pname])
        '''
        return gamma, ppca_means, ppca_sigmas
        #return output_angles.softmax(dim=-1), ppca_means, ppca_sigmas
    
    def module_ppca_gm_means_sigma(self, input_batch, epsilon):
        max_hidden_dim = max(self.hidden_dims)
        batch_size = input_batch.shape[0]
        ppca_means = input_batch.new_zeros([self.K, batch_size, self.output_dim])
        if self.likelihood == 'normal':
            #with pyro.plate('ppca_sigma_plate', self.K):
            if self.group_isotropic:
                ppca_gm_sigma = pyro.sample(
                    f'ppca_gm_sigma',
                    dist.LogNormal(0, input_batch.new_ones(1, 1)).independent(1)
                )
            else:
                ppca_gm_sigma_list = []
                ppca_gm_sigma = input_batch.new_ones(1, 1)
                for i in range(self.d):
                    ppca_gm_sigma_list.append(
                        pyro.sample(
                            f'ppca_gm_sigma_{i}',
                            dist.LogNormal(0, input_batch.new_ones(1, self.n[i])).independent(1)
                        )
                    )
                    ppca_gm_sigma = torch_utils.krp_cw_torch(ppca_gm_sigma_list[i], ppca_gm_sigma, column=False)
        
        #with pyro.plate('alpha_plate', self.K):        
        alpha_gm = pyro.sample(
            f'alpha_gm',
            dist.LogNormal(0, input_batch.new_ones([1, self.group_hidden_dim])).independent(1)
        )
        
        if self.group_iterm is None:
            ppca_gm_means = self.group_term.multi_project(
                input_batch, remove_bias=False, tensorize=False #fast_svd=False
            )
        else:
            ppca_gm_means = self.group_term(
                self.group_iterm(
                    torch_utils.flatten_torch(input_batch), T=True#, fast_svd=False
                )
            )
            
        ppca_gm_means += self.group_term(
            epsilon[:, :self.group_hidden_dim]*alpha_gm[:, :self.group_hidden_dim]
        )
        
        if self.likelihood == 'bernoulli':
            return ppca_gm_means.sigmoid()
        if self.likelihood == 'normal':
            return ppca_gm_means, ppca_gm_sigma
        raise ValueError
        
            
    def module_ppca_gm_means_sigma_guide(self, input_batch, epsilon):
        batch_size = input_batch.shape[0]
        if self.likelihood == 'normal':
            if self.group_isotropic:
                ppca_gm_sigma_p = pyro.param(
                    f'ppca_gm_sigma_p',
                    input_batch.new_ones(1, 1),
                    constraint=constraints.positive
                )
                ppca_gm_sigma = pyro.sample(
                    f'ppca_gm_sigma',
                    dist.Delta(ppca_gm_sigma_p).independent(1)
                )
            else:
                ppca_gm_sigma = input_batch.new_ones(1, 1)
                ppca_gm_sigma_list = []
                for i in range(self.d):
                    ppca_gm_sigma_p = pyro.param(
                        f'ppca_gm_sigma_{i}_p',
                        input_batch.new_ones(1, self.n[i]),
                        constraint=constraints.positive
                    )
                    ppca_gm_sigma_list.append(
                        pyro.sample(
                            f'ppca_gm_sigma_{i}',
                            dist.Delta(ppca_gm_sigma_p).independent(1)
                        )
                    )
                    ppca_gm_sigma = torch_utils.krp_cw_torch(ppca_gm_sigma_list[i], ppca_gm_sigma, column=False)
        else:
            ppca_gm_sigma = input_batch.new_ones(1, 1)
            ppca_gm_sigma_list = [input_batch.new_ones(1, self.n[i]) for i in range(self.d)]
            
        alpha_gm_p = pyro.param(
            f'alpha_gm_p',
            input_batch.new_ones([1, self.group_hidden_dim])
        )
        alpha_gm = pyro.sample(
            f'alpha_gm',
            dist.Delta(alpha_gm_p).independent(1)
        )
        
        if self.group_iterm is None:
            z_mu = self.group_term.linear_mapping.inverse_batch(input_batch)
        else:
            z_mu = self.group_iterm(torch_utils.flatten_torch(input_batch), T=True) 
        if self.group_isotropic:
            zk_mean, zk_cov = self.group_term.get_posterior_gaussian_mean_covariance(
                input_batch,
                noise_sigma=ppca_gm_sigma[0] if ppca_gm_sigma is not None else input_batch.new_ones(1),
                z_mu=z_mu,
                z_sigma=alpha_gm[0]
            )
        else:
            zk_mean, zk_cov = self.group_term.get_posterior_gaussian_mean_covariance(
                input_batch,
                noise_sigma=[x for x in ppca_gm_sigma_list],
                z_mu=z_mu,
                z_sigma=alpha_gm[0]
            )
        ppca_gm_means = self.group_term(
            zk_mean + epsilon[:, :self.group_hidden_dim].mm(
                zk_cov.view(self.group_hidden_dim, self.group_hidden_dim)
            )
        )
        return ppca_gm_means, ppca_gm_sigma    
    
    @config_enumerate(default='parallel')
    def model_with(
        self,
        input_batch,
        labels=None,
        subsample_size=None,
        subsample=None,
        xi_greedy=None,
        highlight_peak=None,
        normalize=False,
        isolate_group=None,
        orthogonolize_terms=None,
        expert=False
    ):
        pyro.module('terms', self.terms)
        if self.iterms is not None:
            pyro.module('iterms', self.iterms)
        batch_size = input_batch.shape[0]
        
        if labels is not None:
            assert len(labels) == batch_size
        if subsample_size is None:
            subsample_size = batch_size
            if subsample is None:
                subsample = torch.arange(subsample_size)
        max_hidden_dim = max(self.hidden_dims)
        if self.group_term is not None:
            max_hidden_dim = max(max_hidden_dim, self.group_hidden_dim)
        with pyro.plate('epsilon_plate', subsample_size):
            epsilon = pyro.sample(
                'epsilon',
                dist.Normal(input_batch.new_zeros(subsample_size, max_hidden_dim), 1.).independent(1)
            )
        if self.group_term is not None:
            pyro.module('group_term', self.group_term)
            if self.likelihood == 'bernoulli':
                ppca_gm_means = self.module_ppca_gm_means_sigma(
                    input_batch[subsample], epsilon#[subsample]
                )
            elif self.likelihood == 'normal':
                ppca_gm_means, ppca_gm_sigma = self.module_ppca_gm_means_sigma(
                    input_batch[subsample], epsilon#[subsample]
                )
            else:
                raise ValueError
        
        if self.likelihood == 'bernoulli':
            pi, ppca_means = self.module_ppca_means_sigmas_weights(
                input_batch[subsample],
                epsilon,#[subsample],
                expert=expert,
                highlight_peak=highlight_peak
            )
        elif self.likelihood == 'normal':
            pi, ppca_means, ppca_sigmas = self.module_ppca_means_sigmas_weights(
                input_batch[subsample],
                epsilon,#[subsample],
                expert=expert,
                highlight_peak=highlight_peak
            )
        else:
            raise ValueError
            
        #print(ppca_means)
        with pyro.plate(
            f'samples',
            batch_size,
            subsample_size=subsample_size,
            subsample=subsample,
            device=input_batch.device
        ) as i:
            #print(i, ppca_means.shape, ppca_sigmas.shape, ppca_gm_means.shape, ppca_gm_sigma.shape)
            assignments = pyro.sample('assignments', dist.Categorical(pi))
            if self.likelihood == 'normal':
                if assignments.dim() == 1:
                    if self.group_term is None:
                        pyro.sample(
                            f'obs',
                            dist.Normal(
                                ppca_means[assignments, torch.arange(subsample_size), :],
                                ppca_sigmas[assignments]
                            #).independent(1),
                            ).to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
                    else:
                        pyro.sample(
                            f'obs',
                            dist.Normal(
                                (
                                    ppca_means + torch_utils.reshape_torch(
                                        ppca_gm_means, [1, subsample_size, -1], use_batch=False
                                    )
                                )[assignments, torch.arange(batch_size), :],
                                (
                                    torch_utils.reshape_torch(
                                        ppca_sigmas, [self.K, -1], use_batch=False
                                    ) + ppca_gm_sigma[0])[assignments]#.view(-1, 1)
                            ).independent(1),#to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
                else:
                    if self.group_term is None:
                        pyro.sample(
                            f'obs',
                            dist.Normal(
                                ppca_means[assignments, :, :][:, 0],
                                ppca_sigmas[assignments].view(self.K, 1, -1)
                            #).independent(1),
                            ).to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
                    else:
                        pyro.sample(
                            f'obs',
                            dist.Normal(
                                (
                                    ppca_means + torch_utils.reshape_torch(
                                        ppca_gm_means, [1, subsample_size, -1], use_batch=False
                                    )
                                )[assignments, :, :][:, 0],
                                torch_utils.reshape_torch(
                                    (ppca_sigmas.view(self.K, -1) + ppca_gm_sigma)[assignments],
                                    [self.K, 1, -1],
                                    use_batch=False
                                )
                            ).independent(1),#to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
            elif self.likelihood == 'bernoulli':
                if assignments.dim() == 1:
                    if self.group_term is None:
                        pyro.sample(
                            f'obs',
                            dist.Bernoulli(
                                ppca_means[assignments, torch.arange(subsample_size), :],
                                validate_args=False
                            ).to_event(1),#to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
                    else:
                        pyro.sample(
                            f'obs',
                            dist.Bernoulli(
                                (
                                    ppca_means + torch_utils.reshape_torch(
                                        ppca_gm_means, [1, subsample_size, -1], use_batch=False
                                    )
                                )[assignments, torch.arange(batch_size), :],
                                validate_args=False
                            ).to_event(1),#to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
                else:
                    if self.group_term is None:
                        pyro.sample(
                            f'obs',
                            dist.Bernoulli(
                                ppca_means[assignments, :, :][:, 0],
                                validate_args=False
                            ).to_event(1),#to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
                    else:
                        pyro.sample(
                            f'obs',
                            dist.Bernoulli(
                                (
                                    ppca_means + torch_utils.reshape_torch(
                                        ppca_gm_means, [1, subsample_size, -1], use_batch=False
                                    )
                                )[assignments, :, :][:, 0],
                                validate_args=False
                            ).to_event(1),#to_event(1),
                            obs=torch_utils.flatten_torch(input_batch[i])#*output_angles[:, k].view(-1, 1)
                        )
            else:
                raise ValueError
                
                
    @config_enumerate(default='parallel')
    def guide_with(
        self,
        input_batch,
        labels=None,
        subsample_size=None,
        subsample=None,
        xi_greedy=None,
        highlight_peak=None,
        normalize=False,
        isolate_group=None,
        orthogonolize_terms=None,
        expert=False
    ):
        pyro.module('terms', self.terms)
        if self.iterms is not None:
            pyro.module('iterms', self.iterms)
        batch_size = input_batch.shape[0]
        if labels is not None:
            assert len(labels) == batch_size
        if subsample_size is None:
            subsample_size = batch_size
            if subsample is None:
                subsample = torch.arange(subsample_size)
        
        max_hidden_dim = max(self.hidden_dims)
        if self.group_term is not None:
            max_hidden_dim = max(max_hidden_dim, self.group_hidden_dim)
        with pyro.plate('epsilon_plate', subsample_size):
            epsilon = pyro.sample(
                'epsilon',
                dist.Normal(input_batch.new_zeros(subsample_size, max_hidden_dim), 1.).independent(1)
            )
        
        if self.group_term is not None:
            pyro.module('group_term', self.group_term)
            if self.training and (isolate_group is not None):
                self.isolate_group_factors(mode=isolate_group)
            ppca_gm_means, ppca_gm_sigma = self.module_ppca_gm_means_sigma_guide(
                input_batch[subsample], epsilon#[subsample]
            )
        if self.training and (orthogonolize_terms is not None):
            self.orthogonolize_k_factors(mode=orthogonolize_terms)
        if self.training and normalize:
            with torch.no_grad():
                self.normalize_parameters()
        if self.group_term is not None:
            gamma, ppca_means, ppca_sigmas = self.module_ppca_means_sigmas_weights_guide(
                torch_utils.flatten_torch(input_batch)[subsample]-dist.Normal(
                    ppca_gm_means,
                    ppca_gm_sigma
                ).sample(),
                epsilon,#[subsample]
                expert=expert,
                xi_greedy=xi_greedy,
                highlight_peak=highlight_peak
            )
        else:
            gamma, ppca_means, ppca_sigmas = self.module_ppca_means_sigmas_weights_guide(
                input_batch[subsample],
                epsilon,#[subsample],
                expert=expert,
                xi_greedy=xi_greedy,
                highlight_peak=highlight_peak
            )
        with pyro.plate(
            f'samples',
            batch_size,
            subsample_size=subsample_size,
            subsample=subsample,
            device=input_batch.device
        ):
            assignments = pyro.sample('assignments', dist.Categorical(gamma))
                
    def generate_auto_guide(
        self,
        guide_type='delta',
        model_method_name='model_with',
        guide_method_name='guide_with'
    ):
        assert guide_type in self._auto_guide_types
        exposed_variables = ['epsilon', 'alpha', 'pi']
        if self.likelihood == 'normal':
            if self.terms_isotropic:
                exposed_variables += ['ppca_sigmas']
            else:
                exposed_variables += [f'ppca_sigma_{i}' for i in range(self.d)]
            if self.group_term is not None:
                if self.group_isotropic:
                    exposed_variables += ['ppca_gm_sigma']
                else:
                    exposed_variables += [f'ppca_gm_sigma_{i}' for i in range(self.d)]
        if guide_type == 'delta':
            guide = AutoDelta(
                poutine.block(getattr(self, model_method_name), expose=exposed_variables)
            )
        elif guide_type == 'diag_normal':
            guide = AutoDiagonalNormal(
                poutine.block(getattr(self, model_method_name), expose=exposed_variables)
            )
        else:
            raise ValueError
        #setattr(self, guide_method_name, classmethod(guide))
        setattr(self, guide_method_name, guide)
        
    def initialize_seed(self, svi, input_batch, subsample_size=None, seed=None):
        raise NotImplemented
        if seed is not None:
            pyro.set_rng_seed(seed)
        pyro.clear_param_store()  
        loss = svi.loss(model.model_for, model.guide_for, input_batch)
        return loss
    