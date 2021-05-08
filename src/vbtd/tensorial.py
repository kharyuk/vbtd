import torch
import torch.nn as nn
import numpy as np
import torch_utils

_classes = ['TensorizedLinear', 'CPTensor', 'TuckerTensor', 'LROTensor', 'TTTensor']
_functions = ['configure_tensor_mixing']

_default_cp_term_config = {
    'type': 'cp',
    'R': 3
}

_default_lro_term_config = {
    'type': 'lro',
    'L': [3, 3, 3],
    'P': 2
}

_default_tucker_term_latent_core_config = {
    'type': 'tucker',
    'r': [9, 3, 3],
}

_default_tucker_term_latent_factor_config = {
    'type': 'tucker',
    'r': [3, 3, 3],
    'r0': 3
}

_default_tt_term_config = {
    'type': 'tt',
    'r': [3, 3, 3]
}


def configure_tensor_mixing(n, tensor_config, tensor_types, mixing):
    assert tensor_config['type'] in tensor_types
    if tensor_config['type'] == 'cp':
        mapping = CPTensor(n, tensor_config, sample_axis=mixing)
    elif tensor_config['type'] == 'lro':
        mapping = LROTensor(n, tensor_config, sample_axis=mixing)
    elif tensor_config['type'] == 'tucker':
        mapping = TuckerTensor(n, tensor_config, sample_axis=mixing)
    elif tensor_config['type'] == 'tt':
        mapping = TTTensor(n, tensor_config, sample_axis=mixing)
    else:
        raise ValueError
    return mapping
    
class TensorizedLinear(nn.Module):
    _tensor_types = ('cp', 'lro', 'tucker', 'tt')
    def __init__(self, n, tensor_config, bias=True, bias_config=None):#, nonlinearity=None):
        super(TensorizedLinear, self).__init__()
        
        self.n = n
        self.d = len(n)
        self.N = int(round(np.prod(self.n)))
        
        self.linear_mapping = configure_tensor_mixing(
            self.n, tensor_config, self._tensor_types, mixing=True
        )
        
        if bias:
            if (bias_config is None):
                self.bias = nn.Parameter(
                    data=torch.Tensor(self.N),
                    requires_grad=True
                )
                self.bias.data.uniform_(-0.03, 0.03)
            else:
                self.bias = configure_tensor_mixing(
                    self.n, bias_config, self._tensor_types, mixing=False
                )
        else:
            self.register_parameter('bias', None)
        
        #if nonlinearity is not None:
        #    self.nonlinearity = nonlinearity()
        #else:
        #    self.register_parameter('nonlinearity', None)
        
    def get_bias(self, tensorize_output=True, use_batch=True):
        if self.bias is None:
            return None
        if callable(self.bias):
            result = self.bias()
        else:
            result = self.bias.clone()
        shape = []
        if tensorize_output:
            shape += self.linear_mapping.n
        else:
            shape += [-1]
        if use_batch:
            shape = [1] + shape
        if len(shape) != result.dim():
            result = torch_utils.reshape_torch(result, shape, use_batch=False)
        return result
        
    def forward(self, input, T=False):
        output = self.linear_mapping(input, T)
        if self.bias is not None:
            if callable(self.bias):
                if T:
                    output += self.bias().t()
                else:
                    output += self.bias()
            else:
                if T:
                    output += self.bias.t()
                else:
                    output += self.bias
        #if self.nonlinearity is not None:
        #    output = self.nonlinearity(output)
        return output
    
    def get_posterior_gaussian_mean_covariance(self, x_batch, noise_sigma=1, z_mu=0., z_sigma=1):
        if self.bias is not None:
            if callable(self.bias):
                output_mean = x_batch - torch_utils.reshape_torch(self.bias(), [1]+self.n, use_batch=False)
            else:
                output_mean = x_batch - torch_utils.reshape_torch(self.bias, [1]+self.n, use_batch=False)
            output_mean = torch_utils.reshape_torch(output_mean, self.linear_mapping.n)
        else:
            output_mean = torch_utils.reshape_torch(x_batch, self.linear_mapping.n)
        #output_mean = torch.mean(output_mean, dim=0, keepdim=True)
        if isinstance(self.linear_mapping, TuckerTensor):
            if not isinstance(noise_sigma, list):
                svds = self.linear_mapping.get_svds()
            else:
                svds = self.linear_mapping.get_svds(weights=[x.sqrt() for x in noise_sigma])
            if svds[-1]:
                U, P, L, Q, _ = svds
                S_cov = Q*L.t()
                for k in range(self.d):
                    output_mean = torch_utils.prodTenMat_torch(output_mean, U[k], k+1, 0)
                output_mean = torch_utils.flatten_torch(output_mean)
                #output_mean = torch.mm(output_mean, P*L.t())
                output_mean = output_mean.mm(L*P.t())
                #output_mean = torch.mm(output_mean, Q.t())
                output_mean = output_mean.mm(Q)
            else:
                U, S, V, shapes_s, _ = svds
                S_cov = x_batch.new_ones([1, 1])
                for k in range(self.d):
                    S_cov = torch_utils.kron_torch(S_cov, V[k]*S[k].t())
                    output_mean = torch_utils.prodTenMat_torch(output_mean, U[k]*S[k].t(), k+1, 0)
                    output_mean = torch_utils.prodTenMat_torch(output_mean, V[k], k+1, 1)
        elif (
            isinstance(self.linear_mapping, CPTensor) or
            isinstance(self.linear_mapping, LROTensor)
        ):
            if not isinstance(noise_sigma, list):
                U, S, V = self.linear_mapping.get_svds(coupled=True)
            else:
                U, S, V = self.linear_mapping.get_svds(weights=[x.sqrt() for x in noise_sigma], coupled=True)
            S_cov = V*S.t()
            output_mean = torch_utils.flatten_torch(output_mean)
            output_mean = output_mean.mm(U*S)
            output_mean = output_mean.mm(V.t())
        elif isinstance(self.linear_mapping, TTTensor):
            #S_cov = x_batch.new_ones([1, 1])
            for k in range(self.d):
                shape = [self.n[k], self.linear_mapping.r[k+1]]
                tmp = self.linear_mapping.cores[k]
                if isinstance(noise_sigma, list):
                    tmp = tmp * noise_sigma[k].sqrt().view(1, -1, 1)
                if k > 0:
                    tmp = torch.einsum('ij,iab,jac->bc', S_cov, tmp, tmp)
                else:
                    tmp = torch.einsum('aib,aic->bc', tmp, tmp)
                #tmp = torch_utils.reshape_torch(tmp, shape, use_batch=False)
                #E, V = torch.eig(tmp, eigenvectors=True)
                #S_cov = (V*E[:, :1].t()).mm(V.t())
                u, s, v = torch.svd(tmp)
                S_cov = (u/s).mm(v.t())
                
                shape = [self.linear_mapping.r[k]*self.n[k], -1]
                tmp = self.linear_mapping.cores[k]
                output_mean = torch_utils.reshape_torch(output_mean, shape)
                output_mean = torch.einsum(
                    'ijk,jl->ilk', output_mean, torch_utils.reshape_torch(tmp, shape, use_batch=False)
                )
        else:
            raise ValueError
        if not isinstance(noise_sigma, list):
            try:
                S_cov = S_cov/np.sqrt(noise_sigma)
            except:
                S_cov = S_cov/noise_sigma.sqrt()
        if not isinstance(self.linear_mapping, TTTensor):
            S_cov = S_cov.mm(S_cov.t())
        n = S_cov.shape[0]
        mask = torch.eye(n, n, device=x_batch.device).byte()
        S_cov[mask] += 1./z_sigma
        u, s, v = torch.svd(S_cov)
        S_cov = (u/s).mm(v.t())
        #E, V = torch.eig(S_cov, eigenvectors=True)
        #S_cov = (V/E[:, :1].t()).mm(V.t())
        output_mean = torch_utils.flatten_torch(output_mean)
        output_mean = output_mean + z_mu / z_sigma ###
        output_mean = output_mean.mm(S_cov)
        S_cov = S_cov.unsqueeze(0)
        return output_mean, S_cov
    
    def multi_project(self, input_batch, remove_bias=True, tensorize=False):
        if remove_bias and (self.bias is not None):
            if callable(self.bias):
                output_batch = input_batch - torch_utils.reshape_torch(self.bias(), [1]+self.n, use_batch=False)
            else:
                output_batch = input_batch - torch_utils.reshape_torch(self.bias, [1]+self.n, use_batch=False)
            output_batch = torch_utils.reshape_torch(output_batch, self.linear_mapping.n)
        else:
            output_batch = torch_utils.reshape_torch(input_batch, self.linear_mapping.n)
            
        if isinstance(self.linear_mapping, TuckerTensor):
            svds = self.linear_mapping.get_svds()
            if svds[-1]:
                U, P, _, _, _ = svds
                for k in range(self.d):
                    output_batch = torch_utils.prodTenMat_torch(output_batch, U[k], k+1, 0)
                output_batch = torch_utils.flatten_torch(output_batch)
                output_batch = output_batch.mm(P.t())
                output_batch = output_batch.mm(P)
                output_batch = torch_utils.reshape_torch(output_batch, self.linear_mapping.r)
                for k in range(self.d):
                    output_batch = torch_utils.prodTenMat_torch(output_batch, U[k], k+1, 1)
            else:
                U, _, _, _, _ = svds
                for k in range(self.d):
                    output_batch = torch_utils.prodTenMat_torch(output_batch, U[k], k+1, 0)
                    output_batch = torch_utils.prodTenMat_torch(output_batch, U[k], k+1, 1)
        elif (
            isinstance(self.linear_mapping, CPTensor) or
            isinstance(self.linear_mapping, LROTensor)
        ):
            U, _, _ = self.linear_mapping.get_svds(coupled=True)
            output_batch = torch_utils.flatten_torch(output_batch)
            output_batch = output_batch.mm(U)
            output_batch = output_batch.mm(U.t())
        elif isinstance(self.linear_mapping, TTTensor):
            orth_list = []
            output_batch = input_batch.clone()
            for k in range(self.d):
                if k > 0:
                    tmp = torch_utils.prodTenMat_torch(
                        self.linear_mapping.cores[k],
                        tmp,
                        0,
                        1
                    )
                else:
                    tmp = self.linear_mapping.cores[k]
                tmp = torch_utils.reshape_torch(tmp, [-1, self.linear_mapping.r[k+1]], use_batch=False)
                tmp = torch_utils.reshape_torch(
                    self.linear_mapping.cores[k], [-1, self.linear_mapping.r[k+1]], use_batch=False
                )
                u, s, v = torch.svd(tmp)
                orth_list.append(u[:, :self.linear_mapping.r[k+1]])
                tmp = s*(v[:, :self.linear_mapping.r[k+1]].t())
                output_batch = torch_utils.reshape_torch(
                    output_batch,
                    [self.linear_mapping.r[k]*self.linear_mapping.n[k], -1],
                    use_batch=True
                )
                output_batch = torch_utils.prodTenMat_torch(output_batch, u, 1, 0)
            u, s, v = torch.svd(tmp)
            output_batch = output_batch.squeeze(2).mm(u).mm(u.t()) ##### ??? 
            for k in range(self.d-1, -1, -1):
                if k == self.d-1:
                    output_batch = output_batch.mm(orth_list[k].t())
                else:
                    output_batch = torch_utils.prodTenMat_torch(output_batch, orth_list[k], self.d-k, 1)
                output_batch = torch_utils.reshape_torch(
                    output_batch, self.n[k:]+[self.linear_mapping.r[k]], use_batch=True
                )
        else:
            raise ValueError
        if (tensorize) and (output_batch.dim() == 2):
            return torch_utils.reshape_torch(output_batch, self.linear_mapping.n)
        if (not tensorize) and (output_batch.dim() > 2):
            return torch_utils.flatten_torch(output_batch)
        return output_batch
    
    def get_sources(self, mode):
        if isinstance(mode, int):
            mode = [mode]
        else:
            assert (np.diff(mode) == 1).all()
            #if isinstance(self.linear_mapping, TTTensor):
            #    assert (np.diff(mode) == 1).all()
            #elif isinstance(self.linear_mapping, (TuckerTensor, CPTensor, LROTensor)):
            #    assert (np.diff(mode) > 0).all()
        if isinstance(self.linear_mapping, CPTensor):
            result = self.linear_mapping.factors[0].new_ones([1, self.linear_mapping.R])
        elif isinstance(self.linear_mapping, LROTensor):
            result = self.linear_mapping.factors[0].new_ones([1, self.linear_mapping.M])
        elif isinstance(self.linear_mapping, TuckerTensor):
            result = self.linear_mapping.factors[0].new_ones([1, 1])
            # use_core case?
        elif isinstance(self.linear_mapping, TTTensor):
            result = self.linear_mapping.cores[0].new_ones([1, 1, self.linear_mapping.r[0]])
        else:
            raise ValueError
        for i in range(len(mode)):
            m = mode[i]
            assert 0 <= m < self.d
            if isinstance(self.linear_mapping, CPTensor):
                result = torch_utils.krp_cw_torch(self.linear_mapping.factors[m], result)
            elif isinstance(self.linear_mapping, LROTensor):
                tmp = self.linear_mapping.factors[m]
                if m >= self.linear_mapping.P:
                    tmp = torch.repeat_interleave(
                        tmp, torch.tensor(self.linear_mapping.L, device=self.factors[0].device), dim=1
                    )
                result = torch_utils.krp_cw_torch(tmp, result)
            elif isinstance(self.linear_mapping, TTTensor):
                result = torch.einsum('ijk,klm->ijlm', result, self.linear_mapping.cores[m])
                r1, n1, n2, r2 = result.shape
                result = torch_utils.reshape_torch(result, [r1, n1*n2, r2], use_batch=False)
            elif isinstance(self.linear_mapping, TuckerTensor):
                tmp = self.linear_mapping.factors[m]
                result = torch_utils.kron_torch(tmp, result)
            else:
                raise ValueError
        if isinstance(self.linear_mapping, TTTensor):
            result = torch_utils.swapunfold_torch(result, 1, use_batch=False)
        return result
        


class CPTensor(nn.Module):
    def __init__(self, n, cp_config, sample_axis=False):
        super(CPTensor, self).__init__()
        # hyperparameters
        self.R = cp_config['R']
        self.n = n
        self.d = len(n)
        self.N = int(round(np.prod(self.n)))
        self.sample_axis = sample_axis
        
        # parameters
        self.factors = nn.ParameterList()
        for k in range(self.d):
            self.factors.append(
                nn.Parameter(
                    data=torch.Tensor(self.n[k], self.R),
                    requires_grad=True
                )
            )
            if 'initializer' in cp_config:
                cp_config['initializer'](self.factors[k])
            else:
                self.factors[k].data.normal_()#-0.03, 0.03)
    
    def hidden_dim(self):
        if self.sample_axis:
            return self.R
        return 0
    
    def set_parameters(self, U):
        assert len(U) == self.d
        for k in range(self.d):
            if U[k] is None:
                continue
            m, n = U[k].shape
            assert (m, n) == (self.n[k], self.R)
            if isinstance(U[k], np.ndarray):
                self.factors[k].data = torch.from_numpy(U[k]).to(
                    device=self.factors[k].device,
                    dtype=self.factors[k].dtype
                )
            elif isinstance(U[k], torch.Tensor):
                self.factors[k].data = U[k].to(
                    device=self.factors[k].device,
                    dtype=self.factors[k].dtype
                )
            else:
                raise ValueError
    
    def recover(self, weights=None, tensorize_output=False):
        nrows = self.N
        if not self.sample_axis:
            nrows = nrows // self.n[0]
        output = self.factors[0].new_zeros(nrows, self.R)
        for r in range(self.R):
            tmp = self.factors[0].new_ones(1)
            for k in range(self.d-1, -1, -1):
                if (k == 0) and (not self.sample_axis):
                    break
                tmp = torch.ger(tmp, self.factors[k][:, r])
                tmp = tmp.view([-1]).contiguous()
                if weights is not None:
                    tmp *= weights[k][r]
            output[:, r] = tmp
        if not self.sample_axis:
            output = self.factors[0].mm(output.t())
            output = torch_utils.flatten_torch(output, use_batch=False)
        return output
    
    def get_svds(self, weights=None, coupled=True, fast=True):
        if self.sample_axis and coupled:
            W = self.recover(weights)
            if fast:
                _, S, V = torch.svd(W.t().mm(W))
                S = S.sqrt()
                U = W.mm(V/S)
            else:
                U, S, V = torch.svd(W)
                U, S, V = U[:, :self.R], S[:self.R], V[:, :self.R]
        else:
            U, S, V = []
            for k in range(self.d):
                if weights is None:
                    u, s, v = torch.svd(self.factors[k])
                else:
                    u, s, v = torch.svd(weights[k]*self.factors[k])
                u, s, v = u[:, :self.R], s[:self.R], v[:, :self.R]
                U.append(u)
                S.append(s)
                V.append(v)
        return U, S, V
    
    def normalize(self):
        for k in range(self.d):
            self.factors[k].data = self.factors[k].data / torch.norm(self.factors[k].data, p='fro', dim=0)
    
    def forward(self, input=None, T=False, tensorize_output=False):
        assert (input is None) != self.sample_axis
        output = self.recover()
        if T:
            output = output.t()
        if self.sample_axis:
            # batch_size = input.shape[0]
            output = torch.einsum('ij,kj->ik', input, output)
        if tensorize_output:
            if T:
                return torch_utils.reshape_torch(output, self.n, use_batch=self.sample_axis)
            else:
                return torch_utils.reshape_torch(output, [self.R]*self.d, use_batch=self.sample_axis)
        return output    
    
    def inverse_batch(self, input_batch):        
        output = torch_utils.flatten_torch(input_batch).mm(self.recover())
        M = input_batch.new_ones([self.R, self.R])
        for k in range(self.d):
            M *= self.factors[k].t().mm(self.factors[k])
        U, S, V = torch.svd(M)
        output = output.mm(U/S).mm(V.t())
        return output                
            

class TuckerTensor(nn.Module):
    def __init__(self, n, tucker_config, sample_axis=False):
        super(TuckerTensor, self).__init__()
        # hyperparameters
        self.sample_axis = sample_axis
        self.n = n
        self.d = len(n)
        self.r = tucker_config['r']
        if self.sample_axis:
            if 'r0' in tucker_config:
                self.use_core = True
                self.r0 = tucker_config['r0']
            else:
                self.use_core = False
                self.r0 = None
        else:
            self.use_core = True
            self.r0 = None
        
        # parameters
        self.factors = nn.ParameterList()
        for k in range(self.d):
            self.factors.append(
                nn.Parameter(
                    data=torch.Tensor(self.n[k], self.r[k]),
                    requires_grad=True
                )
            )
            if 'initializer' in tucker_config:
                tucker_config['initializer'](self.factors[k])
            else:
                self.factors[k].data.normal_()#uniform_(-0.03, 0.03)
        if self.use_core:
            if self.sample_axis:
                shape = np.append(self.r, self.r0).tolist()
            else:
                shape = self.r
            self.core = nn.Parameter(
                data=torch.Tensor(size=shape),
                requires_grad=True
            )
            if 'initializer' in tucker_config:
                tucker_config['initializer'](self.core)
            else:
                self.core.data.normal_()#uniform_(-0.03, 0.03)
        else:
            self.register_parameter('core', None)

    def hidden_dim(self):
        if self.sample_axis:
            if self.use_core:
                return self.r0
            else:
                return int(round(np.prod(self.r)))
        return None
    
    def set_parameters(self, A=None, G=None):
        if A is not None:
            assert len(A) == self.d
            for k in range(self.d):
                if A[k] is None:
                    continue
                m, n = A[k].shape
                assert (m, n) == (self.n[k], self.r[k])
                if isinstance(A[k], np.ndarray):
                    self.factors[k].data = torch.from_numpy(A[k]).to(
                        device=self.factors[k].device,
                        dtype=self.factors[k].dtype
                    )
                elif isinstance(A[k], torch.Tensor):
                    self.factors[k].data = A[k].to(
                        device=self.factors[k].device,
                        dtype=self.factors[k].dtype
                    )
                else:
                    raise ValueError
        if G is not None:
            assert self.use_core
            gshape = list(map(lambda x: x, G.shape))
            if self.r0 is None:
                assert gshape == self.r
            else:
                assert gshape == [self.r0]+self.r
            if isinstance(G, np.ndarray):
                self.core.data = torch.from_numpy(G).to(
                    device=self.core.device,
                    dtype=self.factors[k].dtype
                )
            elif isinstance(G, torch.Tensor):
                self.core.data = G.to(
                    device=self.core.device,
                    dtype=self.factors[k].dtype
                )
            else:
                raise ValueError
            
                
    
    def get_svds(self, weights=None, fast=True):
        if fast:
            chi = 1.2
        core_flag = self.core is not None
        if core_flag:
            assert self.r0 is not None
            G = self.core.clone()
        U, S, V = [], [], []
        for k in range(self.d):
            if fast and (self.n[k] / self.r[k] >= chi):
                _, s, v = torch.svd(torch.mm(self.factors[k].t(), self.factors[k]))
                s = s.sqrt()
                u = self.factors[k].mm(v/s)
            else:
                u, s, v = torch.svd(self.factors[k])
            U.append(u[:, :self.r[k]])
            S.append(s[:self.r[k]].view(-1, 1))
            V.append(v[:, :self.r[k]])
            if core_flag:
                G = torch_utils.prodTenMat_torch(G, V[k]*S[k].t(), k, 0)
        if core_flag:
            permutation = [self.d] + list(range(self.d))
            G = G.permute(permutation).contiguous()
            rp = int(np.prod(self.r))
            tmp = torch_utils.reshape_torch(G, [self.r0, -1], use_batch=False)
            if fast and (self.r0 / rp >= chi):
                P, L, _ = torch.svd(tmp.t().mm(tmp))
                L = L.sqrt()
                Q = tmp.mm(P/L)
            elif fast and (rp / self.r0 >= chi):
                Q, L, _ = torch.svd(tmp.mm(tmp.t()))
                L = L.sqrt()
                P = torch.mm((Q / L).t(), tmp)
            else:
                P, L, Q = torch.svd(tmp.t())
                r = min(rp, self.r0)
                P, L, Q = P[:, :r], L[:r].view(-1, 1), Q[:, :r]
            return U, P, L, Q, core_flag
        return U, S, V, self.r, core_flag
        
    
    def recover(self, weights=None):
        if self.use_core:
            output = self.core.clone()
            for k in range(self.d):
                if weights is None:
                    output = torch_utils.prodTenMat_torch(output, self.factors[k], k, matrix_axis=1)
                else:
                    output = torch_utils.prodTenMat_torch(
                        output, self.factors[k]*weights[k].view(self.n[k], 1), k, matrix_axis=1
                    )
            if not self.sample_axis:
                output = torch_utils.flatten_torch(output, use_batch=False)
            else:
                output = torch_utils.flatten_torch(output.t(), use_batch=True).t()
        else:
            # very inefficient to do so
            output = self.factors[0].new_ones([1, 1])
            for k in range(self.d-1, -1, -1):
                if weights is None:
                    output = torch_utils.kron_torch(output, self.factors[k])
                else:
                    output = torch_utils.kron_torch(output, self.factors[k]*weights[k])
        return output
    
    def forward(self, input=None, T=False, tensorize_output=False):
        assert (input is None) != self.sample_axis
        # batch_size = input.shape[0]
        if self.use_core:
            output = self.core.clone()
        else:
            if T:
                output = torch_utils.reshape_torch(input, self.n)
            else:
                output = torch_utils.reshape_torch(input, self.r)
        offset = int((not self.use_core) and self.sample_axis)
        for k in range(self.d):
            output = torch_utils.prodTenMat_torch(
                output, self.factors[k], k+offset, matrix_axis=int(not T)
            )
        if self.use_core and self.sample_axis:
            output = torch_utils.prodTenMat_torch(output, input, self.d, matrix_axis=1)
            permutation = [self.d] + list(range(self.d))
            output = output.permute(permutation)
        if not tensorize_output:
            return torch_utils.flatten_torch(output, use_batch=self.sample_axis)
        return output
    
    def normalize(self):
        if self.use_core:
            self.core.data = self.core.data / torch.norm(self.core.data, p='fro')
        for k in range(self.d):
            self.factors[k].data = (
                self.factors[k].data / torch.norm(self.factors[k].data, p='fro', dim=0)
            )
            
    def inverse_batch(self, input_batch, tensorize_output=False, fast_svd=False):
        output = input_batch.clone()
        if self.use_core:
            output_batch = self.core.clone()
            for k in range(self.d):
                output_batch = torch_utils.prodTenMat_torch(
                    output_batch,
                    self.factors[k].t().mm(self.factors[k]),
                    k+1,
                    1
                )
                output = torch_utils.prodTenMat_torch(
                    output,
                    self.factors[k],
                    k+1,
                    0
                )
            output = torch_utils.flatten_torch(output, use_batch=True)
            output = output.mm(torch_utils.reshape_torch(self.core, [-1, self.r0], use_batch=False))
            output_batch = torch_utils.reshape_torch(output_batch, [-1, self.r0], use_batch=False).t()
            output_batch = output_batch.mm(
                torch_utils.reshape_torch(self.core, [-1, self.r0], use_batch=False)
            )
            u, s, v = torch.svd(output_batch)
            output = output.mm(u/s).mm(v.t())
            return output
        
        for k in range(self.d):
            if fast_svd:
                u, s, v = torch_utils.fast_svd_torch(self.factors[k])
            else:
                u, s, v = torch.svd(self.factors[k])
                u, s, v = u[:, :self.r[k]], s[:self.r[k]], v[:, :self.r[k]]
            output = torch_utils.prodTenMat_torch(output, u/s, k+1, 0)
            output = torch_utils.prodTenMat_torch(output, v, k+1, 1)
        if tensorize_output and output.dim() == 2:
            output = torch_utils.reshape_torch(output, self.r)
        if not tensorize_output and output.dim() != 2:
            output = torch_utils.flatten_torch(output)
        return output
    
class LROTensor(nn.Module):
    def __init__(self, n, lro_config, sample_axis=False):
        super(LROTensor, self).__init__()
        # hyperparameters
        self.sample_axis = sample_axis
        self.n = n
        self.d = len(n)
        self.P = lro_config['P']
        assert 0 < self.P < self.d
        self.N = int(round(np.prod(self.n)))
        self.sample_axis = sample_axis
        
        self.L = lro_config['L']
        self.R = len(lro_config['L'])
        self.M = int(round(sum(self.L)))
        
        # parameters
        self.factors = nn.ParameterList()
        for k in range(self.d):
            if k < self.P:
                shape = [self.n[k], self.M]
            else:
                shape = [self.n[k], self.R]
            self.factors.append(
                nn.Parameter(
                    data=torch.Tensor(size=shape),
                    requires_grad=True
                )
            )
            
            if 'initializer' in lro_config:
                lro_config['initializer'](self.factors[k])
            else:
                self.factors[k].data.normal_()#uniform_(-0.03, 0.03)
                
    def hidden_dim(self):
        if self.sample_axis:
            return self.M
        return None
    
    def set_parameters(self, U):
        assert len(U) == self.d
        for k in range(self.d):
            if U[k] is None:
                continue
            m, n = U[k].shape
            if k < self.P:
                assert (m, n) == (self.n[k], self.M)
            else:
                assert (m, n) == (self.n[k], self.R)
            if isinstance(U[k], np.ndarray):
                self.factors[k].data = torch.from_numpy(U[k]).to(
                    device=self.factors[k].device,
                    dtype=self.factors[k].dtype
                )
            elif isinstance(U[k], torch.Tensor):
                self.factors[k].data = U[k].to(
                    device=self.factors[k].device,
                    dtype=self.factors[k].dtype
                )
            else:
                raise ValueError
    
    def recover(self, weights=None):
        nrows = self.N
        if not self.sample_axis:
            nrows = nrows // self.n[0]
        output = self.factors[0].new_ones([1, self.R])
        for k in range(self.d-1, -1, -1):
            if (k == 0) and (not self.sample_axis):
                break
            if k < self.P:
                if weights is None:
                    output = torch_utils.krp_cw_torch(output, self.factors[k])
                else:
                    output = torch_utils.krp_cw_torch(output, self.factors[k]*weights[k].view(self.n[k], 1))
            else:
                if weights is None:
                    output = torch_utils.krp_cw_torch(output, self.factors[k])
                else:
                    output = torch_utils.krp_cw_torch(output, self.factors[k]*weights[k].view(self.n[k], 1))
                if k == self.P:
                    output = torch.repeat_interleave(output, torch.tensor(self.L, device=self.factors[0].device), dim=1)
        if not self.sample_axis:
            output = self.factors[0].mm(output.t())
            output = torch_utils.flatten_torch(output, use_batch=False)
        return output
    
    def get_svds(self, weights=None, coupled=True, fast=True):
        if self.sample_axis and coupled:
            W = self.recover(weights)
            if fast:
                _, S, V = torch.svd(W.t().mm(W))
                S = S.sqrt()
                U = W.mm(V/S)
            else:
                U, S, V = torch.svd(W)
                U, S, V = U[:, :self.R], S[:self.R], V[:, :self.R]
        else:
            U, S, V = []
            for k in range(self.d):
                if weights is None:
                    u, s, v = torch.svd(self.factors[k])
                else:
                    u, s, v = torch.svd(self.factors[k]*weights[k])
                u, s, v = u[:, :self.R], s[:self.R], v[:, :self.R]
                U.append(u)
                S.append(s)
                V.append(v)
        return U, S, V
    
    def normalize(self):
        for k in range(self.d):
            self.factors[k].data = (
                self.factors[k].data / torch.norm(self.factors[k].data, p='fro', dim=0)
            )
    
    def forward(self, input=None, T=False, tensorize_output=False):
        assert (input is None) != self.sample_axis
        output = self.recover()
        if T:
            output = output.t()
        if self.sample_axis:
            # batch_size = input.shape[0]
            output = torch.einsum('ij,kj->ik', input, output)
        if tensorize_output:
            if T:
                return torch_utils.reshape_torch(output, self.r, use_batch=self.sample_axis)
            else:
                return torch_utils.reshape_torch(output, self.n, use_batch=self.sample_axis)
        return output
    
    def inverse_batch(self, input_batch, fast_svd=True):
        U, S, V = self.get_svds(coupled=True, fast=fast_svd)
        output = torch_utils.flatten_torch(input_batch).mm(U/S).mm(V.t())
        return output

class TTTensor(nn.Module):
    def __init__(self, n, tt_config, sample_axis=False):
        super(TTTensor, self).__init__()
        # hyperparameters
        self.sample_axis = sample_axis
        self.n = n
        self.d = len(n)
        self.N = int(round(np.prod(self.n)))
        self.r = []
        if isinstance(tt_config['r'], (list, tuple, np.ndarray)):
            assert len(tt_config['r']) == self.d+1
            assert tt_config['r'][0] == 1
        # parameters
        self.cores = nn.ParameterList()
        for k in range(self.d):
            if isinstance(tt_config['r'], (list, tuple, np.ndarray)):
                r1, r2 = tt_config['r'][k:k+2]
            elif isinstance(tt_config['r'], int):
                r1 = r2 = tt_config['r']
                if k == 0:
                    r1 = tt_config['r']
                elif k == self.d-1:
                    r2 = tt_config['r']
            shape = [r1, self.n[k], r2]
            self.r.append(r1)
            if k == self.d-1:
                self.r.append(r2)
            self.cores.append(
                nn.Parameter(
                    data=torch.Tensor(size=shape),
                    requires_grad=True
                )
            )
            if 'initializer' in tt_config:
                tt_config['initializer'](self.cores[k])
            else:
                self.cores[k].data.normal_()#uniform_(-0.03, 0.03)
                
    def hidden_dim(self):
        if self.sample_axis:
            return self.r[-1]
        return None
    
    def set_parameters(self, G):
        assert len(G) == self.d
        for k in range(self.d):
            if G[k] is None:
                continue
            r1, nk, r2 = G[k].shape
            assert (r1, nk, r2) == (self.r[k], self.n[k], self.r[k+1])
            if isinstance(G[k], np.ndarray):
                self.cores[k].data = torch.from_numpy(G[k]).to(
                    device=self.cores[k].device,
                    dtype=self.cores[k].dtype
                )
            elif isinstance(U[k], torch.Tensor):
                self.cores[k].data = G[k].to(
                    device=self.cores[k].device,
                    dtype=self.cores[k].dtype
                )
            else:
                raise ValueError
    
    def recover(self, weights=None):
        nrows = self.N
        if not self.sample_axis:
            nrows = nrows // self.n[0]
        output = self.cores[0].new_ones([1, 1])
        for k in range(self.d):
            if weights is None:
                output = output.mm(torch_utils.reshape_torch(self.cores[k], [self.r[k], -1]))
            else:
                output = output.mm(
                    torch_utils.reshape_torch(self.cores[k]*weights[k].view(1, self.n[k], 1), [self.r[k], -1])
                )
            output = torch_utils.reshape_torch(output, [-1, self.r[k]])
        if not self.sample_axis:
            output = torch_utils.flatten_torch(output, use_batch=False)
        else:
            output = torch_utils.reshape_torch(output, [self.N, self.r[-1]], use_batch=False)
        return output
    
    def get_svds(self, weights=None, fast=True):
        U, S, V = [], [], []
        for k in range(self.d):
            if weights is None:
                if fast:
                    u, s, v = fast_svd_torch(
                        torch_utils.reshape_torch(
                            self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False
                        )*weights[k]
                    )
                else:
                    u, s, v = torch.svd(
                        torch_utils.reshape_torch(self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False)
                    )
            else:
                if fast:
                    u, s, v = fast_svd_torch(
                        torch_utils.reshape_torch(self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False)
                    )
                else:
                    u, s, v = torch.svd(
                        torch_utils.reshape_torch(self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False)
                    )
            u, s, v = u[:, :self.r[k+1]], s[:self.r[k+1]], v[:, :self.r[k+1]]
            U.append(u)
            S.append(s)
            V.append(v)
        return U, S, V
    
    def normalize(self):
        for k in range(self.d):
            tmp = torch_utils.reshape_torch(self.cores[k].data, [self.r[k]*self.n[k], -1], use_batch=False)
            tmp = tmp / torch.norm(tmp, p='fro', dim=0)
            self.cores[k].data = torch_utils.reshape_torch(tmp, [self.r[k], self.n[k], -1], use_batch=False)
            
    def orthogonolize(self, last_core=True):
        for k in range(self.d):
            tmp = torch_utils.reshape_torch(self.cores[k].data, [self.r[k]*self.n[k], -1], use_batch=False)
            if k > 0:
                tmp = r.mm(tmp)
            if (k == self.d-1) and (not last_core):
                self.cores[k].data = torch_utils.reshape_torch(tmp, [self.r[k], self.n[k], -1], use_batch=False)
                continue
            q, r = torch.qr(tmp)
            self.cores[k].data = torch_utils.reshape_torch(q, [self.r[k], self.n[k], -1], use_batch=False)
    
    def forward(self, input=None, T=False, tensorize_output=False):
        assert (input is None) != self.sample_axis
        #assert (T and input is not None)
        if T:
            assert input is not None
        if input is None:
            output = self.cores[-1].new_ones([1, 1])
        else:
            if T:
                output = torch_utils.reshape_torch(input, self.n)
            else:
                output = input.clone()
        if T:
            for k in range(self.d):
                output = torch.einsum(
                    'ijk,jl->ilk',
                    torch_utils.reshape_torch(output, [self.r[k]*self.n[k], -1]),
                    torch_utils.reshape_torch(self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False)
                )
        else:
            for k in range(self.d-1, -1, -1):
                if (k == self.d-1) and (input is None):
                    continue
                output = torch.einsum(
                    'ijk,lj->ilk',
                    torch_utils.reshape_torch(output, [self.r[k+1], -1]),
                    torch_utils.reshape_torch(self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False)
                )
        if tensorize_output:
            output = torch_utils.reshape_torch(output, self.n)
        else:
            output = torch_utils.flatten_torch(output)
        return output
    
    def inverse_batch(self, input_batch, tensorize_output=False, fast_svd=False):
        output = torch_utils.reshape_torch(input_batch, self.n)
        for k in range(self.d):
            if fast_svd:
                u, s, v = fast_svd_torch(
                    torch_utils.reshape_torch(
                        self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False
                    )
                )
            else:
                u, s, v = torch.svd(
                    torch_utils.reshape_torch(self.cores[k], [self.r[k]*self.n[k], -1], use_batch=False)
                )
            output = torch.einsum(
                'ijk,jl->ilk',
                torch_utils.reshape_torch(output, [self.r[k]*self.n[k], -1]),
                (u/s).mm(v.t())
            )
        output = torch_utils.flatten_torch(output)
        return output

    