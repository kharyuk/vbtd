import torch
import copy
import numpy as np

def reshape_torch(input, shape, order='F', use_batch=True):
    view_shape = copy.deepcopy(shape)
    if use_batch:
        batch_size = input.shape[0]
        view_shape = [batch_size] + view_shape
    if order == 'C':
        return input.view([view_shape])    
    output = input.permute(list(range(len(input.stride())-1, -1, -1))).contiguous()
    try:
        output.view_(view_shape[::-1])
    except AttributeError:
        output = torch.reshape(output, view_shape[::-1])
    output = output.permute(list(range(len(output.stride())-1, -1, -1))).contiguous()
    return output

def flatten_torch(input, order='F', use_batch=True):
    shape = [-1]
    return reshape_torch(input, shape, order=order, use_batch=use_batch)

def swapaxes_torch(tensor, axis0, axis1):
    if axis0 == axis1:
        return tensor.clone()
    min_axis = min(axis0, axis1)
    max_axis = max(axis0, axis1)
    d = len(tensor.shape)
    permutation = list(range(min_axis)) + [max_axis] + list(range(min_axis+1, max_axis))
    permutation += [min_axis] + list(range(max_axis+1, d))
    return tensor.permute(permutation).contiguous()

def swapunfold_torch(tensor, mode, use_batch=True):
    offset = int(use_batch)
    mode_size = tensor.shape[mode+offset]
    output = swapaxes_torch(tensor, offset, mode+offset)
    output = reshape_torch(output, [mode_size, -1], order='F', use_batch=use_batch)
    return output
    

def prodTenMat_torch(tensor, matrix, tensor_axis, matrix_axis=0):
    if matrix_axis == 0:
        expression = 'i...,ij->j...'
    else:
        expression = 'j...,ij->i...'
    output = torch.einsum(expression, swapaxes_torch(tensor, 0, tensor_axis), matrix)
    return swapaxes_torch(output, 0, tensor_axis)

def kron_torch(A, B):
    # https://discuss.pytorch.org/t/kronecker-product/3919
    shapeA, shapeB = A.shape, B.shape
    output = torch.einsum('ij,kl->ikjl', A, B)
    output = output.view([shapeA[0]*shapeB[0], shapeA[1]*shapeB[1]])
    return output

def krp_cw_torch(A, B, reverse=False, column=True):
    assert A.dim() == 2
    assert B.dim() == 2
    if column:
        R = A.shape[1]
        assert R == B.shape[1]
        M = A.shape[0]*B.shape[0]
    else:
        M = A.shape[0]
        assert M == B.shape[0]
        R = A.shape[1]*B.shape[1]
    output = A.new_ones(M, R)
    if column:
        for i in range(R):
            if not reverse:
                output[:, i:i+1] = kron_torch(A[:, i:i+1], B[:, i:i+1])
            else:
                output[:, i:i+1] = kron_torch(B[:, i:i+1], A[:, i:i+1])
    else:
        for j in range(M):
            if not reverse:
                output[j:j+1, :] = kron_torch(A[j:j+1, :], B[j:j+1, :])
            else:
                output[j:j+1, :] = kron_torch(B[j:j+1, :], A[j:j+1, :])
    return output

def inversedMatrixKP_torch(X, shapeA, shapeB, docopy=True):
    '''
    Solves the following problem: \min_{A, B} \|X - A \otimes B \|_F
    
    Example:
        nA = [3, 2]
        nB = [2, 3]
        A = np.random.uniform(size=(nA))
        B = np.random.uniform(size=(nB))
        X = np.kron(A, B)
        C, D = inversedMatrixKP(X, nA, nB)
        print(np.linalg.norm(np.kron(C, D) - X)/np.linalg.norm(X))
    '''
    n = X.shape
    assert np.all(np.array(shapeA)*np.array(shapeB) == np.array(n))
    nA, nB = list(shapeA), list(shapeB)
    nY = [nB[0], nA[0], nB[1], nA[1]]
    nYt = [nA[0]*nA[1], nB[0]*nB[1]]
    Y = reshape_torch(X, nY, use_batch=False)
    permutation = [1, 3, 0, 2]
    Y = Y.permute(permutation).contiguous()
    Y = reshape_torch(Y, nYt, use_batch=False)
    A, S, B = torch.svd(Y)
    S = S[0]**0.5
    A = S*A[:, 0]
    B = S*B[:, 0]
    A = reshape_torch(A, nA, use_batch=False)
    B = reshape_torch(B, nB, use_batch=False)
    return A, B

def inverse_kronecker_product(vector, shapes):
    M = int(round(np.prod(shapes)))
    assert len(vector) == M
    d = len(shapes)
    result = []
    tmp = vector.clone()
    for i in range(d-1):
        M = M // shapes[i]
        tmp = reshape_torch(tmp, [-1, 1], use_batch=False)
        u, tmp = inversedMatrixKP_torch(tmp, [shapes[i], 1], [M, 1])
        result.append(u)
    result.append(tmp)
    return result



def fast_svd_torch(a, chi=1.2):
    assert a.dim() == 2
    m, n = a.shape
    if m / n >= chi:
        _, s, v = torch.svd(a.t().mm(a))
        s = s.sqrt()
        v = v[:, :len(s)]
        u = a.mm(v/s)
    elif n / m >= chi:
        u, s, _ = torch.svd(a.mm(a.t()))
        s = s.sqrt()
        u = u[:, :len(s)]
        v = a.t().mm(u/s)
    else:
        u, s, v = torch.svd(a)
        R = len(s)
        u, v = u[:, :R], v[:, :R]
    return u, s, v


# FINALLY, https://pytorch.org/docs/master/notes/extending.html
# https://pytorch.org/docs/stable/sparse.html
class fast_svd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix, chi=1.2):
        assert input_matrix.dim() == 2
        m, n = input_matrix.shape
        if m / n >= chi:
            _, s, v = torch.svd(input_matrix.t().mm(input_matrix))
            s = s.sqrt()
            v = v[:, :len(s)]
            u = input_matrix.mm(v/s)
        elif n / m >= chi:
            u, s, _ = torch.svd(input_matrix.mm(input_matrix.t()))
            s = s.sqrt()
            u = u[:, :len(s)]
            v = input_matrix.t().mm(u/s)
        else:
            u, s, v = torch.svd(input_matrix)
            R = len(s)
            u, v = u[:, :R], v[:, :R]
        ctx.save_for_backward(input_matrix, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, grad_output):
        a, u, s, v = ctx.saved_tensors
        grad_a = grad_u = grad_s = grad_v = None
        if ctx.needs_input_grad[1]:
            grad_u = grad_output.mm(v*s)
        if ctx.needs_input_grad[2]:
            grad_s = grad_output.t().mm(u).t().mm(v)
        if ctx.needs_input_grad[3]:
            grad_v = (u*s).t().mm(grad_output)
        if ctx.needs_input_grad[0]:
            pass
        return grad_a, grad_u, grad_s, grad_v
