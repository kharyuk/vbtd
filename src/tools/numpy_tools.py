import numpy as np

def reshape_np(input_batch, shape, order='F', use_batch=True):
    if use_batch:
        batch_size = len(input_batch)
        lshape = [batch_size] + list(shape)
    else:
        lshape = shape
    return np.reshape(input_batch, lshape, order=order)

def flatten_np(input_batch, order='F', use_batch=True):
    return reshape_np(input_batch, [-1], order=order, use_batch=use_batch)

def batch_relative_residual_np(Y, Yhat):
    rr = np.linalg.norm(flatten_np(Y - Yhat, use_batch=True), axis=-1)
    rr /= np.linalg.norm(flatten_np(Y, use_batch=True), axis=-1)
    return rr

def fast_svd_np(A, eps=1e-8, dotaxis=1, tol=1.2):
    assert A.ndim == 2
    m, n = A.shape
    if dotaxis is None:
        if m > n*tol:
            dotaxis = 0
        elif n > m*tol:
            dotaxis = 1
        else:
            u, s, vt = np.linalg.svd(A)
            slen = len(s)
            v = vt.T
    if dotaxis is not None:
        if dotaxis == 0:
            v, s, _ = np.linalg.svd(A.T@A)
        elif dotaxis == 1:
            u, s, _ =  np.linalg.svd(A@A.T)
        else:
            raise ValueError
        s = s**0.5
    cum = np.cumsum(s[::-1])
    I = (cum > eps*(np.linalg.norm(A)**2.)).sum()
    s = s[:I]
    if dotaxis is None:
        u, v = u[:, :I], v[:, :I]
    else:
        if dotaxis == 0:
            v = v[:, :I]
            u = (A@v)/s
        else:
            u = u[:, :I]
            v = (A.T@u)/s
    return u, s, v
