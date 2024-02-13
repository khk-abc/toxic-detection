import warnings

import torch
import ot

from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
from time import time
from matplotlib import pyplot as plt


def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper


def migrad(P, Kx, Ky):
    '''
    compute the gradient w.r.t. KDE mutual information
    Parameters
    ----------
    P : transportation plan
    Ks: source kernel matrix
    Kt: target kernel matrix

    Returns
    ----------
    negative gradient w.r.t. MI
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = torch.outer(f_x, f_y)  # matmul
    constC = torch.zeros((len(Kx), len(Ky))).to(Kx.device)
    # there's a negative sign in ot.gromov.tensor_product
    f_xy = -tensor_product(constC, Kx, Ky, P)
    P_f_xy = P / f_xy
    P_grad = -tensor_product(constC, Kx, Ky, P_f_xy)
    P_grad = torch.log(f_xy / f_x_f_y) + P_grad
    return -P_grad

def tensor_product(constC, hC1, hC2, T):
    r"""Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-tensor-product>`
    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    Returns
    -------
    tens : array-like, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result
    """
    # print(hC1.shape,hC2.shape,T.shape)
    A = - torch.mm(
        torch.mm(hC1, T), hC2.T
    )
    # print(constC.shape,A.shape)
    tens = constC + A
    # tens -= tens.min()
    return tens

def compute_kernel_for_transport(C, h):
    '''
    compute Gaussian kernel matrices
    Parameters
    ----------
    C: source-target pairwise distance matrix
    h : bandwidth
    Returns
    ----------
    K: source-target kernel
    '''
    std = torch.sqrt((C ** 2).mean() / 2)
    h = h * std
    # Gaussian kernel (without normalization)
    K = torch.exp(-(C / h) ** 2 / 2)
    return K




def compute_kernel_for_transport_batch(C, h):
        '''
        compute Gaussian kernel matrices
        Parameters
        ----------
        C: source-target pairwise distance matrix
        h : bandwidth
        Returns
        ----------
        K: source-target kernel
        '''
        bz,source_len,target_len = C.shape
        std = torch.sqrt((C ** 2).sum(dim=-1).sum(dim=-1) / (2*source_len*target_len))
        h = h * std
        # Gaussian kernel (without normalization)
        K = torch.exp(-(C / h.unsqueeze(dim=-1).unsqueeze(dim=-1)) ** 2 / 2)
        return K

# @timer
def sinkhorn_log(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False,
                 kernel_source=None, kernel_target=None,lam=1,use_kernel=False,step=2,
                 log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem in log space
    and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm  :ref:`[2] <references-sinkhorn-log>` with the
    implementation from :ref:`[34] <references-sinkhorn-log>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-log:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I.,
        Trouvé, A., & Peyré, G. (2019, April). Interpolating between
        optimal transport and MMD using Sinkhorn divergences. In The
        22nd International Conference on Artificial Intelligence and
        Statistics (pp. 2681-2690). PMLR.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    def get_logT(u, v):
        # return Mr + u[:, None] + v[None, :]
        return Mr + u.unsqueeze(1).repeat(1,Mr.shape[1]) + v.unsqueeze(0).repeat(Mr.shape[0],1)

    if a is None:
        margin_distribution_a = torch.ones(M.shape[-2])/M.shape[-2].to(M.device)
    else:
        margin_distribution_a=a
    if b is None:
        margin_distribution_b = torch.ones(M.shape[-1])/M.shape[-1].to(M.device)
    else:
        margin_distribution_b = b

    # init data
    dim_a = margin_distribution_a.shape[-1]
    dim_b = margin_distribution_b.shape[-1]

    if log:
        log = {'err': []}

    Mr = - M / reg

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        u = torch.zeros(dim_a).to(M.device)
        v = torch.zeros(dim_b).to(M.device)
    else:
        u, v = warmstart

    loga = torch.log(a)
    logb = torch.log(b)

    err = 1
    p = torch.exp(get_logT(u, v))
    for ii in range(numItermax):
        if ii%step==0:
            if use_kernel:
                if kernel_source is not None and kernel_target is not None:
                    grad_p = migrad(p, kernel_source, kernel_target)
                    Mr = Mr + lam*(grad_p / reg)
                elif kernel_source is None or kernel_target is None:
                    trans_kernel = compute_kernel_for_transport(p, h=1)
                    Mr = Mr + 0.3 * trans_kernel

        # v = logb - torch.logsumexp(Mr + u[:, None], 0)
        # u = loga - torch.logsumexp(Mr + v[None, :], 1)
        v = logb - torch.logsumexp(Mr + u.unsqueeze(1).repeat(1,Mr.shape[1]), 0)
        u = loga - torch.logsumexp(Mr + v.unsqueeze(0).repeat(Mr.shape[0],1), 1)

        p = torch.exp(get_logT(u, v))

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            tmp2 = torch.sum(p, 0)
            err = torch.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr:
                break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log['niter'] = ii
        log['log_u'] = u
        log['log_v'] = v
        log['u'] = torch.exp(u)
        log['v'] = torch.exp(v)

        return p, log

    else:
        return p

# @timer
def sinkhorn_log_batch(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False,
                       kernel_source=None, kernel_target=None, lam=1,use_kernel=False,step=2,period=10,
                       log=False, warn=True, warmstart=None, **kwargs):
    def get_logT(u, v):
        # print(Mr.shape,u.shape,v.shape)
        return Mr + u[:,:, None] + v[:,None, :]

    if a is None:
        margin_distribution_a = torch.ones(M.shape[-2])/M.shape[-2].to(M.device)
    else:
        margin_distribution_a=a
    if b is None:
        margin_distribution_b = torch.ones(M.shape[-1])/M.shape[-1].to(M.device)
    else:
        margin_distribution_b = b

    # init data
    bz = M.shape[0]
    dim_a = margin_distribution_a.shape[-1]
    dim_b = margin_distribution_b.shape[-1]


    if log:
        log = {'err': []}

    Mr = - M / reg

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        u = torch.zeros(bz,dim_a).to(M.device)
        v = torch.zeros(bz,dim_b).to(M.device)
    else:
        u, v = warmstart


    loga = torch.log(a)
    logb = torch.log(b)

    err = 1
    p = torch.exp(get_logT(u, v))

    if use_kernel:
        for ii in range(numItermax):
            uprev = u.clone()
            vprev = v.clone()
            if ii%step==0:
                trans_kernel = compute_kernel_for_transport_batch(p, h=1)
                Mr = Mr + 0.1 * trans_kernel

            # print(Mr.shape,u[:,:,None].shape,logb.shape)

            # v = logb - torch.logsumexp(Mr + u[:,:, None], 1)
            # u = loga - torch.logsumexp(Mr + v[:,None, :], 2)
            v = logb - torch.logsumexp(Mr + u.unsqueeze(2).repeat(1,1,Mr.shape[-1]), 1)
            u = loga - torch.logsumexp(Mr + v.unsqueeze(1).repeat(1,Mr.shape[-2],1), 2)

            p = torch.exp(get_logT(u, v))

            if ii % period == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations

                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = torch.sum(p, 1)
                err = torch.norm(tmp2 - b,dim=-1).mean()  # violation of marginal
                # err = torch.norm(tmp2 - b,dim=-1).max()  # violation of marginal
                if log:
                    log['err'].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(ii, err))
                if err < stopThr:
                    break

            if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                warnings.warn('Numerical errors at iteration %d' % ii)
                u = uprev
                v = vprev
                break

        else:
            if warn:
                warnings.warn("Sinkhorn did not converge. You might want to "
                              "increase the number of iterations `numItermax` "
                              "or the regularization parameter `reg`.")
    else:
        for ii in range(numItermax):

            uprev = u.clone()
            vprev = v.clone()
            # print(u.grad)


            # v = logb - torch.logsumexp(Mr + u[:, :, None], 1)
            # u = loga - torch.logsumexp(Mr + v[:, None, :], 2)
            v = logb - torch.logsumexp(Mr + u.unsqueeze(2).repeat(1,1,Mr.shape[-1]), 1)
            u = loga - torch.logsumexp(Mr + v.unsqueeze(1).repeat(1,Mr.shape[-2],1), 2)

            p = torch.exp(get_logT(u, v))

            if ii % period == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations

                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = torch.sum(p, 1)
                # err = torch.norm(tmp2 - b,dim=-1).mean()  # violation of marginal
                err = torch.norm(tmp2 - b, dim=-1).max()  # violation of marginal
                if log:
                    log['err'].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(ii, err))
                if err < stopThr:
                    break

            if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                warnings.warn('Numerical errors at iteration %d' % ii)
                u = uprev
                v = vprev
                break
        else:
            if warn:
                warnings.warn("Sinkhorn did not converge. You might want to "
                              "increase the number of iterations `numItermax` "
                              "or the regularization parameter `reg`.")

    if log:
        log['niter'] = ii
        log['log_u'] = u
        log['log_v'] = v
        log['u'] = torch.exp(u)
        log['v'] = torch.exp(v)

        return p, log

    else:
        return p


def sinkhorn_stabilized_batch(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False, print_period=10,
                        log=False, warn=True, **kwargs):
    if a is None:
        margin_distribution_a = torch.ones(M.shape[-2]) / M.shape[-2].to(M.device)
    else:
        margin_distribution_a = a
    if b is None:
        margin_distribution_b = torch.ones(M.shape[-1]) / M.shape[-1].to(M.device)
    else:
        margin_distribution_b = b

    # init data
    bz = M.shape[0]
    dim_a = margin_distribution_a.shape[-1]
    dim_b = margin_distribution_b.shape[-1]

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = torch.zeros(bz,dim_a).to(M.device), torch.zeros(bz,dim_b).to(M.device)
    else:
        alpha, beta = warmstart


    u, v = torch.ones(bz,dim_a).to(M.device), torch.ones(bz,dim_b).to(M.device)
    u /= dim_a
    v /= dim_b

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape((bz,dim_a, 1))
                        - beta.reshape((bz, 1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(-(M - alpha.reshape((bz,dim_a, 1)) - beta.reshape((bz,1, dim_b)))
                      / reg + torch.log(u.reshape((bz,dim_a, 1))) + torch.log(v.reshape((bz,1, dim_b))))

    K = get_K(alpha, beta)
    transp = K
    err = 1
    for ii in range(numItermax):

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (torch.einsum('bst,bs->bt',K,u))
        u = a / (torch.einsum('bst,bt->bs',K,v))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            u = torch.ones(bz,dim_a).to(M.device) / dim_a
            v = torch.ones(bz,dim_b).to(M.device) / dim_b

            K = get_K(alpha, beta)

        if ii % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = get_Gamma(alpha, beta, u, v)
            err = torch.norm(torch.sum(transp, dim=1) - b,dim=-1).max()

            if log:
                log['err'].append(err)

            if verbose:
                if ii % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        if err <= stopThr:
            break

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        logu = alpha / reg + torch.log(u)
        logv = beta / reg + torch.log(v)
        log["n_iter"] = ii
        log['logu'] = logu
        log['logv'] = logv
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])

        return get_Gamma(alpha, beta, u, v), log
    else:
        return get_Gamma(alpha, beta, u, v)





def pairwise_dis_batch(x, y, metric='cosine'):
    if metric == 'l2':
        pairwise_dis = torch.sqrt(((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1))
    elif metric == 'l1':
        pairwise_dis = (torch.abs(x[:, :, None, :] - y[:, None, :, :])).sum(dim=-1)
    elif metric == 'cosine':
        pairwise_dis = (1 - torch.cos(torch.einsum('bsd,btd->bst', x, y))) / 2
    return pairwise_dis


def compute_kernel_batch(Cx, Cy, h):
    '''
    compute Gaussian kernel matrices
    Parameters
    ----------
    Cx: source pairwise distance matrix
    Cy: target pairwise distance matrix
    h : bandwidth
    Returns
    ----------
    Kx: source kernel
    Ky: targer kernel
    '''
    # print(Cx.shape)
    bz,source_len,source_len = Cx.shape
    bz,target_len,target_len = Cy.shape
    # [bz,]
    std1 = torch.sqrt((Cx ** 2).sum(dim=-1).sum(dim=-1) / (2*source_len*source_len))
    std2 = torch.sqrt((Cy ** 2).sum(dim=-1).sum(dim=-1) / (2*target_len*target_len))
    h1 = h * std1
    h2 = h * std2
    # print(h1.shape)
    # Gaussian kernel (without normalization)
    Kx = torch.exp(-(Cx / h1.unsqueeze(dim=-1).unsqueeze(dim=-1)) ** 2 / 2)
    Ky = torch.exp(-(Cy / h2.unsqueeze(dim=-1).unsqueeze(dim=-1)) ** 2 / 2)
    return Kx, Ky

if __name__=='__main__':
    import ot
    from multiprocessing import Pool

    torch.manual_seed(42)
    source = torch.rand(2,7,768)
    target = torch.randn(2,9,768)
    print(source[0][0][:5])
    # source = torch.ones(2,7,768)/7
    # target = 2*torch.ones(2,9,768)/9

    paire_dis_sour = pairwise_dis_batch(source,source)
    paire_dis_tar = pairwise_dis_batch(target,target)
    ker_x,ker_y = compute_kernel_batch(paire_dis_sour,paire_dis_tar,1)

    a=torch.ones(2,7)/source.shape[1]
    b=torch.ones(2,9)/target.shape[1]

    m=torch.bmm(source,target.permute(0,2,1))

    batch_res = sinkhorn_log_batch(a, b, m, reg=1,stopThr=1e-6,
                                   use_kernel=False,
                                   kernel_source=ker_x,kernel_target=ker_y)
    print(batch_res[0])
    print(batch_res[0].sum(0))
    print(batch_res[0].sum(1))


    # batch_res = sinkhorn_stabilized_batch(a, b, m, reg=1,stopThr=1e-6,
    #                                use_kernel=False,
    #                                kernel_source=ker_x,kernel_target=ker_y)
    # print(batch_res[0].shape)

    # res = []
    # for bz in range(a.shape[0]):
    #     res.append(sinkhorn_log(a[bz],b[bz],m[bz],reg=1,numItermax=100,stopThr=1e-6,
    #                             use_kernel=False,
    #                             kernel_source=ker_x[bz], kernel_target=ker_y[bz]
    #                             ))
    #     print(res)


    # ress = []
    # for bz in range(a.shape[0]):
    #     res = ot.sinkhorn(a[bz],b[bz],m[bz],reg=1,stopThr=1e-6,
    #                       log=False,verbose=False,method='sinkhorn_log')
    #     ress.append(res)
    #     print(res)



