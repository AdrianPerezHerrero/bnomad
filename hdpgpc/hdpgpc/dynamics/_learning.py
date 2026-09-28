"""Original ML LDS matrix update; operation order and stabilization retained."""

import torch


def estimate_lds_parameters(backend, A_prior, Gamma_prior, C_prior, Sigma_prior, y_samples, means, covars, model='dynamic'):
    """Compute new matrices of linear tranformation using ML approach following
    the model described by:
        f_t = A f_{t-1} + Gamma
        y_t = C f_t + Sigma
    Parameters
    ----------
    A_priors : list of matrix (txsxs_samples)
        Previous matrices of linear transformation.
    Gamma_priors : list of matrix (txsxs_samples)
        Previous matrices of projection noise.
    C_priors : list of matrix (txsxs_samples)
        Previous matrices observations projection.
    Sigma_priors : list of matrix (txsxs_samples)
        Previous matrices observation projection noise.
    y_samples : ordered array of arrays (txs_samples)
        List of ovservations received till moment.
    means : ordered array of arrays (txs_samples)
        List of means computed till moment
    covars : ordered array of matrix (txsxs_samples)
        List of covariance computed till moment
    Returns
    -------
    Matrices (sxs_samples) new matrices of linear transformation model
    which maximize likelihood of the model (A,Gamma, C, Sigma).
    """

    # Start computing essential means
    exp_ft_ft1 = []
    exp_ft1_ft = []
    exp_ft_ft = []
    T = len(means)
    P = []
    J = []

    for t in range(T):
        P.append(torch.matmul(A_prior, torch.matmul(covars[t], A_prior.T)) + Gamma_prior)
        J.append(torch.linalg.solve(P[t].T, torch.matmul(A_prior, covars[t].T)).T)

    for t in range(T - 1):
        exp_ft_ft1.append(torch.matmul(covars[t + 1], J[t].T) + torch.matmul(means[t + 1], means[t].T))
        exp_ft1_ft.append(torch.matmul(J[t], covars[t + 1]) + torch.matmul(means[t], means[t + 1].T))

    for t in range(T):
        exp_ft_ft.append(covars[t] + torch.matmul(means[t], means[t].T))

    # Start computing of new matrices
    # A
    A_aux1 = torch.zeros(A_prior.shape)
    A_aux2 = torch.zeros(A_prior.shape)
    # Gamma
    Gamma_aux = torch.zeros(Gamma_prior.shape)
    # C
    C_aux1 = torch.zeros(C_prior.shape)
    C_aux2 = torch.zeros(C_prior.shape)
    # Sigma
    Sigma_aux = torch.zeros(Sigma_prior.shape)
    zer = torch.tensor([0.0])
    if torch.cuda.is_available() and backend.cuda:
        A_aux1 = A_aux1.cuda()
        A_aux2 = A_aux2.cuda()
        Gamma_aux = Gamma_aux.cuda()
        C_aux1 = C_aux1.cuda()
        C_aux2 = C_aux2.cuda()
        Sigma_aux = Sigma_aux.cuda()
        zer = zer.cuda()

    if model == 'static':

        jitter = torch.mul(1e-8, torch.eye(A_prior.shape[0]))
        A_new = torch.eye(A_prior.shape[0])
        C_new = torch.eye(C_prior.shape[0])
        Gamma_new = torch.zeros(Gamma_prior.shape)
        if torch.cuda.is_available() and backend.cuda:
            jitter = jitter.cuda()
            A_new = A_new.cuda()
            C_new = C_new.cuda()
        for t in range(T):
            # Sigma computations
            Sigma_aux = Sigma_aux + (torch.matmul(y_samples[t], y_samples[t].T)
                                     - torch.matmul(means[t], y_samples[t].T)
                                     - torch.matmul(y_samples[t], means[t].T) + exp_ft_ft[t])
        Sigma_new = Sigma_aux / T
        Sigma_new = (Sigma_new + Sigma_new.T) / 2

        if torch.isclose(torch.linalg.det(Sigma_new), zer):
            Sigma_new = Sigma_new + jitter

    elif model == 'dynamic':

        jitter = torch.mul(1e-8, torch.eye(A_prior.shape[0]))
        if torch.cuda.is_available() and backend.cuda:
            jitter = jitter.cuda()
        for t in range(T - 1):
            # A computations
            A_aux1 = A_aux1 + exp_ft_ft1[t]
            A_aux2 = A_aux2 + exp_ft_ft[t]

        for t in range(T):
            # C computations
            C_aux1 = C_aux1 + torch.matmul(y_samples[t], means[t].T)
            C_aux2 = C_aux2 + exp_ft_ft[t]

        # Numerical stability
        if torch.isclose(torch.linalg.det(A_aux2), zer):
            A_aux2 = A_aux2 + jitter
        if torch.isclose(torch.linalg.det(A_aux1), zer):
            A_aux1 = A_aux1 + jitter

        if torch.isclose(torch.linalg.det(C_aux2), zer):
            C_aux2 = C_aux2 + jitter
        if torch.isclose(torch.linalg.det(C_aux1), zer):
            C_aux1 = C_aux1 + jitter

        A_new = torch.linalg.solve(A_aux2.T, A_aux1.T).T
        C_new = torch.linalg.solve(C_aux2.T, C_aux1.T).T

        for t in range(1, T):
            # Gamma computations
            Gamma_aux = Gamma_aux + (exp_ft_ft[t] - torch.matmul(A_new, exp_ft1_ft[t - 1])
                                     - torch.matmul(exp_ft_ft1[t - 1], A_new.T)
                                     + torch.linalg.multi_dot([A_new, exp_ft_ft[t - 1], A_new.T]))

        for t in range(T):
            # Sigma computations
            Sigma_aux = Sigma_aux + (torch.matmul(y_samples[t], y_samples[t].T)
                                     - torch.linalg.multi_dot([C_new, means[t], y_samples[t].T])
                                     - torch.linalg.multi_dot([y_samples[t], means[t].T, C_new.T])
                                     + torch.linalg.multi_dot([C_new, exp_ft_ft[t], C_new.T]))

        # Compute the final covariance matrix and add a small noise for stability
        if T == 1:
            Gamma_new = Gamma_aux  # + jitter_gamma
        else:
            Gamma_new = Gamma_aux / (T - 1)  # + jitter_gamma/T

        # Ensure symmetry of the covariance matrices
        Gamma_new = (Gamma_new + Gamma_new.T) / 2

        if torch.isclose(torch.linalg.det(Gamma_new), zer):
            Gamma_new = Gamma_new + jitter

        Sigma_new = Sigma_aux / T  # + jitter_alpha/T

        # Ensure symmetry of the covariance matrices
        Sigma_new = (Sigma_new + Sigma_new.T) / 2

        if torch.isclose(torch.linalg.det(Sigma_new), zer):
            Sigma_new = Sigma_new + jitter

    else:
        print("ERROR: Only dynamic and static models implemented.")

    return A_new, Gamma_new, C_new, Sigma_new
