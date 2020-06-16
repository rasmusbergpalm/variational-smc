import os
import random

import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from variational_smc import *
import matplotlib.pyplot as plt


def init_model_params(Dx, Dy, r, obs, rs=npr.RandomState(0)):
    mu0 = np.zeros(Dx)
    Sigma0 = np.eye(Dx)

    dt = 1 / 10
    A = np.array([  # 4x4
        # x, y, vx, vy
        [1.0, 0.0, 0.0, 0.0],  # x
        [0.0, 1.0, 0.0, 0.0],  # y
        [dt, 0.0, 1.0, 0.0],  # vx
        [0.0, dt, 0.0, 1.0],  # vy
    ]).T

    Q = np.zeros((Dx, Dx))
    C = np.zeros((Dy, Dx))
    if obs == 'sparse':
        C[:Dy, :Dy] = np.eye(Dy)
    else:
        C = rs.normal(size=(Dy, Dx))
    R = r * np.eye(Dy)

    return (mu0, Sigma0, A, Q, C, R)


def init_params(Dx, Dy, rs=npr.RandomState(0)):
    mu0 = np.zeros(Dx)
    Sigma0 = np.eye(Dx)

    A = np.eye(4)
    Q = np.eye(Dx)
    C = rs.normal(size=(Dy, Dx))
    R = np.eye(Dy)

    return (mu0, Sigma0, A, Q, C, R)


def init_prop_params(T, Dx, scale=0.5, rs=npr.RandomState(0)):
    return [(scale * rs.randn(Dx),  # Bias
             1. + scale * rs.randn(Dx),  # Linear times A/mu0
             scale * rs.randn(Dx))  # Log-var
            for t in range(T)]


def generate_data(model_params, T=5, rs=npr.RandomState(0)):
    mu0, Sigma0, A, Q, C, R = model_params
    Dx = mu0.shape[0]
    Dy = R.shape[0]

    x_true = np.zeros((T, Dx))
    y_true = np.zeros((T, Dy))

    for t in range(T):
        if t > 0:
            x_true[t, :] = rs.multivariate_normal(np.dot(A, x_true[t - 1, :]), Q)
        else:
            x_true[0, :] = rs.multivariate_normal(mu0, Sigma0)
        y_true[t, :] = rs.multivariate_normal(np.dot(C, x_true[t, :]), R)

    return x_true, y_true


def log_marginal_likelihood(model_params, T, y_true):
    mu0, Sigma0, A, Q, C, R = model_params
    Dx = mu0.shape[0]
    Dy = R.shape[1]

    log_likelihood = 0.
    xfilt = np.zeros(Dx)
    Pfilt = np.zeros((Dx, Dx))
    xpred = mu0
    Ppred = Sigma0

    for t in range(T):
        if t > 0:
            # Predict
            xpred = np.dot(A, xfilt)
            Ppred = np.dot(A, np.dot(Pfilt, A.T)) + Q

        # Update
        yt = y_true[t, :] - np.dot(C, xpred)
        S = np.dot(C, np.dot(Ppred, C.T)) + R
        K = np.linalg.solve(S, np.dot(C, Ppred)).T
        xfilt = xpred + np.dot(K, yt)
        Pfilt = Ppred - np.dot(K, np.dot(C, Ppred))

        sign, logdet = np.linalg.slogdet(S)
        log_likelihood += -0.5 * (np.sum(yt * np.linalg.solve(S, yt)) + logdet + Dy * np.log(2. * np.pi))

    return log_likelihood


class lgss_smc:
    """
    Class for defining functions used in variational SMC.
    """

    def __init__(self, T, Dx, Dy, N):
        self.T = T
        self.Dx = Dx
        self.Dy = Dy
        self.N = N

    def log_normal(self, x, mu, Sigma):
        dim = Sigma.shape[0]
        sign, logdet = np.linalg.slogdet(Sigma)
        log_norm = -0.5 * dim * np.log(2. * np.pi) - 0.5 * logdet
        Prec = np.linalg.inv(Sigma)
        return log_norm - 0.5 * np.sum((x - mu) * np.dot(Prec, (x - mu).T).T, axis=1)

    def log_prop(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, A, Q, C, R = model_params
        mut, lint, log_s2t = prop_params[t]
        s2t = np.exp(log_s2t)

        if t > 0:
            mu = mut + np.dot(A, Xp.T).T * lint
        else:
            mu = mut + lint * mu0

        return self.log_normal(Xc, mu, np.diag(s2t))

    def log_target(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, A, Q, C, R = model_params
        if t > 0:
            logF = self.log_normal(Xc, np.dot(A, Xp.T).T, Q)
        else:
            logF = self.log_normal(Xc, mu0, Sigma0)
        logG = self.log_normal(np.dot(C, Xc.T).T, y[t], R)
        return logF + logG

    # These following 2 are the only ones needed by variational-smc.py
    def log_weights(self, t, Xc, Xp, y, prop_params, model_params):
        return self.log_target(t, Xc, Xp, y, prop_params, model_params) - \
               self.log_prop(t, Xc, Xp, y, prop_params, model_params)

    def sim_prop(self, t, Xp, y, prop_params, model_params, rs=npr.RandomState(0)):
        mu0, Sigma0, A, Q, C, R = model_params
        mut, lint, log_s2t = prop_params[t]
        s2t = np.exp(log_s2t)

        if t > 0:
            mu = mut + np.dot(A, Xp.T).T * lint
        else:
            mu = mut + lint * mu0
        return mu + rs.randn(*Xp.shape) * np.sqrt(s2t)


if __name__ == '__main__':
    # Model hyper-parameters
    T = 65
    Dx = 4
    Dy = 2
    r = .1
    obs = 'sparse'

    # Training parameters
    param_scale = 0.5
    num_epochs = 1_000_000
    step_size = 0.001
    N = 256

    data_seed = npr.RandomState(0)

    name = "nbody"
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
    revision = os.environ.get("REVISION") or 'rand-%08x' % random.randrange(16 ** 8)
    message = os.environ.get('MESSAGE')
    train_writer = SummaryWriter(tensorboard_dir + '/%s/%s/train/%s' % (name, revision, message))

    true_model_params = init_model_params(Dx, Dy, r, obs, data_seed)


    def get_data():
        return generate_data(true_model_params, T, data_seed)


    print("Generating data...")
    x_true, y_true = get_data()

    # plt.plot(x_true[:, 0], x_true[:, 1], 'b-', y_true[:, 0], y_true[:, 1], 'r.')
    # plt.show()

    lml = log_marginal_likelihood(true_model_params, T, y_true)
    print("True log-marginal likelihood: " + str(lml))

    seed = npr.RandomState(0)

    # Initialize proposal parameters
    ip = init_params(Dx, Dy, data_seed)
    prop_params = init_prop_params(T, Dx, param_scale, seed)
    combined_init_params = (ip, prop_params)

    lgss_smc_obj = lgss_smc(T, Dx, Dy, N)


    def objective(combined_params, iter):
        # x_true, y_true = get_data()
        model_params, prop_params = combined_params
        return -vsmc_lower_bound(prop_params, model_params, y_true, lgss_smc_obj, seed)


    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    progress = tqdm.tqdm(total=num_epochs)


    def log_perf(combined_params, iter, grad):
        progress.update()
        if iter % 10 == 0:
            model_params, prop_params = combined_params
            elbo = -objective(combined_params, iter)
            lml = log_marginal_likelihood(model_params, T, y_true)
            train_writer.add_scalar("loss/loss", -elbo, global_step=iter)
            train_writer.add_scalar("loss/elbo", elbo, global_step=iter)
            train_writer.add_scalar("loss/logpx", lml, global_step=iter)


    # SGD with adaptive step-size "adam"
    optimized_params = adam(objective_grad, combined_init_params, step_size=step_size, num_iters=num_epochs, callback=log_perf)
    opt_model_params, opt_prop_params = optimized_params
    (mu0, Sigma0, A, Q, C, R) = opt_model_params
    print("mu0", mu0)
    print("Sigma0", Sigma0)
    print("A", A)
    print("Q", Q)
    print("C", C)
    print("R", R)

    print(sim_q(opt_prop_params, opt_model_params, y_true, lgss_smc_obj, seed))
