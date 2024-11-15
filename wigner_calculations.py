import numpy as np
import sympy as sp
from sympy.physics.wigner import wigner_d
from multiprocessing import Pool
import os

D = 101
alpha_res = 20
beta_res = 20
L = sp.Integer((D - 1) / 2) # Arbitrarily choosing only odd dimensions? Well, there are other reasons xd but still
d_alpha, d_beta, d_gamma = sp.symbols('alpha, beta, gamma', real=True)
d_matrix = wigner_d(L,d_alpha,d_beta,d_gamma)

def evaluate_matrix(args):
        alpha, beta = args
        eval_matrix = d_matrix.subs({d_alpha: alpha, d_beta: beta, d_gamma: 0}).evalf()
        return np.array(eval_matrix.tolist()).astype(np.complex128)

if __name__ == "__main__":
    # Honestly, I should be consistent for this naming
    alphas = np.linspace(0,np.pi,alpha_res)
    betas = np.linspace(0,np.pi/2,beta_res)
    cwd = os.getcwd()
    path = "/data/wigners/"
    filename = "w_dim" + str(D) + ".npy"

    alpha_beta_pairs = [(alpha, beta) for alpha in alphas for beta in betas]

    with Pool() as pool:
        results = pool.map(evaluate_matrix, alpha_beta_pairs)

    wigner_matrix = np.array(results).reshape(alpha_res, beta_res, D, D)
    np.save(cwd+path+filename,wigner_matrix)
