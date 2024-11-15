import numpy as np
import sympy as sp
from sympy.physics.wigner import wigner_d
from multiprocessing import Pool
import os

D = 17
interaction_pattern = "r"
concurrences_savepath = "/data/wigners/"
concurrences_filename = interaction_pattern + "concs_dim" + str(D) + ".npy"

n_qubits = 2
qubitstate_size = 2**n_qubits
if interaction_pattern == "r":
    seed_filename = interaction_pattern+"seed_dim101_q" + str(n_qubits) + ".txt"
elif interaction_pattern == "o":
    seed_filename = "batch_instates/ordered_seed.txt"
    
# seed_filename = "batch_instates/random_seed.txt"
# seed_filename = interaction_pattern+"seed_dim" + str(D) + "_q" + str(n_qubits) + ".txt"

cwd = os.getcwd()
wigners_path = "/data/wigners/"
wigners_filename = "w_dim" + str(D) + ".npy"
wigner_matrix = np.load(cwd+wigners_path+wigners_filename)
start_skip = 1000

def compute_concurrence(a, b, full_states):
    transformed_states = np.einsum('pq,tpmn->tqmn', np.conj(wigner_matrix[a, b, :, :]), full_states)
    concurrence_values = np.sum(2*np.absolute(np.multiply(transformed_states[:,:,0,0],transformed_states[:,:,1,1]) - np.multiply(transformed_states[:,:,0,1],transformed_states[:,:,1,0])),axis=1) # Each is multiplied by the probability of measuring D(alpha,beta) |m>
    steady_concurrence = np.average(concurrence_values[start_skip:],axis=0)
    return a, b, steady_concurrence

if __name__ == "__main__":
    p = np.arange(0,D,1)
    sqrt2 = np.sqrt(2)
    w = 2*np.pi/D
    im = complex(0.0,1.0)
    omegap = np.exp(im*w*p)


    # beta = 0.587033
    # gamma = 3.09554
    coin_beta = np.pi/2
    coin_gamma = np.pi/2
    theta = 3 * np.pi/8
    phi = np.pi/2

    instate = np.zeros((D,2,2),np.complex128)
    pstate = np.zeros((D,2,2),np.complex128)
    U = np.zeros((D,2,2),np.complex128)
    U[:,0,0] = np.exp(im*coin_gamma) * np.cos(coin_beta/2) * np.exp(-im*p)
    U[:,1,0] = -np.exp(im*coin_gamma) * np.sin(coin_beta/2) * np.exp(im*p)
    U[:,0,1] = np.sin(coin_beta/2) * np.exp(-im*p)
    U[:,1,1] = np.cos(coin_beta/2) * np.exp(im*p)

    instate[0,0,0] = np.cos(theta) * np.cos(theta)
    instate[0,1,0] = np.cos(theta) * np.sin(theta) * np.exp(im*phi)
    instate[0,0,1] = np.cos(theta) * np.sin(theta) * np.exp(im*phi)
    instate[0,1,1] = np.sin(theta) * np.sin(theta) * np.exp(2*im*phi)

    transf = np.zeros((D,D),np.complex128)

    for m in p:
        transf[m,:] = omegap**m
    np.multiply(1/np.sqrt(D),transf,out=transf);
    np.einsum('pq,qmn -> pmn',transf,instate,out=pstate);

    seed = np.genfromtxt(cwd+"/data/seeds/"+seed_filename, usecols = 0, delimiter=",", dtype = int)
    T = seed.size

    alpha_res, beta_res, D, D = wigner_matrix.shape
    # Honestly, I should be consistent for this naming
    alphas = np.linspace(0,np.pi,alpha_res)
    betas = np.linspace(0,np.pi/2,beta_res)

    measurement_concurrences = np.zeros((alpha_res,beta_res),np.float64)

    full_pstates = np.zeros((T+1,D,2,2),np.complex128)
    
    full_pstates[0,:,:,:] = pstate
    t = 1
    for int_qubit in seed:
        if int_qubit:
            full_pstates[t,:,:,:] = np.einsum('pjk,pkm->pjm',U,full_pstates[t-1,:,:,:])
        else:
            full_pstates[t,:,:,:] = np.einsum('pjk,pmk->pmj',U,full_pstates[t-1,:,:,:])
        t += 1
    full_states = np.einsum('pq,tpjk->tqjk',np.conj(transf),full_pstates)
        
    args = [(a, b, full_states) for a in range(alpha_res) for b in range(beta_res)]
    with Pool() as pool:
        results = pool.starmap(compute_concurrence, args)
    for a, b, value in results:
        measurement_concurrences[a, b] = value

    np.save(cwd+concurrences_savepath+concurrences_filename,measurement_concurrences)