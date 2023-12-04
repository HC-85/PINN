import numpy as np
import tensorflow as tf
from latin_hypercube_sampling import *

def step_burger(u, mu, ep, tau, h):
    periodic_u = np.concatenate([[u[-1]], u, [u[0]]])
    kernel_pure = np.array([mu*tau/(h**2), 0, mu*tau/(h**2)])
    u_m1 = np.pad(periodic_u[:-1], (1, 0), mode='constant', constant_values = 0)
    u_p1 = np.pad(periodic_u[1:], (0, 1), mode='constant', constant_values = 0)
    return np.convolve(periodic_u, kernel_pure, mode='same')[1:-1] + (periodic_u*(u_m1+u_p1)*ep*tau/(2*h))[1:-1]


def evolve_burger(x, t, mu, ep, mode, sampler = None):
    h = x[1]-x[0]
    tau = t[1]-t[0]
    evolution = np.zeros((len(t), len(x)))
    
    if mode == 'sine':
        a, c = next(sampler)
 
        b = np.max([np.min([np.random.geometric(.2,1)[0],5]),1])
        if a==0 or b==0:
            raise Exception(f'{a}, {b}')
        evolution[0] = a*np.sin(b*x+c)
        
    elif mode == 'noise':
        evolution[0] = tf.random.normal(shape=x.shape, mean = 0, stddev = .1).numpy()

    else:
        raise ValueError('mode must be sine or noise')
        
    for step in range(1, len(t)):
        evolution[step] = step_burger(evolution[step-1], mu, ep,tau, h)

    return evolution


def build_dataset(x, t, mu, ep, num_evols, mode:dict = {'sine':.5, 'noise':.5}):
    state = tf.TensorArray(dtype = tf.float32, size = num_evols, dynamic_size=False)
    target = tf.TensorArray(dtype = tf.float32, size = num_evols, dynamic_size=False)
    
    lhs_sampler = latin_hypercube_sampling(((0.1, 1), (0, 2*np.pi)), num_evols//2)
    n_sine = int(num_evols*mode['sine'])
    for i in range(n_sine):
        evolution = evolve_burger(x, t, mu, ep, 'sine', lhs_sampler)
        state = state.write(i, evolution[:-1])
        target = target.write(i, evolution[1:])
   
    for i in range(num_evols-n_sine):
        evolution = evolve_burger(x, t, mu, ep, 'noise')
        state = state.write(i, evolution[:-1])
        target = target.write(i, evolution[1:])

    state = state.stack()
    target = target.stack()
    
    #state = tf.reshape(state, shape = (-1, state.shape[-1]))
    #target = tf.reshape(target, shape = (-1, target.shape[-1]))

    return state, target