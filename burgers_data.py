import numpy as np
import tensorflow as tf
from latin_hypercube_sampling import *

def step_burger(u, mu, ep, tau, h):
    periodic_u = np.concatenate([[u[-1]], u, [u[0]]])
    kernel_pure = np.array([mu*tau/(h**2), 0, mu*tau/(h**2)])
    u_m1 = np.pad(periodic_u[:-1], (1, 0), mode='constant', constant_values = 0)
    u_p1 = np.pad(periodic_u[1:], (0, 1), mode='constant', constant_values = 0)
    return np.convolve(periodic_u, kernel_pure, mode='same')[1:-1] + (periodic_u*(u_m1+u_p1)*ep*tau/(2*h))[1:-1]


def _evolve_burger(x, t, mu, ep, mode):
    h = x[1]-x[0]
    tau = t[1]-t[0]
    evolution = np.zeros((len(t), len(x)))
    
    if mode == 'sine':
        a, b, c = list(tf.random.normal(shape=(3,), mean = 0, stddev = 3).numpy())
        a = a/2
        a = np.sign(a) if np.abs(a)>1 else a 
        b = round(b) if b>1 else 1
        evolution[0] = np.sin(b*x+c*np.pi)

    if mode == 'noise':
        evolution[0] = tf.random.normal(shape=x.shape, mean = 0, stddev = .1).numpy()
        
    for step in range(1,len(t)):
        evolution[step] = step_burger(evolution[step-1], mu, ep,tau, h)
    return evolution


def _build_dataset(x, t, mu, ep, num_evols):
    state = tf.TensorArray(dtype = tf.float32, size = num_evols, dynamic_size=False)
    target = tf.TensorArray(dtype = tf.float32, size = num_evols, dynamic_size=False)
    
    for i in range(num_evols):

        if i < num_evols//2:
            evolution = _evolve_burger(x, t, mu, ep, 'sine')
        else:
            evolution = _evolve_burger(x, t, mu, ep, 'noise')

        state = state.write(i, evolution[:-1])
        target = target.write(i, evolution[1:])

    state = state.stack()
    target = target.stack()
    
    state = tf.reshape(state, shape = (-1, state.shape[-1]))
    target = tf.reshape(target, shape = (-1, target.shape[-1]))

    return state, target


def evolve_burger(x, t, mu, ep, mode, sampler = None):
    h = x[1]-x[0]
    tau = t[1]-t[0]
    evolution = np.zeros((len(t), len(x)))
    
    if mode == 'sine':
        a, c = next(sampler)
        b = np.minimum(np.random.geometric(.2, 1), 5)[0]
        evolution[0] = a*np.sin(b*x+c)
        
    elif mode == 'noise':
        evolution[0] = tf.random.normal(shape=x.shape, mean = 0, stddev = .1).numpy()

    else:
        raise ValueError('mode must be sine or noise')
        
    for step in range(1, len(t)):
        evolution[step] = step_burger(evolution[step-1], mu, ep,tau, h)

    return evolution


def build_dataset(x, t, mu, ep, num_evols):
    state = tf.TensorArray(dtype = tf.float32, size = num_evols, dynamic_size=False)
    target = tf.TensorArray(dtype = tf.float32, size = num_evols, dynamic_size=False)
    
    lhs_sampler = latin_hypercube_sampling(((0,1), (0,2*np.pi)), num_evols//2)
    
    for i in range(num_evols):
        if i%2==0:
            evolution = evolve_burger(x, t, mu, ep, 'sine', lhs_sampler)
        else:
            evolution = evolve_burger(x, t, mu, ep, 'noise')

        state = state.write(i, evolution[:-1])
        target = target.write(i, evolution[1:])

    state = state.stack()
    target = target.stack()
    
    #state = tf.reshape(state, shape = (-1, state.shape[-1]))
    #target = tf.reshape(target, shape = (-1, target.shape[-1]))

    return state, target