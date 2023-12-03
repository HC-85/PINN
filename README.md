# Physics-Informed Neural Network (PINN) TensorFlow implementation
From the paper [_Deep learning models for global coordinate transformations that linearise PDEs_](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/abs/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192)

Deep autoencoder architecture that finds a near-identity coordinate transformation which turns a non-linear PDE into a linear PDE. 
The dynamics of the system in these news coordinates are given by the Koopman operator **K**.
  
Its functionality is shown by learning the dynamics for Burgers' Equation.

