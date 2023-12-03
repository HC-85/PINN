# Physics-Informed Neural Network (PINN) TensorFlow implementation
From the paper [_Deep learning models for global coordinate transformations that linearise PDEs_](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/abs/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192)

Deep autoencoder architecture that finds a near-identity coordinate transformation which turns a non-linear PDE into a linear PDE. 
The dynamics of the system in these news coordinates are given by the Koopman operator **K**.
  
Its functionality is shown by learning the dynamics for Burgers' Equation.

## Bateman-Burgers' Equation
Convection-diffusion PDE

$$
u_t + \epsilon u u_x = \mu u_{xx}
$$

where $\mu$ is a diffusion coefficient, $\epsilon$ is the advection coefficient, and $u(x, t)$ is the fluid speed.

### Analytical solution
We can use the Cole-Hopf transformation:

$$
\nu = \exp\left(-\frac{1}{2\mu}\int u\right) 
$$

To linearize the system to a simple heat equation in the new regime ($\epsilon = 1$):

$$
\nu_t = \mu\nu_{xx}
$$

## Dataset generation
We use the approximations:

$$
\begin{align}
\frac{\partial u}{\partial t} &= \frac{u_{x, t+\tau} - u_{x, t}}{\tau}\\
\frac{\partial u}{\partial x} &= \frac{u_{x+h, t} + u_{x-h, t}}{2h}\\
\frac{\partial ^2 u}{\partial x^2} &= \frac{u_{x+h, t}-2u_{x, t}+u_{x-h, t}}{h^2}
\end{align}
$$

with $\Delta t = \tau$ and $\Delta x = h$.

Plugging into the PDE and rearranging:

$$
\frac{u_{x, t+\tau} + u_{x, t}}{\tau} = \mu \frac{u_{x+h, t}-2u_{x, t}+u_{x-h, t}}{h^2} - \epsilon u_{x,t}\frac{u_{x+h, t} + u_{x-h, t}}{2h}
$$

$$
\begin{align}
u_{x, t+\tau} &= \mu\tau \frac{u_{x+h, t}-2u_{x, t}+u_{x-h, t}}{h^2} - \epsilon u_{x,t}\tau\frac{u_{x+h, t} + u_{x-h, t}}{2h} - u_{x, t}\\
u_{x, t+\tau} &= u_{x-h, t}\left[\frac{\mu\tau}{h^2}\right] + u_{x,t}u_{x+h,t}\left[\frac{\epsilon\tau}{2h}\right] + u_{x,t}u_{x-h,t}\left[\frac{\epsilon\tau}{2h}\right] + u_{x+h, t}\left[\frac{\mu\tau}{h^2}\right]
\end{align}
$$
