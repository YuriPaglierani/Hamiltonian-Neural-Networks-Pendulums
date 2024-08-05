import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Callable

# Single Pendulum Derivatives
def single_pendulum_position_derivative(state: jnp.ndarray, mass: float, length: float, g: float) -> jnp.ndarray:
    _, p = state
    dH_dp = p / (mass * length**2)
    return jnp.array([dH_dp])

def single_pendulum_momentum_derivative(state: jnp.ndarray, mass: float, length: float, g: float) -> jnp.ndarray:
    theta, _ = state
    dH_dtheta = mass * g * length * jnp.sin(theta)
    return jnp.array([-dH_dtheta])

# if you have doubts about the implementation, look at this link: https://dassencio.org/46
def double_pendulum_position_derivative(state: jnp.ndarray, m1: float, m2: float, l1: float, l2: float, g: float) -> jnp.ndarray:
    theta1, theta2, p1, p2 = state
    dH_dp1 = (l2 * p1 - l1 * p2 * jnp.cos(theta1 - theta2)) / (l1**2 * l2 * (m1 + m2 * jnp.sin(theta1 - theta2)**2) + 1e-6) # avoid blow up
    dH_dp2 = (-m2 * l2 * p1 * jnp.cos(theta1 - theta2) + (m1 + m2) * l1 * p2) / (m2 * l1 * l2**2 * (m1 + m2 * jnp.sin(theta1 - theta2)**2) + 1e-6)
    return jnp.array([dH_dp1, dH_dp2])

def double_pendulum_momentum_derivative(state: jnp.ndarray, m1: float, m2: float, l1: float, l2: float, g: float) -> jnp.ndarray:
    theta1, theta2, p1, p2 = state
    
    def compute_hs(th1, th2, p1, p2):
        h1 = p1 * p2 * jnp.sin(th1 - th2) / (l1 * l2 * (m1 + m2*jnp.sin(th1 - th2)**2) + 1e-6)
        h2 = (m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * jnp.cos(th1 - th2)) / (2 * l1**2 * l2**2 * (m1 + m2*jnp.sin(th1 - th2)**2)**2 + 1e-6)
        return h1, h2

    h1, h2 = compute_hs(theta1, theta2, p1, p2)

    dH_dtheta1 = g * (m1 + m2) * l1 * jnp.sin(theta1) + h1 - h2 * jnp.sin(2*(theta1 - theta2))
    dH_dtheta2 = m2 * g * l2 * jnp.sin(theta2) - h1 + h2 * jnp.sin(2*(theta1 - theta2))
    return jnp.array([-dH_dtheta1, -dH_dtheta2])

@partial(jit, static_argnums=(1, 2, 3))
def symplectic_integrator_step(
    state: jnp.ndarray,
    position_derivative_fn: Callable,
    momentum_derivative_fn: Callable,
    integration_method: str,
    dt: float,
    *args
) -> jnp.ndarray:
    """
    Perform a single step of the specified integration method for the given pendulum system.

    Args:
        state: Current state of the system.
        position_derivative_fn: Function to compute the derivatives with respect to momentum.
        momentum_derivative_fn: Function to compute the derivatives with respect to position.
        integration_method: Either "stormer_verlet" or "symplectic_euler".
        dt: Time step.
        *args: Additional arguments to pass to the derivative functions (parameters).

    Returns:
        New state after one time step.
    """
    if integration_method == "stormer_verlet":
        half_step = dt / 2
        momentum_derivatives = momentum_derivative_fn(state, *args)
        
        # Half step for momenta
        momenta_half = state[len(state)//2:] + half_step * momentum_derivatives
        
        # Full step for positions
        state_half = jnp.concatenate([state[:len(state)//2], momenta_half])
        position_derivatives = position_derivative_fn(state_half, *args)
        positions_new = state[:len(state)//2] + dt * position_derivatives
        
        # Final momenta
        state_new = jnp.concatenate([positions_new, momenta_half])
        momentum_derivatives_new = momentum_derivative_fn(state_new, *args)
        momenta_new = momenta_half + half_step * momentum_derivatives_new
        
        return jnp.concatenate([positions_new, momenta_new])

    elif integration_method == "symplectic_euler":
        momentum_derivatives = momentum_derivative_fn(state, *args)
        
        # Update momenta
        momenta_new = state[len(state)//2:] + dt * momentum_derivatives
        
        # Update positions
        state_half = jnp.concatenate([state[:len(state)//2], momenta_new])
        position_derivatives = position_derivative_fn(state_half, *args)
        positions_new = state[:len(state)//2] + dt * position_derivatives
        
        return jnp.concatenate([positions_new, momenta_new])

    else:
        raise ValueError(f"Unknown integration method: {integration_method}")