"""
Code for neural differential equation frameworks
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import jacfwd, vmap


def RK4_step(z: Tensor, dt: Tensor, ODE_f: Callable[[Tensor], Tensor]) -> Tensor:
    """Perform a single Runge-Kutta 4 time step"""
    k1 = ODE_f(z)
    k2 = ODE_f(z + 0.5 * dt * k1)
    k3 = ODE_f(z + 0.5 * dt * k2)
    k4 = ODE_f(z + dt * k3)
    z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


class NeuralGeodesicFlows(ABC, nn.Module):
    """
    Learns simulation based geodesic flows in the latent space of an autoencoder
    https://www.research-collection.ethz.ch/server/api/core/bitstreams/d14c6293-9115-4fa2-bf83-e5628f6581f8/content
    """

    encoder: nn.Module
    decoder: nn.Module
    metric: nn.Module

    def __init__(self, latent_dim: int):
        super().__init__()
        self.M_dim = latent_dim // 2
        assert latent_dim % 2 == 0, "Latent dimension must be even for geodesic flows"

    def _check_metric_dim(self):
        """Check metric dimension after initialization"""
        assert self.M_dim == getattr(
            self.metric, "dim_M", None
        ), "Metric dimension must match half the latent dimension"

    def connection_coeffs(self, x: Tensor) -> Tensor:
        """
        Compute Christoffel symbols (connection coefficients).
        Returns Gamma^k_ab with shape (bs, m, m, m)
        """

        # Compute Jacobian and metric value
        def single_jacobian(x_single: Tensor) -> Tensor:
            return jacfwd(self.metric)(x_single.unsqueeze(0)).squeeze(0)

        partial_g = vmap(single_jacobian)(x).squeeze()  # shape (bs, m, m, m)
        g = self.metric(x)  # shape (bs, m, m) - compute once!
        inverse_g = torch.linalg.inv(g.float()).to(x.dtype)  # reuse g

        # Formula: Gamma^k_ab = 1/2 g^ki (partial_a g_ib + partial_b g_ai - partial_i g_ab)
        term1 = torch.einsum("Bki,Baib->Bkab", inverse_g, partial_g)
        term2 = torch.einsum("Bki,Bbai->Bkab", inverse_g, partial_g)
        term3 = torch.einsum("Bki,Biab->Bkab", inverse_g, partial_g)

        return 0.5 * (term1 + term2 - term3)

    def geodesic_ODE_function(self, z: Tensor) -> Tensor:
        """
        RHS of the geodesic ODE: dz/dt = f(z)
        z = [x, v] where x is position and v is velocity
        z has shape (bs, 2*m)
        """
        m = self.M_dim
        x = z[:, :m]  # position components, shape (bs, m)
        v = z[:, m : 2 * m]  # velocity components, shape (bs, m)

        Gamma = self.connection_coeffs(x)  # shape (bs, m, m, m)

        dxbydt = v  # shape (bs, m)
        dvbydt = -torch.einsum("Bkab,Ba,Bb->Bk", Gamma, v, v)  # shape (bs, m)

        return torch.cat([dxbydt, dvbydt], dim=1)  # shape (bs, 2*m)

    def exp(self, z: Tensor, t: Tensor, num_steps: int) -> Tensor:
        """
        Geodesic exponential map.
        Starting at z=(x_initial, v_initial) in TM, return exp(x_initial, v_initial, t_final)
        i.e., the point c(t_final) on the unique geodesic c(t) with c(0)=x_initial, dc/dt(0)=v_initial
        """
        dt = t / num_steps

        # Integration loop (replaces jax.lax.scan)
        for _ in range(num_steps):
            z = RK4_step(z, dt, self.geodesic_ODE_function)

        return z

    @abstractmethod
    def recon_loss(
        self, x: Tensor, y: Tensor, x_sizes: Tensor, y_sizes: Tensor, **kwargs
    ) -> Tensor:
        """Compute reconstruction loss given input and output sequences"""
        raise NotImplementedError

    @abstractmethod
    def trans_loss(
        self,
        x: Tensor,
        y: Tensor,
        x_sizes: Tensor,
        y_sizes: Tensor,
        t: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute transition loss given input and output sequences and transition time"""
        raise NotImplementedError
