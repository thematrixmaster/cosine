from abc import ABC, abstractmethod
from typing import List

import torch
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

####################################################################################################
# Coupling Classes
####################################################################################################


class Coupling(ABC):
    def setup(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class FullyMaskCoupling(Coupling):
    """A coupling that masks the entire target sequence"""

    def sample(self, x1: Tensor, mask_token: int) -> tuple[Tensor, Tensor]:
        return mask_token * torch.ones_like(x1), x1


class PartialMaskCoupling(Coupling):
    def __init__(self, mask_prop: float = 0.8) -> None:
        self.mask_prob = mask_prop

    def sample(self, x1: Tensor, mask_token: int) -> tuple[Tensor, Tensor]:
        I = torch.rand_like(x1.float()) > self.mask_prob
        x0 = mask_token * torch.ones_like(x1)
        x0 = x1 * I + x0 * (~I)
        return x0, x1


#####################################################################################################
# Scheduler Classes
#####################################################################################################


class KappaScheduler(ABC):
    """Base class for kappa schedulers"""

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        return -2 * (t**3) + 3 * (t**2) + self.a * (t**3 - 2 * t**2 + t) + self.b * (t**3 - t**2)

    def derivative(self, t: Tensor) -> Tensor:
        return -6 * (t**2) + 6 * t + self.a * (3 * t**2 - 4 * t + 1) + self.b * (3 * t**2 - 2 * t)


class LinearScheduler(CubicScheduler):
    def __init__(self) -> None:
        super().__init__(a=0.0, b=0.0)

    def __call__(self, t: Tensor) -> Tensor:
        return t

    def derivative(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


#####################################################################################################
# Conditional Probability Path Classes
#####################################################################################################


class ConditionalPath(ABC):
    """Base class for conditional probability paths p_t(xt|x0, x1)"""

    def sample_jump_schedule(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, x0: Tensor, x1: Tensor, t: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class FactorizedPath(ConditionalPath):
    """Factorized token-wise conditional probability path -> linear interpolation"""

    def __init__(self, kappa: KappaScheduler, vocab_size: int) -> None:
        self.kappa = kappa
        self.vocab_size = vocab_size

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor, vocab_size=None, *args, **kwargs) -> Tensor:
        vocab_size = vocab_size if vocab_size is not None else self.vocab_size
        p0 = x2prob(x0, vocab_size=self.vocab_size)
        p1 = x2prob(x1, vocab_size=self.vocab_size)
        ndims = len(p0.shape)
        t = t.reshape(-1, *(1,) * (ndims - 1))
        pt = (1 - self.kappa(t)) * p0 + self.kappa(t) * p1
        return sample_p(pt)


######################################################################################################
# Sampler Class
######################################################################################################


class DiscreteSampler(ABC):
    def __init__(
        self,
        scheduler: KappaScheduler,
        num_steps: int = 100,
        t_min: float = 1e-5,
        temperature: float = 1.0,
        adapt_step_size: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.t_min = t_min
        self.num_steps = num_steps
        self.temperature = temperature
        self.default_h = 1 / num_steps
        self.h = self.adaptative_h if adapt_step_size else self.constant_h

    def u(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Computes the marginal rate"""
        raise NotImplementedError

    def apply_u(self, xt: Tensor, ut: Tensor, h: Tensor, *args, **kwargs) -> Tensor:
        """Apply the marginal rate ut with step size h on xt to obtain xt+1"""
        raise NotImplementedError

    def adaptative_h(self, t: Tensor) -> Tensor:
        coeff = (1 - self.scheduler(t)) / self.scheduler.derivative(t)
        _h = self.default_h * torch.ones_like(t, device=t.device)
        h_adapt = torch.minimum(_h, coeff)
        return h_adapt

    def constant_h(self, t: Tensor) -> Tensor:
        return self.default_h * torch.ones_like(t, dtype=t.dtype, device=t.device)

    @torch.no_grad()
    def __call__(self, xt: Tensor, *args, **kwargs) -> List[Tensor]:
        device = xt.device
        t = self.t_min * torch.ones(xt.shape[0], 1)
        x_ts = [xt.clone()]

        with tqdm(desc="Euler Sampling") as pbar:
            while t.max() <= 1 - self.default_h:
                ut = self.u(xt, t.to(device), *args, **kwargs)
                adapt_h = self.h(t)
                xt, *_ = self.apply_u(xt.cpu(), ut, adapt_h)
                t = t + adapt_h
                x_ts.append(xt.clone())
                xt = xt.to(device)
                pbar.update(1)

        return x_ts


#####################################################################################################
# Utility Functions
#####################################################################################################


def x2prob(x: Tensor, vocab_size: int) -> Tensor:
    """Converts sequence of tokens to class distribution representation"""
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def sample_p(pt: Tensor, temperature: float = 1.0) -> Tensor:
    """Samples protein sequence from class distribution representation"""
    if pt.ndim == 2:
        pt = pt.unsqueeze(0)
    b, l, _ = pt.shape
    pt = rearrange(pt, "b l c -> (b l) c")
    xt = torch.multinomial(pt / temperature, 1)
    return xt.reshape(b, l)


def sample_cond_pt(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, kappa: KappaScheduler):
    t = t.reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)
