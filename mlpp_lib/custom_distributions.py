import torch
from torch.distributions import Distribution, constraints, Normal

class TruncatedNormalDistribution(Distribution):
    """
    Implementation of a truncated normal distribution in [a, b] with 
    differentiable sampling. 
    
    Source: The Truncated Normal Distribution, John Burkardt 2023
    """

    def __init__(self, mu_bar: torch.Tensor, sigma_bar: torch.Tensor, a: torch.Tensor,b: torch.Tensor):
        """_summary_

        Args:
            mu_bar (torch.Tensor): The mean of the underlying Normal. It is not the true mean.
            sigma_bar (torch.Tensor): The std of the underlying Normal. It is not the true std.
            a (torch.Tensor): The left boundary
            b (torch.Tensor): The right boundary
        """
        self._n = Normal(mu_bar, sigma_bar)
        self.mu_bar = mu_bar
        self.sigma_bar = sigma_bar
        super().__init__()
        
        self.a = a 
        self.b = b
        
        
    def icdf(self, p):
        # inverse cdf
        p_ = self._n.cdf(self.a) + p * (self._n.cdf(self.b) - self._n.cdf(self.a))
        return self._n.icdf(p_)
    
    def mean(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Returns the true mean of the distribution.
        """
        alpha = (self.a - self.mu_bar) / self.sigma_bar
        beta = (self.b - self.mu_bar) / self.sigma_bar
        
        sn = torch.distributions.Normal(torch.zeros_like(self.mu_bar), torch.ones_like(self.mu_bar))
        
        scale = (torch.exp(sn.log_prob(beta)) - torch.exp(sn.log_prob(alpha)))/(sn.cdf(beta) - sn.cdf(alpha))
        
        return self.mu_bar - self.sigma_bar * scale 
    
    def variance(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Returns the true variance of the distribution.
        """
        alpha = (self.a - self.mu_bar) / self.sigma_bar
        beta = (self.b - self.mu_bar) / self.sigma_bar
        
        sn = torch.distributions.Normal(torch.zeros_like(self.mu_bar), torch.ones_like(self.mu_bar))
        
        pdf_a = torch.exp(sn.log_prob(alpha))
        pdf_b = torch.exp(sn.log_prob(beta))
        CDF_a = sn.cdf(alpha)
        CDF_b = sn.cdf(beta)
        
        return self.sigma_bar**2 * (1.0 - (beta*pdf_b - alpha*pdf_a)/(CDF_b - CDF_a) - ((pdf_b - pdf_a)/(CDF_b - CDF_a))**2)
    
    
    def moment(self, k):
        # Source: A Recursive Formula for the Moments of a Truncated Univariate Normal Distribution  (Eric Orjebin)
        if k == -1:
            return torch.zeros_like(self.mu_bar)
        if k == 0:
            return torch.ones_like(self.mu_bar)
        
        alpha = (self.a - self.mu_bar) / self.sigma_bar
        beta = (self.b - self.mu_bar) / self.sigma_bar
        sn = torch.distributions.Normal(torch.zeros_like(self.mu_bar), torch.ones_like(self.mu_bar))
        
        scale = ((self.b**(k-1) * torch.exp(sn.log_prob(beta)) - self.a**(k-1) * torch.exp(sn.log_prob(alpha))) / (sn.cdf(beta) - sn.cdf(alpha)))
        
        return (k-1)* self.sigma_bar ** 2 * self.moment(k-2) + self.mu_bar * self.moment(k-1) - self.sigma_bar * scale
        
    def sample(self, shape):
        return self.rsample(shape)
    
    def rsample(self, shape):
        # get some random probability [0,1]
        p = torch.distributions.Uniform(0,1).sample(shape)
        # apply the inverse cdf on p 
        return self.icdf(p)
    
    @property
    def arg_constraints(self):
        return {
            'mu_bar': constraints.real,
            'sigma_bar': constraints.positive,
        }
        
    @property
    def has_rsample(self):
        return True
         
class CensoredNormalDistribution(Distribution):
    r"""Implements a censored Normal distribution. 
    Values of the underlying normal that lie outside the range [a,b] 
    are assigned to a and b respectively. 
    
    .. math::
        f_Y(y) =
            \begin{cases}
            a, & \text{if } y \leq a  \\
            \sim N(\bar{\mu}, \bar{\sigma})  & \text{if } a < y < b  \\
            b, & \text{if } y \geq b  \\
            \end{cases}

    
    """

    def __init__(self, mu_bar: torch.Tensor, sigma_bar: torch.Tensor, a: torch.Tensor,b: torch.Tensor):
        """
        Args:
            mu_bar (torch.Tensor): The mean of the latent normal distribution
            sigma_bar (torch.Tensor): The std of the latend normal distribution
            a (torch.Tensor): The lower bound of the distribution.
            b (torch.Tensor): The upper bound of the distribution.
        """
        
        
        self._n = Normal(mu_bar, sigma_bar)
        self.mu_bar = mu_bar
        self.sigma_bar = sigma_bar
        super().__init__()
        
        self.a = a 
        self.b = b
        

    def mean(self):
        alpha = (self.a - self.mu_bar) / self.sigma_bar
        beta = (self.b - self.mu_bar) / self.sigma_bar
        
        sn = torch.distributions.Normal(torch.zeros_like(self.mu_bar), torch.ones_like(self.mu_bar))
        E_z = TruncatedNormalDistribution(self.mu_bar, self.sigma_bar, self.a, self.b).mean()
        return (
            self.b * (1-sn.cdf(beta))
            + self.a * sn.cdf(alpha)
            + E_z * (sn.cdf(beta) - sn.cdf(alpha))
        )
        
        
    def variance(self):
        # Variance := Var(Y) = E(Y^2) - E(Y)^2
        alpha = (self.a - self.mu_bar) / self.sigma_bar
        beta = (self.b - self.mu_bar) / self.sigma_bar
        sn = torch.distributions.Normal(torch.zeros_like(self.mu_bar), torch.ones_like(self.sigma_bar))
        tn = TruncatedNormalDistribution(mu_bar=self.mu_bar, sigma_bar=self.sigma_bar, a=self.a, b=self.b)
        
        # Law of total expectation:
        # E(Y^2)    = E(Y^2|X>b)*P(X>b) + E(Y^2|X<a)*P(X<a) + E(Y^2 | a<X<b)*P(a<X<b)
        #           = b^2 * P(X>b)       + a^2 * P(X<a)     + E(Z^2~TruncNormal(mu,sigma,a,b)) * P(a<X<b)
       
        E_z2 = tn.moment(2) # E(Z^2)
        E_y2 =  self.b**2 * (1-sn.cdf(beta)) + self.a**2 * sn.cdf(alpha) + E_z2 * (sn.cdf(beta) - sn.cdf(alpha)) # E(Y^2)
        
        return E_y2 - self.mean()**2 # Var(Y)=E(Y^2)-E(Y)^2
        

    def sample(self, shape):
        # note: clipping degenerates the gradients. 
        # Do not use for MC optimization. 
        s = self._n.sample(shape)
        return torch.clip(s, min=self.a, max=self.b)
    
    @property
    def arg_constraints(self):
        return {
        'mu_bar': constraints.real,
        'sigma_bar': constraints.positive,  # Enforce positive scale
        }