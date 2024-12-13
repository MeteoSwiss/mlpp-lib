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