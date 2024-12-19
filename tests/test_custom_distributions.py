import pytest
import torch 

from mlpp_lib.custom_distributions import TruncatedNormalDistribution, CensoredNormalDistribution

@pytest.mark.parametrize("ab", [(4.8, 6.2), (4.0, 5.5), (4.2, 5.8)], ids=['left capped', 'right capped', 'centered'])
def test_truncated_normal(ab):
    
    a,b = ab
    
    tn = TruncatedNormalDistribution(mu_bar=torch.tensor(5.0), sigma_bar=torch.tensor(0.5), a=torch.tensor(a), b=torch.tensor(b))
    
    
    samples = tn.sample((50000,))
    
    empirical_mean = samples.mean()
    empirical_var = samples.var()
    
    
    decimal_places = 2
    tolerance = 10**-decimal_places
    
    # test mean and variance
    assert  torch.allclose(empirical_mean, tn.mean(), atol=tolerance)
    assert  torch.allclose(empirical_var, tn.variance(), atol=tolerance)
    
    # test first moment
    decimal_places = 8
    tolerance = 10**-decimal_places
    assert torch.allclose(tn.mean(), tn.moment(1))
    
    
@pytest.mark.parametrize("ab", [(4.8, 6.2), (4.0, 5.5), (4.2, 5.8)], ids=['left capped', 'right capped', 'centered'])
def test_censored_normal(ab):
    
    a,b = ab
    
    cn = CensoredNormalDistribution(mu_bar=torch.tensor(5.0), sigma_bar=torch.tensor(0.5), a=torch.tensor(a), b=torch.tensor(b))
    
    
    samples = cn.sample((50000,))
    
    empirical_mean = samples.mean()
    empirical_var = samples.var()
    
    
    decimal_places = 2
    tolerance = 10**-decimal_places
    
    # test mean and variance
    assert  torch.allclose(empirical_mean, cn.mean(), atol=tolerance)
    assert  torch.allclose(empirical_var, cn.variance(), atol=tolerance)
    
