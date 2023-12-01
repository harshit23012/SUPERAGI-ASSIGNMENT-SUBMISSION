## Question 5

### Estimates for Probability \( p \)

1. **Maximum Likelihood Estimate (MLE)**

   MLE estimates the parameter $p$ by maximizing the likelihood function. For a binomial distribution, the likelihood function is given by the binomial probability of observing $k$ heads in $n$ flips:

   $ \mathcal{L}(p) = P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $

   The MLE estimate of $ p $ is the value that maximizes this likelihood. For the binomial distribution, this is simply the proportion of heads observed:

   $ p_{\text{MLE}} = \frac{k}{n} $

2. **Bayesian Estimate**

   For the Bayesian estimate, we use a uniform prior for $ p $ over the interval $[0,1]$. The posterior distribution for a binomial likelihood with a uniform prior is a Beta distribution, specifically $\beta(k+1, n-k+1) $. The expected value of this Beta distribution is the Bayesian estimate for $ p $:

   $ p_{\text{Bayesian}} = \frac{k + 1}{n + 2} $

3. **Maximum a Posteriori (MAP) Estimate**

   The MAP estimate is the mode of the posterior distribution. For the Beta distribution $\beta(k+1, n-k+1)$, the mode is given by:

   $ p_{\text{MAP}} = \frac{k + 1 - 1}{n + 2 - 2} = \frac{k}{n} $

   However, if $ k = 0 $ or $ k = n $, the mode is not well-defined (it's at the boundaries of the distribution). In such cases, we might revert to using the mean of the distribution (i.e., the Bayesian estimate).

**In summary:**
- MLE Estimate: $ p_{\text{MLE}} = \frac{k}{n} $
- Bayesian Estimate: $ p_{\text{Bayesian}} = \frac{k + 1}{n + 2} $
- MAP Estimate: $ p_{\text{MAP}} = \frac{k}{n} $ (with special considerations for $ k = 0 $ or $ k = n $
