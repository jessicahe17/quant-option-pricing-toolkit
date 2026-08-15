# Monte Carlo Validation

## 1. Objective

This experiment validates the Monte Carlo pricing engine against the analytical Black-Scholes price for a European call option.

The validation has two objectives:

1. Verify that the Monte Carlo estimator is statistically consistent with the Black-Scholes benchmark.
2. Empirically verify the theoretical Monte Carlo convergence rate, where the standard error decreases at a rate proportional to $N^{-1/2}$.


## 2. Risk-Neutral Pricing Framework

In derivative pricing, the arbitrage-free value $V_0$ of a European option at time $t = 0$ is defined as the discounted expected payoff under the risk-neutral pricing measure $\mathbb{Q}$:

$$V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}} [\Pi(S_T)]$$

Under the risk-neutral measure $\mathbb{Q}$, the asset price process $S_t$ follows a Geometric Brownian Motion (GBM) governed by the stochastic differential equation:

$$dS_t = (r - q)S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}$$

where $r$ is the risk-free rate, $q$ is the continuous dividend yield, $\sigma$ is the volatility, and $W_t^{\mathbb{Q}}$ is a standard Brownian motion under $\mathbb{Q}$.

Integrating this stochastic differential equation yields the exact terminal asset price at maturity $T$:

$$S_T = S_0 \exp \left[ \left( r - q - \frac{1}{2}\sigma^2 \right) T + \sigma \sqrt{T} Z \right], \quad Z \sim \mathcal{N}(0,1)$$

The Monte Carlo simulation therefore uses the risk-neutral drift $r - q$, rather than an estimated historical or physical-measure expected return. This allows the simulated discounted payoff to be used for no-arbitrage derivative pricing.


## 3. Validation Methodology

### 3.1 Black-Scholes Benchmark

The analytical Black-Scholes model was evaluated using the exact same market parameters and option specification as the Monte Carlo model.

| Parameter | Symbol | Value |
| :--- | :--- | :--- |
| Spot Price | $S_0$ | 100 |
| Strike Price | $K$ | 100 |
| Volatility | $\sigma$ | 20% |
| Risk-free Rate | $r$ | 5% |
| Dividend Yield | $q$ | 2% |
| Maturity | $T$ | 1 year |

Using these parameters, the analytical Black-Scholes benchmark price for the European call option was evaluated:

$$V_{\text{BS}} = 9.227006$$

### 3.2 Monte Carlo Estimate

The Monte Carlo experiment simulated 100,000 risk-neutral GBM paths with 252 time steps per year using a random seed of 42.

For each path $i$, the terminal underlying asset value $S_T^{(i)}$ was passed to the European call option payoff function:

$$\Pi(S_T^{(i)}) = \max\left(S_T^{(i)} - K, 0\right)$$

The discounted sample mean was then calculated as the Monte Carlo price estimate $\hat{V}_{\text{MC}}$:

$$\hat{V}_{\text{MC}} = e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \Pi(S_T^{(i)})$$

The statistical precision of the Monte Carlo estimate is quantified by the standard error (SE):

$$\text{SE}(\hat{V}_{\text{MC}}) = \frac{s_{\Pi}}{\sqrt{N}}$$

where $s_{\Pi}$ is the sample standard deviation of the discounted payoff observations.

### 3.3 Statistical Validation

Because Monte Carlo simulation relies on random sample paths, we do not expect $\hat{V}_{\text{MC}} = V_{\text{BS}}$ exactly. Instead, we assess statistical consistency using the standardized error and a confidence interval.

We calculate the standardized error $Z$, which measures the deviation between the Monte Carlo estimate and the analytical benchmark in units of Monte Carlo standard errors:

$$Z = \frac{\hat{V}_{\text{MC}} - V_{\text{BS}}}{\text{SE}(\hat{V}_{\text{MC}})}$$

Using the standard normal critical value for a 95% confidence level ($z_{0.025} \approx 1.96$), the 95% confidence interval for the estimated option price is constructed as:

$$\text{CI}_{95\%} = \hat{V}_{\text{MC}} \pm 1.96 \cdot \text{SE}(\hat{V}_{\text{MC}})$$


## 4. Results

### 4.1 Monte Carlo vs. Black-Scholes

The main validation experiment yielded the following comparison between the Monte Carlo engine and the Black-Scholes benchmark:

| Metric | Result |
| :--- | :--- |
| Black-Scholes price | 9.227006 |
| Monte Carlo price | 9.258790 |
| Standard error | 0.044089 |
| Absolute difference | 0.031785 |
| Standardized error ($Z$) | 0.7209 |
| 95% confidence interval | [9.172376, 9.345204] |
| BS price inside CI | Yes |

The Monte Carlo estimate is approximately 0.72 standard errors above the Black-Scholes benchmark. The Black-Scholes price lies within the 95% Monte Carlo confidence interval.

Therefore, the Monte Carlo estimate is statistically consistent with the analytical Black-Scholes benchmark at the 95% confidence level for this parameter configuration.

### 4.2 Convergence Analysis

To evaluate the empirical convergence rate of the Monte Carlo estimator, the experiment was repeated across varying numbers of sample paths $N \in \{5\,000, 10\,000, 25\,000, 50\,000, 100\,000, 250\,000\}$.

| Paths ($N$) | MC Price | Absolute Error | Standard Error | $\text{SE} \cdot \sqrt{N}$ |
| :--- | :--- | :--- | :--- | :--- |
| 5,000 | 9.558189 | 0.331183 | 0.200544 | 14.180570 |
| 10,000 | 9.370442 | 0.143436 | 0.139766 | 13.976604 |
| 25,000 | 9.138996 | 0.088009 | 0.086631 | 13.697536 |
| 50,000 | 9.211793 | 0.015213 | 0.062093 | 13.884433 |
| 100,000 | 9.262359 | 0.035353 | 0.043857 | 13.868764 |
| 250,000 | 9.294959 | 0.067954 | 0.027819 | 13.909313 |

Under Monte Carlo theory, the standard error decreases according to $\text{SE} \propto N^{-1/2}$, which implies that the product $\text{SE} \cdot \sqrt{N}$ should remain approximately constant:

$$\text{SE} \cdot \sqrt{N} \approx C$$

Across all runs, the empirical product $\text{SE} \cdot \sqrt{N}$ remains highly stable between 13.7 and 14.2.

A log-log regression of standard error against the number of paths produced an empirical slope of **-0.5040**, which is extremely close to the theoretical value of **-0.5**. This provides empirical evidence that the Monte Carlo estimator exhibits the expected $\mathcal{O}(N^{-1/2})$ convergence rate.

### 4.3 Convergence Visualization

The convergence behavior is illustrated below.

#### Monte Carlo Price Convergence

![Monte Carlo price convergence](figures/mc_price_convergence.png)

#### Standard Error Convergence

![Monte Carlo standard error convergence](figures/mc_standard_error_convergence.png)

The dashed reference line has a theoretical slope of $-1/2$,
corresponding to the expected Monte Carlo standard-error scaling $SE \propto N^{-1/2}$. The empirical standard-error curve closely follows this reference, with the slope of -0.5040.

## 5. Interpretation

### Non-Monotonic Price Convergence

A key quantitative observation from the empirical results is that the Monte Carlo price estimate does not converge monotonically toward the Black-Scholes price. For example, the absolute error increases between 50,000 paths (0.015213) and 100,000 paths (0.035353), and further increases at 250,000 paths (0.067954).

This behavior is expected because Monte Carlo estimates are random. Increasing the number of sample paths systematically reduces the variance and standard error of the estimator, shrinking the width of the confidence interval. However, the reduction in estimator variance does not guarantee that every individual sample realization with a higher path count will land closer to the analytical benchmark than a lower-path realization.

Thus, a critical distinction must be drawn:
* **Precision improves systematically** as $N$ increases, with the standard error exhibiting the theoretical $\mathcal{O}(N^{-1/2})$ scaling.
* **Realized error does not necessarily decrease monotonically** due to residual stochastic sampling noise.


## 6. Conclusion

The Monte Carlo pricing engine produces estimates statistically consistent with the analytical Black-Scholes benchmark for the tested European call option. The convergence experiment further demonstrates the expected $N^{-1/2}$ scaling in Monte Carlo standard error, with an empirical convergence slope of -0.5040.

Together, these experiments provide quantitative validation of the Monte Carlo simulation and pricing components under the Black-Scholes/GBM framework.