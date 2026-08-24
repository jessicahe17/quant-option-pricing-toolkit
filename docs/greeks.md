# Black-Scholes Greeks

## Overview

The Black-Scholes model provides analytical sensitivities of option value with
respect to key market variables. These sensitivities are known as the Greeks.

This library currently supports:

- Delta
- Gamma
- Vega
- Theta
- Rho

All Greeks are calculated analytically within the Black-Scholes framework.



## Conventions

The Greeks returned by this library follow mathematical definitions and are
reported per underlying unit.

The option contract multiplier is not applied.

Examples:

- Vega is expressed per 1.00 change in volatility.
- Theta is annualized.
- Contract-level Greeks can be obtained by multiplying by
  `Option.multiplier`.


When time to expiry is non-positive or volatility is zero, the current implementation returns `None` for the Greeks rather than imposing limiting-value conventions.


## Delta

Delta measures sensitivity to the underlying asset price:

$$
\Delta=\frac{\partial V}{\partial S}
$$

Call:

$$
\Delta_c=e^{-qT}N(d_1).
$$

Put:

$$
\Delta_p=e^{-qT}(N(d_1)-1).
$$



## Gamma

Gamma measures the sensitivity of Delta to changes in the underlying price:

$$
\Gamma=\frac{\partial^2V}{\partial S^2}.
$$

Formula:

$$
\Gamma=
\frac{e^{-qT}N'(d_1)}
{S\sigma\sqrt{T}}.
$$

Gamma is identical for European calls and puts with the same parameters.



## Vega

Vega measures sensitivity to volatility:

$$
\nu=\frac{\partial V}{\partial\sigma}
$$

Formula:

$$
\nu=
Se^{-qT}N'(d_1)\sqrt{T}.
$$



## Theta

Theta measures sensitivity to the passage of time:

$$
\Theta=\frac{\partial V}{\partial t}.
$$


For a European call,

$$
\Theta_c =
-\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt{T}}
+qSe^{-qT}N(d_1)
-rKe^{-rT}N(d_2).
$$

For a European put,

$$
\Theta_p =
-\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt{T}}
-qSe^{-qT}N(-d_1)
+rKe^{-rT}N(-d_2).
$$

The library reports annualized Theta; daily Theta can be approximated as:

$$
\Theta_{daily}=\frac{\Theta}{365}
$$



## Rho

Rho measures sensitivity to the risk-free interest rate:

$$
\rho=\frac{\partial V}{\partial r}
$$

Call:

$$
\rho_c=KT e^{-rT}N(d_2).
$$

Put:

$$
\rho_p=-KT e^{-rT}N(-d_2).
$$


