# SARIMAX Equations

### Seasonal Auto-Regressive Integrating Moving Average with Exogenous Features

**Autoregression**

At any time t the output can be written as a function of past samples, a constant and noise.

$$y_t = {\beta} + {\epsilon} + {\sum_{i=1}^p{{\theta} y_{t-i}}}$$

where:
$${\beta}\; {\equiv}\;Constant$$

$${\epsilon}\; {\equiv}\;Noise$$

$${p}\; {\equiv}Number\; of\; lags\; for\; the\; output$$ 

If we define the lag operator L we can rewrite this as:

$$ {L^n}{y_t} = {y_{t-n}}$$

We can also define the polynomial function ${\Theta(L)^p}$ of order p we can rewrite the equation as:
$$ y_{t} = {\Theta(L)^{p}y_{t} + {\epsilon_t}} $$

Note that ${\beta}$ is part of ${\theta}$.

**Moving Average**

$$y_{t} = {\Phi(L)^{q}}{\epsilon_{t}} + {\epsilon}$$

where q is the number of lags for the error ${\epsilon}$ and ${\Phi}$ is the polinomial function of L with order q.

Therefore, the ARMA (Autoregressive Moving Average) model is:

$$y_{t} = {\Theta(L)^{p}}*y_{t} + {\Phi(L)^{q} \epsilon_{t}} + {\epsilon_{t}}$$

**Integration Operator**

The integration component of the model helps with non-stationarity of data.  If we define the ${\Delta^{d}}$ as follows:
$$y_{t}^{[d]} = {\Delta^{d}}*y_{t} = y_{t}^{[d-1]} - y_{t-1}^{[d-1]}$$
where $y^{[0]}$ is $y_{t}$ and d is the order of the differencing used.  Substituting $y_{t}^{[d]}$ into our ARMA equation.

$$y_{t}^{[d]} = {\Theta(L)^{p}}y_{t}^{[d]} + {\Phi(L)^{q}}{\epsilon_{t}^{[d]}} + {\epsilon_{t}^{[d]}}$$

We then get the following ARIMA equation:
$${\Delta^{d}}y_{t} = {\Theta(L)^{p}}{\Delta^{d}}y_{t} + {\Phi(L)^{q}}{\Delta^{d}}{\epsilon_{t}} + {\Delta^{d}}{\epsilon_t}$$

Which can be written in the following simplified form:

$${\Theta(L)^{p}}{\Delta^{d}}y_{t} = {\Phi(L)^{q}}{\Delta^{d}}{\epsilon_{t}}$$

**Seasonal Components**

Seasonal components are handled with a similar equation applied to lags if seasonal period s.  We can define a seasonal difference ${\Delta^{D}_{s}}$ with D is the differencing for seasonal lags.  If we let $L^{s}$ ${\equiv}$ seasonal lag operator and P and Q are also seasonal lags then we can write our SARIMA equation as:

$${\Delta^{D}_{s}}y_{t} = {\theta(L^{s})^{P}}{\Delta^{D}_{s}}y_{t} + {\phi(L^{s})^{Q}}{\Delta^{D}_{s}}{\epsilon_{t}}$$

Which simplifies to:
$${\theta(L^{s})^{P}}{\Delta^{D}_{s}}y_{t} = {\phi(L^{s})^{Q}}{\Delta^{D}_{s}}{\epsilon_{t}}$$

Therefore, the general SARIMAX equation becomes:

$${\Theta(L)^{p}}{\theta(L^s)^{P}}{\Delta^{d}}{\Delta^{D}_{s}}y_{t} = {\Phi(L)^{q}}{\phi(L^{s})^{Q}}{\Delta^{d}}{\Delta^{D}_{s}}{\epsilon_{t}}$$

**Exogenous Regressors**

Adding exogenous we get ARIMA becomes ARIMAX:
$${\Theta(L)^{p}}{\Delta^{d}}y_{t} = {\Phi(L)^{q}}{\Delta^{d}}{\epsilon_{t}} + {\sum_{i=1}^{n}\beta_{i}x_{t}^{i}}$$

and SARIMA becomes SARIMAX:

$${\Theta(L)^{p}}{\theta(L^s)^{P}}{\Delta^{d}}{\Delta^{D}_{s}}y_{t} = {\Phi(L)^{q}}{\phi(L^{s})^{Q}}{\Delta^{d}}{\Delta^{D}_{s}}{\epsilon_{t}} +
{\sum_{i=1}^{n}\beta_{i}x_{t}^{i}}$$






$${\Delta^{p}_s} = \frac{True Positive}{True Positive + False Positive}$$