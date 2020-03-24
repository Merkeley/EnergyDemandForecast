SARIMAX Equations - 
Seasonal Auto-Regressive Integrating Moving Average with Exogenous Features

At any time t the output can be written as a function of past samples, a constant and noise.

$$y_t = {\beta} + {\epsilon} + {\sum_{i=1}^p{{\theta} y_{t-i}}}$$

where:
$${\beta}\; {\equiv}\;Constant$$
<br>
$${\epsilon}\; {\equiv}\;Noise$$
<br>
$${p}\; {\equiv}Number\; of\; lags\; for\; the\; output$$ 
<br><br>
If we define the lag operator L we can rewrite this as:

$$ {L^n}{y_t} = {y_{t-n}}$$

We can also define the polynomial function ${\theta(L)^p}$ of order p we can rewrite the equation as:
$$ y_{t} = {\theta(L)^{p}y_{t} + {\epsilon_t}} $$



<br>
<br>

$${\Delta^{p}_s} = \frac{True Positive}{True Positive + False Positive}$$