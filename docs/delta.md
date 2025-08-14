Delta method with CUPED should work like the following. For each randomization unit i we observe $Y_i$, $N_i$.

Our estimator of the mean will have the form

$$
\frac{\sum_{i=1}^n Y_i - \theta (Z_i - E[Z_i])}{\sum_{i=1}^n N_i}
$$

where $Z_i$ is the covariate for unit $i$, $E[Z]$ is the average of the covariate across all units, and $\theta$ is a parameter that we estimate from the data.


In order to estimate $\theta$, we find the value that minimizes the variance of the estimator. The variance of the estimator can be expressed as:

$$
\begin{align}
\text{Var}\left( \frac{\sum_{i=1}^n Y_i - \theta \sum_{i=1}^n (Z_i - \bar{Z})}{\sum_{i=1}^n N_i} \right)
&= \text{Var}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i} \right)
\\
&\quad +  \theta^2 \text{Var}\left( \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right) \\
&\quad - 2 \theta \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right)
\end{align}
$$


By calculating the derivative of this variance with respect to $\theta$ and setting it to zero, we can find the optimal value of $\theta$ that minimizes the variance:
$$\begin{align}
2 \theta \text{Var}\left( \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right) - 2 \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right) &= 0
\end{align}$$

This gives us the optimal $\theta$ as:
$$\theta = \frac{\text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right)}{\text{Var}\left( \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right)}$$

If instead of a single covariate $Z$ we have multiple covariates, we can extend this to a vector of covariates $Z = (Z_1, Z_2, \ldots, Z_k)$ and compute the covariance and variance matrices accordingly. The formula for $\theta$ becomes:
