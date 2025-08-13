Delta method with CUPED should work like the following. For each randomization unit i we observe $Y_i$, $N_i$.

Our estimator of the mean will have the form

$$
\frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i} - \theta \left( \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right)
$$

where $Z_i$ is the covariate for unit $i$, $E[Z]$ is the average of the covariate across all units, and $\theta$ is a parameter that we estimate from the data.

In order to estimate $\theta$, we find the value that minimizes the variance of the estimator. The variance of the estimator can be expressed as:
$$
\begin{align}
\text{Var}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i} - \theta \left(
\frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right) \right) &= \text{Var}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i} \right) \\
&\quad + \theta^2 \text{Var}\left( \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right) \\
&\quad - 2 \theta \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right)
\end{align}
$$
By calculating the derivative of this variance with respect to $\theta$ and setting it to zero, we can find the optimal value of $\theta$ that minimizes the variance:
$$\begin{align}
2 \theta \text{Var}\left( \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right) - 2 \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right) &= 0
\end{align}$$

This gives us the optimal $\theta$ as:
$$\theta = \frac{\text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right)}{\text{Var}\left( \frac{\sum_{i=1}^n Z_i}{n} - E[Z] \right)}$$

Since $E[Z]$ is a constant, we can simplify both terms involving $E[Z]$ in the variance expression. The $\theta$ will be
$$\theta = \frac{\text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} \right)}{\text{Var}\left( \frac{\sum_{i=1}^n Z_i}{n}\right)}$$

If instead of a single covariate $Z$ we have multiple covariates, we can extend this to a vector of covariates $Z = (Z_1, Z_2, \ldots, Z_k)$ and compute the covariance and variance matrices accordingly. The formula for $\theta$ becomes:
$$\theta = \text{Var}\left(
    \frac{\sum_{i=1}^n Z_i}{n}
\right)^{-1} \cdot \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i},
\frac{\sum_{i=1}^n Z_i}{n} \right)$$
Where $\text{Var}\left(\frac{\sum_{i=1}^n Z_i}{n}\right)$ is the covariance matrix of the covariate means and $\text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} \right)$ is the vector of covariances between the ratio and each covariate. They can be simplified to:
$$\text{Var}\left(\frac{\sum_{i=1}^n Z_i}{n}\right) = \frac{1}{n} \Sigma_Z^2$$


Therefore,

$$
\theta =  \frac{(\Sigma_Z^2)^{-1}}{n} \cdot \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} \right)
$$

For a single covariate, this simplifies to:
$$
\theta = \frac{\text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{n} \right)}{\frac{1}{n} \text{Var}\left( Z\right)}
$$

How can I estimate this covariance?
