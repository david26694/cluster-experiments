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


Using linearisation (Taylor expansion) we can also express the optimal $\theta$ as:
$$
\theta = \frac{
    \text{Cov}\left(
            \frac{\bar{Y}}{E[N]} -
            \frac{\bar{N} E[Y]}{E[N]^2},
        \frac{\bar{Z}}{E[N]} - \frac{E[Z] \bar{N}}{E[N]^2}
    \right)
}{
    \text{Var}\left(
        \frac{\bar{Z}}{E[N]} -
        \frac{E[Z] \bar{N}}{E[N]^2}
    \right)
}
$$
Where $\bar{Y} = \frac{1}{n} \sum_{i=1}^n Y_i$, $\bar{Z} = \frac{1}{n} \sum_{i=1}^n Z_i$, and $\bar{N} = \frac{1}{n} \sum_{i=1}^n N_i$ are the sample means of $Y$, $Z$, and $N$ respectively.

Because the user is the randomisation unit and user level observations are iid, we have:
$$
\sqrt(n) (\bar{Y}, \bar{N}, \bar{Z}, \bar{N}) \xrightarrow{d} N(\mu, \Sigma)
$$
Where $\mu = (E[Y], E[N], E[Z], E[N])$ and $\Sigma$ is the covariance matrix of $(Y, N, Z, N)$.

If we define
$$
\beta_1 = (1 / E[N], - E[Y] / E[N]^2, 0, 0)^T
$$
$$
\beta_2 = (0, 0, 1 / E[N], - E[Z] / E[N]^2)^T
$$
then we can express the optimal $\theta$ as:
$$
\theta = \beta_1^T \Sigma \beta_2 / \beta_2^T \Sigma \beta_2
$$

In this case, the variance of the estimator can be expressed as:
$$\begin{align}
\text{Var}\left( \frac{\sum_{i=1}^n Y_i - \theta \sum_{i=1}^n (Z_i - \bar{Z})}{\sum_{i=1}^n N_i} \right)
&= \text{Var}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i} \right)
\\
&\quad +  \theta^2 \text{Var}\left( \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right) \\
&\quad - 2 \theta \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right)
\end{align}
$$


If instead of a single covariate $Z$ we have multiple covariates, we can extend this to a vector of covariates $\bold{Z} = (Z_1, Z_2, \ldots, Z_k)$ and compute the covariance and variance matrices accordingly.

In this case, the variance of the estimator becomes:
$$\begin{align}
\text{Var}\left( \frac{\sum_{i=1}^n Y_i - \theta^T \sum_{i=1}^n (Z_i - \bar{Z})}{\sum_{i=1}^n N_i} \right)
&= \text{Var}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i} \right)
\\
&\quad +  \theta^T \text{Var}\left( \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right) \theta
\\
&\quad - 2 \theta^T \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n Z_i}{\sum_{i=1}^n N_i} \right)
\end{align}
$$

The optimal $\theta$ in this case is given by:
$$\theta = \text{Var}\left( \frac{\sum_{i=1}^n \bold{Z}_i}{\sum_{i=1}^n N_i} \right)^{-1} \cdot \text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n \bold{Z}_i}{\sum_{i=1}^n N_i} \right) $$

This can be computed using matrix operations, where $\theta$ is a vector of coefficients corresponding to each covariate in $\bold{Z}$. The term $\text{Cov}\left( \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n N_i}, \frac{\sum_{i=1}^n \bold{Z}_i}{\sum_{i=1}^n N_i} \right)$ is a vector of covariances between the outcome and each covariate, and $\text{Var}\left( \frac{\sum_{i=1}^n \bold{Z}_i}{\sum_{i=1}^n N_i} \right)$ is the covariance matrix of the covariates.

Using linearisation, the optimal $\theta$ can also be expressed as:
$$
\theta =
\text{Var}\left(\frac{\bar{\bold{Z}}}{E[N]} - \frac{E[\bold{Z}] \bar{N}}{E[N]^2}\right)^{-1} \cdot
\text{Cov}\left(\frac{\bar{Y}}{E[N]} - \frac{\bar{N} E[Y]}{E[N]^2}, \frac{\bar{\bold{Z}}}{E[N]} - \frac{E[\bold{Z}] \bar{N}}{E[N]^2}\right)
$$

If we have multiple covariates, we consider the vector $(Y, N, Z_1, Z_2, \ldots, Z_k, N)$ where $Z_1, Z_2, \ldots, Z_k$ are the covariates.

For
$$
\text{Cov}\left(\frac{\bar{Y}}{E[N]} - \frac{\bar{N} E[Y]}{E[N]^2}, \frac{\bar{\bold{Z}}}{E[N]} - \frac{E[\bold{Z}] \bar{N}}{E[N]^2}\right)
$$, its component $l$ is given by:
$$\text{Cov}\left(\frac{\bar{Y}}{E[N]} - \frac{\bar{N} E[Y]}{E[N]^2}, \frac{\bar{Z_l}}{E[N]} - \frac{E[Z_l] \bar{N}}{E[N]^2}\right)
$$
By expanding the covariance, we get:
$$\begin{align}
&= \frac{\text{Cov}(\bar{Y}, \bar{Z_l})}{E[N]^2} - \frac{E[Y] \text{Cov}(\bar{N}, \bar{Z_l})}{E[N]^3} - \frac{E[Z_l] \text{Cov}(\bar{Y}, \bar{N})}{E[N]^3} + \frac{E[Y] E[Z_l] \text{Var}(\bar{N})}{E[N]^4}
\end{align}$$

If we define the vectors:
$$cov(Y, \bold{Z}) = (\text{Cov}(Y, Z_1), \text{Cov}(Y, Z_2), \ldots, \text{Cov}(Y, Z_k))^T$$
$$cov(N, \bold{Z}) = (\text{Cov}(N, Z_1), \text{Cov}(N, Z_2), \ldots, \text{Cov}(N, Z_k))^T$$
$$E[\bold{Z}] = (E[Z_1], E[Z_2], \ldots, E[Z_k])^T$$

Then we can express the covariance vector as:
$$\text{Cov}\left(\frac{\bar{Y}}{E[N]} - \frac{\bar{N} E[Y]}{E[N]^2}, \frac{\bar{\bold{Z}}}{E[N]} - \frac{E[\bold{Z}] \bar{N}}{E[N]^2}\right) = \frac{cov(Y, \bold{Z})}{E[N]^2} - \frac{E[Y] cov(N, \bold{Z})}{E[N]^3} - \frac{E[\bold{Z}] \text{Cov}(Y, N)}{E[N]^3} + \frac{E[Y] E[\bold{Z}] \text{Var}(N)}{E[N]^4}
$$

I have a matrix $A = (a_{ij})$ where $a_{ij} = \text{Cov}(Z_i, Z_j)$ for $i, j = 1, 2, \ldots, k$ and a vector $b = (b_i)$ where $b_i = \text{Cov}(Z_i, N)$ for $i = 1, 2, \ldots, k$. I build the matrix:
$$ D =\begin{bmatrix}
A & b \\
b^T & \text{Var}(N)
\end{bmatrix}$$

I have a matrix $$c_{ij} = x^2 a_{ij} + y_i y_j \text{Var}(N) - x y_j b_i - x y_i b_j$$
for some constants $x$ and a vector $\bold{y} = (y_i)$.

I want to get a matrix $K$ such that:
$$ K D K^T = C $$

If
$$ K = \begin{bmatrix}x I & -\bold{y}
\end{bmatrix}$$
where $I$ is the identity matrix of size $k$ and $\bold{y}$ is a column vector of ones of size $k$, then we have:
$$ K D K^T = C $$

In our case, we have $x = 1 / E[N]$ and $\bold{y} = E[\bold{Z} ] / E[N]^2$.
