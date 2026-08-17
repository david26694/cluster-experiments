# Which power equation matches the relative-lift test we actually run?

## The question

For a relative effect the standard error depends on the effect size. Three power
equations are in circulation in this repo, and they disagree:

| | power equation | where it appears |
|---|---|---|
| **(a)** | `power = Φ( (m − z_α·SE(0)) / SE(m) )` | `MDE-new-ratio-metrics`, `_effect_dependent_mde_calculation` |
| **(b)** | `power = Φ( m/SE(m) − z_α )` | `slides/relative_lift_ols.tex`, `test_relative_mde_quadratic.py` |
| **(c)** | `power = Φ( (m − x\*) / SE(m) )` | derived below |

They differ only in the **rejection threshold** they assume. This note derives
the threshold the library's own test actually uses, shows it is neither (a)'s nor
(b)'s, and quantifies the gap.

---

## 1. Setup

Write the relative lift as `L = τ / Y`, numerator over baseline. The delta method
gives its variance as a quadratic in the true lift `m` — this is exactly
`StandardErrorCurve`:

```
SE(m)² = A + B·m² − 2C·m
```

with `A = SE(0)²`, `B = effect_var`, `C = effect_cov`. For the delta method on a
ratio metric the arms are independent, so `C = −B` and this collapses to the
familiar form

```
SE(m)² = σ²_t/r_c² + σ²_c/r_c² · (1 + m)²
```

Throughout, `A`, `B`, `C` are treated as known constants. This is the standard
plug-in approximation and **all three conventions make it** — it is not what
separates them.

## 2. What the library computes at analysis time

This is the crux, and it is settled by reading the code rather than by argument.

`LiftRegressionTransformer.fit` and `DeltaMethodLiftTransformer.fit` both compute
the standard error at the **observed** lift `L̂`
([relative_lift_transformer.py:213](cluster_experiments/relative_lift_transformer.py#L213)):

```python
var_percent_lift = se_null_squared + effect_var * L̂**2 - 2 * effect_cov * L̂
```

and the p-value is the Wald statistic built from it
([relative_lift_transformer.py:96](cluster_experiments/relative_lift_transformer.py#L96)):

```python
z_score = self._relative_lift_value / self._se_relative_lift
```

So, writing `SE(·)` for the curve, the test reports significance when

```
|L̂| / SE(L̂) > z_α
```

**The denominator is a deterministic, increasing function of the numerator.**
Not an independent variance estimate — the same random quantity appears on both
sides. Everything below follows from that.

## 3. The rejection region is a threshold on `L̂`

Consider the one-sided (`greater`) case. Reject when

```
L̂ > z_α · SE(L̂)
```

Define `g(x) = x − z_α·SE(x)`. Then `g(0) = −z_α·SE(0) < 0`, and as `x → ∞`,
`SE(x) ≈ √B·x`, so `g(x) ≈ x(1 − z_α√B)`, which is positive provided
`z_α√B < 1`. And `g` is strictly increasing on `x > 0` whenever it matters, since

```
g′(x) = 1 − z_α · (B·x − C) / SE(x)
```

and `(B·x − C)/SE(x) → √B` from below. So `g` has exactly one positive root
`x\*`, and

```
reject  ⟺  L̂ > x\*        where   x\* = z_α · SE(x\*)
```

**The Wald test with an effect-dependent standard error is exactly a simple
threshold test on the estimate**, with the threshold given by a fixed point.
This is the step both (a) and (b) miss.

### Closed form for `x\*`

`SE(x)²` is quadratic, so squaring `x = z_α·SE(x)` gives a quadratic in `x`:

```
(1 − z_α²·B)·x²  +  2·z_α²·C·x  −  z_α²·A  =  0
```

Take the positive root. No iteration needed — the same shape as the MDE solve.

## 4. Power and the MDE

Under a true effect `m`, `L̂ ~ N(m, SE(m)²)` asymptotically, so

```
power(m) = P(L̂ > x\*) = Φ( (m − x\*) / SE(m) )
```

and setting `power(m) = 1 − β` gives

```
m = x\* + z_β · SE(m)          ← convention (c)
```

Compare the three, all of the form `m = threshold + z_β·SE(m)`:

| | threshold |
|---|---|
| (a) | `z_α · SE(0)` |
| (b) | `z_α · SE(m)` |
| (c) | `z_α · SE(x\*)` — the fixed point |

## 5. Ordering: (c) lies strictly between (a) and (b)

`SE` is increasing on the relevant range, so the ordering follows from ordering
the arguments `0`, `x\*`, `m`.

**(a) understates.** `x\* > 0`, so `SE(x\*) > SE(0)`, so `threshold_a < threshold_c`.
A smaller threshold means higher predicted power, hence a **smaller** MDE.

**(b) overstates.** At the MDE, `power > 50%` requires `m > x\*` (since
`power(x\*) = Φ(0) = 0.5`). Hence `SE(m) > SE(x\*)` and `threshold_b > threshold_c`,
giving a **larger** MDE.

```
MDE(a)  <  MDE(c)  <  MDE(b)
```

with equality throughout when `B = C = 0`, i.e. for absolute effects — which is
why no absolute-effect number moves under any of them.

## 6. Where each one goes wrong

**(a)** assumes the threshold is fixed at `z_α·SE(0)`. That is the correct
threshold for a **score-type** test, which uses the null-restricted variance in
the denominator — the pooled-variance two-proportion z-test is the standard
example. It is not what this library computes.

**(b)** comes from treating the Wald statistic as `Z ~ N(m/SE(m), 1)`. That
requires `ŜE` to be an estimate of `SE(m)` that is *independent* of the numerator.
But `ŜE = SE(L̂)` is perfectly correlated with `L̂`: a large draw of `L̂` inflates
its own denominator. Ignoring that correlation is what pushes the threshold out to
`SE(m)`, and it is why (b) diverges fastest.

## 7. A consequence worth knowing: the test is conservative

Since `x\* > z_α·SE(0)`, the actual size is

```
size = P(L̂ > x\* | m = 0) = 1 − Φ( x\* / SE(0) )   <   α
```

With `SE(0) = 5%` and `α = 0.05` one-sided:

| baseline rel. SE `√B` | `z_α·SE(0)` | `x\*` | **true size** |
|---|---|---|---|
| 0.00 | 0.0822 | 0.0822 | 0.0500 |
| 0.02 | 0.0822 | 0.0834 | 0.0477 |
| 0.05 | 0.0822 | 0.0896 | 0.0366 |
| 0.10 | 0.0822 | 0.1157 | 0.0103 |
| 0.20 | 0.0822 | 0.2707 | ~0 |

The `greater` test loses almost all of its size once the baseline is noisy. The
`less` test is anti-conservative in mirror image: on the negative side
`SE(m)² = σ²_t/r_c² + σ²_c/r_c²·(1+m)²` *decreases* as `m → −1`, so the threshold
moves the other way.

**This is Fieller's problem.** A ratio with a noisy denominator does not have a
well-behaved Wald confidence interval; the exact (Fieller) confidence set can be
unbounded or even the complement of an interval. The fixed point `x\*` is the
one-sided Fieller boundary. Once `√B` is large, the right response is not a
better MDE formula — it is that the delta-method interval itself should not be
trusted, and the experiment is underpowered for a relative estimand regardless.

## 8. Numbers

`α = 0.05` one-sided, `power = 0.8`, `SE(0) = 5%`:

| baseline rel. SE | MDE (a) | MDE (c) | MDE (b) | c/a | b/c |
|---|---|---|---|---|---|
| 0.00 | 0.1243 | 0.1243 | 0.1243 | 1.000 | 1.000 |
| 0.02 | 0.1252 | 0.1264 | 0.1270 | 1.009 | 1.005 |
| 0.05 | 0.1298 | 0.1375 | 0.1420 | 1.059 | 1.033 |
| 0.10 | 0.1453 | 0.1836 | 0.2102 | 1.263 | 1.145 |
| 0.20 | 0.2022 | 0.4531 | 0.6870 | 2.241 | 1.516 |
| 0.30 | 0.2937 | 1.1357 | 2.5227 | 3.867 | 2.221 |

In the regime real experiments live in (`√B` under ~5%, i.e. the baseline
estimated to better than 5% relative precision) all three agree to within about
6%, and (a) versus (c) differ by under 1% below `√B = 2%`. The conventions only
diverge materially in the regime where the delta-method interval is itself
suspect.

## 9. Decision

**(b), the slides' convention, is what ships.** `_effect_dependent_mde_calculation`
solves `m = (z_α + z_β)·SE(m)`, and `_normal_power_calculation` evaluates the
standard error at the effect being tested so the two invert each other exactly.

This note is kept because the analysis above is not superseded by that choice,
and because two of its conclusions are worth carrying forward:

1. **(a) was wrong** and is now fixed. Evaluating the threshold at `SE(0)` models
   a score-type test using the null-restricted variance; this library computes a
   Wald statistic. Section 2 settles that from the code.
2. **(b) is not exact either.** The rejection boundary of the Wald test is the
   fixed point `x\*`, not `z_α·SE(m)`, so (b) overstates the threshold and returns
   a conservative — larger — MDE. Section 5 gives the ordering
   `MDE(a) < MDE(c) < MDE(b)`.

Being conservative is a defensible place to land: (b) never under-sizes an
experiment, and the gap to (c) is 0.5% at 2% baseline noise and 3% at 5% — the
regime real experiments occupy. It grows to 1.5× at 20% baseline noise, but by
then Section 7 applies and the delta-method interval is not trustworthy anyway.

Two caveats that survive whichever convention is chosen:

1. All three treat `A`, `B`, `C` as known. In the regime where they visibly
   disagree, that assumption is also under strain.
2. All three are exact only for the *model* of the test (plug-in variance,
   asymptotic normality of `L̂`), not for the finite-sample test. Separating them
   empirically would need a Monte Carlo against `DeltaMethodAnalysis.get_pvalue`;
   that was started and not finished, so no empirical claim is made here.

### If (c) is ever wanted

It is a small change: solve the fixed point `(1 − z_α²B)x² + 2z_α²Cx − z_α²A = 0`
for `x\*`, then `m = x\* + z_β·SE(m)` — two quadratics with the same shape as the
one already implemented. The power function would become `Φ((m − x\*)/SE(m))`.
