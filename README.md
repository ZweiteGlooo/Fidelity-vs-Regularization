# Fidelity vs Regularization in MOLS Image Deblurring

This project studies image deblurring as a **Multi-Objective Least Squares
(MOLS)** problem. The core mathematical trade-off is **data fidelity vs
regularization**, or equivalently **detail vs smoothness** and **variance vs
stability**. The phrase "bias vs complexity" is used as an interpretation of
that trade-off, not as the formal optimization problem itself. The experiment
follows Boyd and Vandenberghe, *Introduction to Applied Linear Algebra*,
Chapter 15, where several squared-norm objectives are optimized together and
the solution is interpreted through an optimal trade-off curve.

## Mathematical Model

Let `x in R^n` be the unknown clean image after vectorizing a 256x256 grayscale
image, let `y in R^n` be the observed blurred and noisy image, and let `A` be
the discrete Gaussian blur operator. The forward model is

```text
y = A x_true + eta,
```

where `eta` is measurement noise. Recovering `x_true` from `y` is ill posed:
blur removes high-frequency information, and direct inversion amplifies noise.
This is exactly the setting where Tikhonov regularization is used: we stabilize
the inverse problem by adding a penalty that suppresses unstable solutions.

## Ill-Posed Inverse Problem

Image deblurring is an ill-posed inverse problem because small perturbations in
the observed image can produce large changes in the reconstructed image. The
blur operator removes or weakens high-frequency detail, so the inverse map from
`y` back to `x` is numerically unstable. When noise is present, direct inversion
tries to reconstruct information that the blur has already suppressed, which
amplifies noise and creates artificial roughness. Regularization introduces
stability by restricting the reconstruction to smoother, less oscillatory
images, but this stability comes at the cost of possible loss of sharp detail.

The two least-squares objectives are explicitly

```text
J1(x) = ||A x - y||_2^2          data fidelity objective
J2(x) = ||L x||_2^2              regularization objective
```

where `L` is the discrete Laplacian. The MOLS problem is therefore

```text
minimize_x ( J1(x), J2(x) ).
```

More rigorously, the feasible decision variable is the vectorized image

```text
x in X = [0, 1]^n,     n = 256^2,
```

and the objective map is

```text
F : X -> R^2,     F(x) = (J1(x), J2(x)).
```

The true multi-objective problem is not to find one universal minimizer, since
the two objectives conflict. Instead, the goal is to approximate the efficient
set

```text
P = { x in X : there is no z in X with F(z) <= F(x)
      componentwise and F(z) != F(x) }.
```

The image-space set `P` contains the Pareto-optimal reconstructions; its image
`F(P)` is the Pareto front plotted by the program.

The physical meaning of each objective is:

- `J1(x) = ||A x - y||_2^2`: after proposing a clean image `x`, the model blurs
  it using `A` and compares the result with the actual observed blurred/noisy
  image `y`. A small value means the reconstruction is physically consistent
  with what the camera or observation process measured.
- `J2(x) = ||L x||_2^2`: the Laplacian measures rapid local changes in the
  reconstructed image. A small value means the reconstruction is smoother, less
  oscillatory, and less likely to be fitting noise.

This produces the bias-complexity interpretation:

- small `J1` means low data-fitting bias, but it can reproduce noise and create
  a complex image;
- small `J2` means low complexity and strong smoothing, but it can bias the
  image away from the observed data;
- the meaningful solutions are the non-dominated compromises between these two
  objectives.

## Connection to Boyd Chapter 15

Boyd Chapter 15 formulates MOLS objectives as squared residuals

```text
J_i(x) = ||A_i x - b_i||_2^2,
```

and studies the weighted bi-criterion least-squares problem

```text
minimize_x J1(x) + lambda J2(x),    lambda > 0.
```

Our implementation is this Chapter 15 construction with

```text
A1 = A,      b1 = y,
A2 = L,      b2 = 0.
```

Thus each weighted solve is

```text
minimize_x ||A x - y||_2^2 + lambda ||L x||_2^2.
```

This is also standard Tikhonov regularized fitting. The normal equations solved
by conjugate gradients are

```text
(A^T A + lambda L^T L) x = A^T y.
```

The parameter `lambda` is not the final object of study. It is a scalarization
weight used to sample the MOLS trade-off curve. Increasing `lambda` prioritizes
low complexity; decreasing `lambda` prioritizes data fit.

## Normal Equations Derivation

For a fixed scalarization weight `lambda > 0`, define

```text
Phi_lambda(x) = ||A x - y||_2^2 + lambda ||L x||_2^2.
```

Expand each squared norm:

```text
Phi_lambda(x)
  = (A x - y)^T(A x - y) + lambda (L x)^T(L x)
  = x^T A^T A x - 2 y^T A x + y^T y
    + lambda x^T L^T L x.
```

Taking the gradient with respect to `x` gives

```text
grad Phi_lambda(x)
  = 2 A^T(A x - y) + 2 lambda L^T L x.
```

At the minimizer this gradient is zero:

```text
A^T(A x - y) + lambda L^T L x = 0.
```

Rearranging yields the regularized normal equations:

```text
(A^T A + lambda L^T L) x = A^T y.
```

This derivation is important because it shows exactly how the MOLS weighted
problem becomes a linear system. The code solves this system with conjugate
gradients, using only matrix-vector products instead of explicitly forming the
large dense matrix.

## Matrix And Operator Construction

The image is resized to 256x256, so `n = 256 * 256 = 65536`. A full dense
matrix `A in R^(n x n)` would contain more than four billion entries, so the
implementation represents `A` and `L` as operators:

- `A x` applies a Gaussian blur to the image obtained from reshaping `x`.
- `L x` applies the five-point discrete Laplacian stencil
  `-4 center + north + south + east + west`.
- `A^T` is applied with the same Gaussian operator because the chosen reflective
  Gaussian blur is symmetric.
- `L^T` is applied with the same Laplacian operator because the stencil is
  symmetric.

Therefore the conjugate-gradient matrix-vector product is

```text
z -> A(A z) + lambda L(L z),
```

which corresponds to `(A^T A + lambda L^T L)z`. This construction keeps the
mathematics faithful to the matrix formulation while making the computation
feasible for 256x256 images.

## Why Laplacian Regularization?

The regularization matrix could have been chosen as the identity, which would
penalize `||x||_2^2`, the overall magnitude or energy of the reconstructed
image. That is useful in some inverse problems, but it is not the most natural
choice for deblurring because pixel intensity magnitude is not the main problem.
The main instability is high-frequency oscillation: noise and artificial sharp
changes introduced while trying to invert the blur. The Laplacian penalty
`||Lx||_2^2` directly measures local curvature and roughness, so it discourages
oscillatory reconstructions while still allowing broad smooth structures and
meaningful edges to remain. For this reason, Laplacian regularization better
matches the physical goal of image restoration: suppress noise without merely
shrinking the whole image toward zero.

## Conditioning And Regularization

Image deblurring is ill-conditioned because blur attenuates high-frequency
components. In matrix terms, the singular values of `A` associated with fine
detail are small, so an unregularized least-squares solution can divide by small
values and greatly amplify noise. This explains why direct inversion or plain
least squares tends to produce unstable reconstructions.

The singular value decomposition makes this precise. If

```text
A = U Sigma V^T,
```

then the unregularized least-squares/pseudoinverse solution can be written as

```text
x_ls = A^+ y
     = sum_i (u_i^T y / sigma_i) v_i.
```

With noisy data `y = A x_true + eta`, the noise contribution is

```text
sum_i (u_i^T eta / sigma_i) v_i.
```

When `sigma_i` is small, even a small noise coefficient `u_i^T eta` is magnified
by `1 / sigma_i`. For blur operators, the small-singular-value directions are
typically high-frequency image components, so the unstable modes appear visually
as rough oscillations or amplified noise.

Tikhonov regularization improves numerical behavior by solving with

```text
A^T A + lambda L^T L
```

instead of only `A^T A`. The added term penalizes rough high-frequency
solutions, which are precisely the components most likely to be noise-amplified.
There is still a trade-off:

- very small `lambda`: weak stabilization, low bias, high noise sensitivity;
- very large `lambda`: strong stabilization, high bias, oversmoothed image;
- intermediate `lambda`: useful compromise between data fidelity and stability.

The MOLS view makes this conditioning issue visible as an objective-space
trade-off, rather than hiding it inside a single parameter choice.

In the special case `L = I`, the Tikhonov solution has the SVD filter form

```text
x_lambda = sum_i [ sigma_i / (sigma_i^2 + lambda) ] (u_i^T y) v_i.
```

The factor `sigma_i / (sigma_i^2 + lambda)` replaces the unstable inverse
factor `1 / sigma_i`. Very small singular directions are damped instead of
amplified. In this project `L` is a Laplacian rather than the identity, but the
same principle applies: the regularizer suppresses unstable, rough components
while preserving enough structure to fit the observation.

The script also saves `outputs/svd_regularization_explanation.png`, a simple
SVD interpretation figure. It compares the unregularized inverse factor
`1 / sigma`, which explodes for small singular values, with the Tikhonov filter
`sigma / (sigma^2 + lambda)`, which damps those unstable directions.

## Epsilon-Constraint MOLS

The default method is the epsilon-constraint version of the same MOLS problem:

```text
minimize_x J1(x)
subject to J2(x) <= epsilon.
```

The code first computes weighted Tikhonov anchor points, then refines candidate
solutions for several `epsilon` bounds on `J2`. This makes the output explicitly
multi-objective: the reported candidates are points in objective space
`(J1(x), J2(x))`, not just independent reconstructions from unrelated lambdas.

This also matches classical multi-objective optimization practice: the
weighted-sum method and the epsilon-constraint method are two scalarization
strategies for exploring non-dominated trade-offs. Here they are applied to
least-squares objectives, so the method remains directly connected to Boyd's
MOLS framework.

## Pareto Front Interpretation

A candidate `x_a` dominates `x_b` if

```text
J1(x_a) <= J1(x_b),   J2(x_a) <= J2(x_b),
```

with at least one strict inequality. A Pareto-optimal reconstruction is not
dominated by any other tested reconstruction.

This matters because a single number cannot honestly summarize the reconstruction
task. A lower `J1` is not automatically better if it is achieved by increasing
`J2` so much that the image becomes noisy or overly complex. Likewise, a lower
`J2` is not automatically better if the image becomes too smooth to explain the
data. Pareto optimality keeps only solutions where improving one objective
requires sacrificing the other.

The Pareto front in `outputs/pareto.png` is the empirical optimal trade-off
curve. Moving along it shows the cost of reducing complexity in terms of worse
data fidelity, or the cost of improving data fidelity in terms of greater
roughness. A steep region means a large complexity improvement can be bought
with little fidelity loss; a flat region means fidelity improves substantially
with little extra roughness. This is the strongest visual evidence for the
bias-complexity trade-off.

The highlighted image is selected from the Pareto set. By default the script
uses the available original image as a reference and highlights the Pareto image
with the smallest reference MSE; alternatively, `--best-selection compromise`
chooses the point closest to the ideal objective-space corner after normalizing
`J1` and `J2`.

## Quantitative Image Metrics

The experiment reports both optimization objectives and image-quality metrics:

- `residual_norm = ||A x - y||_2`: physical consistency with the observed
  blurred/noisy image.
- `fidelity_error = J1(x) = ||A x - y||_2^2`: squared residual norm used as the
  first MOLS objective.
- `roughness_norm = ||L x||_2`: measured smoothness/complexity of the
  reconstruction.
- `noise_penalty = J2(x) = ||L x||_2^2`: squared roughness norm used as the
  second MOLS objective.
- `reference_error_norm = ||x - x_true||_2`: direct reconstruction error against
  the known clean image.
- `reference_mse = mean((x - x_true)^2)`: reconstruction error against the
  known original image used to create the synthetic blurred/noisy observation.
- `reference_psnr = -10 log10(reference_mse)`: peak signal-to-noise ratio in dB;
  larger values indicate better reconstruction quality.
- `reference_ssim`: structural similarity index against the known original
  image; values closer to 1 indicate stronger structural similarity.
- `compromise_score`: distance from the normalized ideal point in objective
  space, used when selecting `--best-selection compromise`.

These metrics serve different purposes. `J1` and `J2` define the MOLS problem
and the Pareto front. The residual norm explains whether a candidate is
consistent with the measured blurred image. The reference error, MSE, PSNR, and
SSIM are external validation metrics available because this is a controlled
experiment with a known original image. A solution can be Pareto-optimal in
`(J1, J2)` without being the best by PSNR or SSIM, which is why the README and
output separate optimization trade-off from reference-based evaluation.

The most important quantitative comparison is therefore two-layered:

```text
Optimization layer:  compare candidates by (J1, J2) and Pareto dominance.
Validation layer:    compare candidates by error norm, MSE, PSNR, and SSIM.
```

The first layer answers the mathematical MOLS question. The second layer checks
how well the selected compromise recovers the known original image.

## Experimental Evidence

The implementation is designed to make the experiment auditable rather than
only visual. A complete run produces:

- multiple candidate reconstructions over a logarithmic range of regularization
  weights;
- an epsilon-constraint Pareto set extracted from those Tikhonov candidates;
- side-by-side reconstruction grids in `outputs/summary.png` and
  `outputs/pareto_summary.png`;
- the objective-space Pareto front in `outputs/pareto.png`;
- quantitative metrics along the Pareto front in `outputs/metric_tradeoffs.png`;
- a compact report table in `outputs/experiment_summary.csv`;
- full numeric records in `outputs/metrics.csv` and `outputs/pareto_front.csv`;
- an SVD conditioning explanation figure in
  `outputs/svd_regularization_explanation.png`.

For the default visual run on `5.1.14.tiff`, the selected Pareto reconstruction
is chosen from the non-dominated set, not from all candidates blindly. The CSV
files report its residual norm, roughness norm, reconstruction error norm, MSE,
PSNR, and SSIM. This prevents the evaluation from relying only on pictures:
the images show qualitative behavior, while the metrics measure reconstruction
accuracy and physical consistency with the observation.

## Sources Used

- Boyd and Vandenberghe, *Introduction to Applied Linear Algebra*, Chapter 15:
  MOLS, weighted squared-norm objectives, and optimal trade-off curves.
- Tikhonov and Arsenin, *Solutions of Ill-Posed Problems*: inverse problems and
  the need for regularization to stabilize noisy inverse solutions.
- Deb, *Multi-Objective Optimization using Evolutionary Algorithms*: Pareto
  dominance and non-dominated solution interpretation.
- Barligea, Hochstaffl, and Schreier, "A Generalized Variable Projection
  Algorithm for Least Squares Problems in Atmospheric Remote Sensing": least
  squares fitting context and scientific inverse-problem motivation.

## Running

The project requirement is a 256x256 working image. The script now uses that
size by default:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

For a quick verification run:

```powershell
.\.venv\Scripts\python.exe src\main.py --quality-preset fast
```

Useful outputs:

- `outputs/metrics.csv`: all candidates and objective values.
- `outputs/pareto_front.csv`: only non-dominated candidates.
- `outputs/experiment_summary.csv`: representative Pareto candidates with
  metrics and interpretation.
- `outputs/pareto.png`: objective-space Pareto front.
- `outputs/metric_tradeoffs.png`: PSNR, SSIM, and residual norm along the
  Pareto front.
- `outputs/svd_regularization_explanation.png`: SVD-based conditioning figure.
- `outputs/best_pareto_compromise.png`: selected Pareto reconstruction.
