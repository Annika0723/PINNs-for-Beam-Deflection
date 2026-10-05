# PINNs for Beam Deflection

A computational mechanics project exploring **physics-informed neural networks (PINNs)** and the **Deep Energy Method (DEM)** for beam deformation, forward boundary-value problems, and inverse identification of material properties.

Rather than treating neural networks purely as data-fitting models, this project incorporates governing equations, boundary conditions, and variational principles from structural mechanics directly into the learning process. The repository progresses from a one-dimensional Euler–Bernoulli beam formulation to inverse estimation of Young's modulus and finally to a two-dimensional continuum elasticity problem solved through energy minimization.

---

## Overview

Conventional neural networks learn mappings primarily from labeled data. In contrast, physics-informed neural networks constrain the approximation using known physical laws.

For structural mechanics, this means that a neural network can represent a displacement field while its derivatives—computed through automatic differentiation—are required to satisfy equilibrium equations and boundary conditions.

This repository investigates three related problems:

1. **One-dimensional beam deflection using a strong-form PINN**
2. **Forward and inverse Euler–Bernoulli beam problems**
3. **Two-dimensional linear elasticity using the Deep Energy Method**

The project was developed as an exploration of how neural-network function approximators can be coupled with classical structural mechanics, and how the same framework can be extended from solving displacement fields to identifying unknown physical parameters.

---

# 1. Physical Background

## 1.1 Euler–Bernoulli Beam Theory

For a slender elastic beam subjected to a transverse distributed load \(q(x)\), the Euler–Bernoulli governing equation is

\[
EI\frac{d^4w}{dx^4}=q(x),
\]

where

- \(w(x)\) is the transverse displacement,
- \(E\) is Young's modulus,
- \(I\) is the second moment of area,
- \(EI\) is the flexural rigidity.

A physics-informed neural network approximates the unknown displacement as

\[
w_\theta(x)\approx w(x),
\]

where \(\theta\) denotes the trainable neural-network parameters.

Because the network is differentiable, PyTorch automatic differentiation can directly evaluate

\[
\frac{dw_\theta}{dx},\quad
\frac{d^2w_\theta}{dx^2},\quad
\frac{d^3w_\theta}{dx^3},\quad
\frac{d^4w_\theta}{dx^4}.
\]

The governing equation can therefore be incorporated into the loss function without generating conventional finite-element shape functions or assembling a global stiffness matrix.

---

# 2. Repository Structure

| File | Purpose |
|---|---|
| `簡支梁1_Beam_Deflection_PINN.ipynb` | Introductory 1D PINN for a simply supported beam under uniformly distributed loading |
| `正反問題.ipynb` | Dimensionless fourth-order PINN for a cantilever beam, including forward prediction and inverse identification of Young's modulus |
| `DEM.ipynb` | Two-dimensional plane-stress elasticity solved using the Deep Energy Method |
| `PINNs Beam Deflection Final Project Presentation.pdf` | Final project presentation |
| `README.md` | Project documentation |

The three notebooks represent increasing levels of physical complexity:

```text
1D beam PINN
      ↓
Fourth-order forward PINN
      ↓
Inverse material identification
      ↓
2D continuum elasticity
      ↓
Deep Energy Method
```

---

# 3. Study I — Simply Supported Beam PINN

**Notebook:** `簡支梁1_Beam_Deflection_PINN.ipynb`

The first notebook provides a pedagogical implementation of a physics-informed neural network for a simply supported beam subjected to a uniformly distributed load.

## Problem definition

The numerical parameters used in the notebook are

\[
L=10.0,
\qquad
w=4.0,
\qquad
EI=625\times10^3.
\]

Instead of directly solving the complete fourth-order beam equation, the bending-moment distribution is prescribed analytically as

\[
M(x)=\frac{wL}{2}x-\frac{w}{2}x^2.
\]

The PINN then satisfies the curvature relation

\[
\frac{d^2u}{dx^2}+\frac{M(x)}{EI}=0.
\]

This distinction is important: the notebook is a physics-informed solution of the beam curvature equation **given the known moment field**, rather than an independent solution of the full fourth-order equilibrium equation.

---

## Neural network

The displacement field is approximated using a fully connected network with

- input: beam coordinate \(x\),
- output: displacement \(u(x)\),
- 4 hidden layers,
- 150 neurons per hidden layer,
- SiLU activation in the first hidden layer,
- Tanh activation in subsequent hidden layers,
- Xavier initialization.

Automatic differentiation is used to calculate \(du/dx\) and \(d^2u/dx^2\).

---

## Physics-informed loss

The total loss combines the governing-equation residual with displacement and slope constraints:

\[
\mathcal{L}
=
\mathcal{L}_{PDE}
+
\lambda_{BC}\mathcal{L}_{BC}.
\]

The physics loss is

\[
\mathcal{L}_{PDE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
u_\theta''(x_i)
+
\frac{M(x_i)}{EI}
\right]^2.
\]

The implementation additionally constrains beam displacement and slope at selected locations.

### Training configuration

- Optimizer: Adam
- Maximum epochs: 40,000
- Collocation points per epoch: 2,000
- Collocation points are randomly resampled during training
- Initial learning rate: \(10^{-3}\)
- Learning-rate scheduler: `ReduceLROnPlateau`

---

## Result

The network prediction is compared with the analytical beam solution

\[
u(x)
=
\frac{w}{24EI}
\left(
x^4-2Lx^3+L^3x
\right).
\]

The stored notebook output reports

| Metric | Result |
|---|---:|
| Maximum absolute error | \(8.7268\times10^{-5}\) |
| Mean absolute error | \(2.8626\times10^{-5}\) |

This notebook primarily serves as an introduction to constructing physics-based residuals with automatic differentiation.

---

# 4. Study II — Fourth-Order PINN for a Cantilever Beam

**Notebook:** `正反問題.ipynb`

The second notebook implements the Euler–Bernoulli equation more directly by differentiating the neural-network displacement field up to the **fourth derivative**.

The beam properties are

\[
L=1.0~\mathrm{m},
\]

\[
E=210~\mathrm{GPa},
\]

\[
I=8.33\times10^{-6}~\mathrm{m}^4,
\]

and

\[
q=10,000~\mathrm{N/m}.
\]

For a cantilever beam under uniformly distributed loading,

\[
EI\frac{d^4w}{dx^4}=q.
\]

The analytical solution used for verification is

\[
w(x)=
\frac{q}{24EI}
\left(
x^4-4Lx^3+6L^2x^2
\right).
\]

---

## 4.1 Non-dimensionalization

To improve numerical conditioning, the formulation uses dimensionless coordinates and displacement.

Define

\[
\bar{x}=\frac{x}{L}
\]

and use the displacement scale

\[
w_{ref}
=
\frac{qL^4}{EI}.
\]

The governing equation becomes

\[
\frac{d^4\bar{w}}{d\bar{x}^4}=1.
\]

This form avoids directly mixing very different numerical scales associated with \(E\), \(I\), \(q\), and displacement.

---

## 4.2 Boundary conditions

The cantilever boundary conditions are imposed through the loss function.

At the fixed end,

\[
\bar{w}(0)=0,
\]

\[
\frac{d\bar{w}}{d\bar{x}}(0)=0.
\]

At the free end, zero bending moment and zero shear force require

\[
\frac{d^2\bar{w}}{d\bar{x}^2}(1)=0,
\]

\[
\frac{d^3\bar{w}}{d\bar{x}^3}(1)=0.
\]

The forward loss is therefore

\[
\mathcal{L}_{forward}
=
\mathcal{L}_{PDE}
+
\lambda_{BC}\mathcal{L}_{BC}.
\]

In the implementation,

\[
\lambda_{BC}=100.
\]

---

## 4.3 Neural-network architecture

The forward solver uses

```text
Input x
  ↓
64 neurons + Tanh
  ↓
64 neurons + Tanh
  ↓
64 neurons + Tanh
  ↓
64 neurons + Tanh
  ↓
Output w(x)
```

Linear-layer weights are initialized using Xavier initialization.

PyTorch automatic differentiation recursively evaluates derivatives up to

\[
\frac{d^4w}{dx^4}.
\]

---

## 4.4 Optimization

A two-stage optimization strategy is used.

### Stage 1 — Adam

- 300 collocation points
- 2,000 Adam epochs
- initial learning rate: \(10^{-3}\)
- cosine-annealing learning-rate schedule

### Stage 2 — L-BFGS

The Adam solution is subsequently refined using the quasi-Newton L-BFGS optimizer.

This combination is commonly useful for physics-informed optimization: Adam first provides robust global progress, while L-BFGS can significantly improve convergence near a low-loss solution.

---

## Forward result

The trained PINN is evaluated against the analytical Euler–Bernoulli solution at 500 points along the beam.

The stored result gives

\[
\boxed{
\text{tip deflection error}=0.0038\%
}
\]

showing that the neural approximation closely reproduces the analytical cantilever solution for this benchmark problem.

---

# 5. Study III — Inverse Identification of Young's Modulus

One of the central extensions of this project is the transition from a **forward problem** to an **inverse problem**.

Instead of assuming that all material parameters are known, Young's modulus \(E\) is treated as an unknown trainable physical parameter.

The objective becomes

\[
\text{measured displacement}
+
\text{governing physics}
\quad
\Longrightarrow
\quad
E.
\]

This is conceptually related to parameter identification problems encountered in structural health monitoring, experimental mechanics, and inverse computational mechanics.

---

## 5.1 Positive parameterization

Young's modulus must remain positive during optimization.

Instead of directly optimizing \(E\), the model introduces

\[
E=\exp(\eta),
\]

where \(\eta\) is an unconstrained trainable parameter.

This guarantees

\[
E>0
\]

throughout training.

---

## 5.2 Synthetic sensor measurements

Ten displacement sensors are distributed from

\[
x=0.1L
\]

to

\[
x=L.
\]

Synthetic measurements are generated from the analytical cantilever solution.

Gaussian noise with an amplitude equal to approximately 1% of the maximum displacement is subsequently added to represent measurement uncertainty.

Therefore, the inverse solver does not receive an exact analytical curve; it must infer the material parameter from sparse and noisy displacement observations.

---

## 5.3 Inverse loss function

The inverse problem combines three sources of information:

\[
\mathcal{L}_{inverse}
=
\mathcal{L}_{PDE}
+
\lambda_{BC}\mathcal{L}_{BC}
+
\lambda_{data}\mathcal{L}_{data}.
\]

The terms represent

- **physics consistency** with the beam equation,
- **boundary-condition consistency**,
- **agreement with sparse displacement measurements**.

The implementation uses

\[
\lambda_{BC}=100,
\]

\[
\lambda_{data}=200.
\]

Young's modulus is initialized at only

\[
E_{initial}=50~\mathrm{GPa},
\]

while the synthetic ground truth is

\[
E_{true}=210~\mathrm{GPa}.
\]

Training again combines Adam and L-BFGS.

---

## Inverse result

The final identified modulus is

\[
\boxed{
E_{PINN}=209.58~\mathrm{GPa}
}
\]

compared with

\[
E_{true}=210.00~\mathrm{GPa}.
\]

The relative parameter error is therefore approximately

\[
\frac{|209.58-210|}{210}\times100
\approx0.20\%.
\]

This example demonstrates an important capability of physics-informed learning: the neural network can simultaneously reconstruct the displacement field and infer an unknown constitutive parameter from sparse observations.

---

# 6. Study IV — Two-Dimensional Deep Energy Method

**Notebook:** `DEM.ipynb`

The final part of the project moves beyond one-dimensional beam theory and represents the structure as a **two-dimensional elastic continuum**.

Instead of minimizing a strong-form PDE residual, this model uses the **Deep Energy Method (DEM)**.

DEM is based on the variational principle of minimum potential energy.

For an elastic structure,

\[
\Pi
=
\mathcal{U}
-
\mathcal{W},
\]

where

- \(\Pi\) is the total potential energy,
- \(\mathcal{U}\) is internal strain energy,
- \(\mathcal{W}\) is external work.

Mechanical equilibrium corresponds to a stationary/minimum state of the potential-energy functional.

The neural network therefore searches for a displacement field

\[
\mathbf{u}_\theta(x,y)
=
\begin{bmatrix}
u_x(x,y)\\
u_y(x,y)
\end{bmatrix}
\]

that minimizes \(\Pi\).

---

# 7. Two-Dimensional Elasticity Formulation

## Geometry and material

The DEM example uses

| Parameter | Value |
|---|---:|
| Beam length | \(10~\mathrm{m}\) |
| Beam height | \(1~\mathrm{m}\) |
| Thickness | \(1~\mathrm{m}\) |
| Young's modulus | \(25~\mathrm{GPa}\) |
| Poisson's ratio | 0.30 |
| Distributed load | \(-80,000~\mathrm{N/m}\) |
| Grid | \(200\times20\) points |

A plane-stress linear-elastic constitutive model is adopted.

---

## 7.1 Strain field

Automatic differentiation is used to obtain displacement gradients.

The small-strain tensor components are

\[
\varepsilon_{xx}
=
\frac{\partial u_x}{\partial x},
\]

\[
\varepsilon_{yy}
=
\frac{\partial u_y}{\partial y},
\]

and

\[
\varepsilon_{xy}
=
\frac{1}{2}
\left(
\frac{\partial u_x}{\partial y}
+
\frac{\partial u_y}{\partial x}
\right).
\]

---

## 7.2 Elastic strain-energy density

For the plane-stress formulation, the implementation defines

\[
\lambda_{ps}
=
\frac{E\nu}{1-\nu^2}
\]

and

\[
\mu
=
\frac{E}{2(1+\nu)}.
\]

The strain-energy density is evaluated as

\[
\psi
=
\frac{1}{2}
\lambda_{ps}
\left(
\varepsilon_{xx}+\varepsilon_{yy}
\right)^2
+
\mu
\left(
\varepsilon_{xx}^2+
\varepsilon_{yy}^2+
2\varepsilon_{xy}^2
\right).
\]

The total internal energy is approximated numerically as

\[
\mathcal{U}
=
\int_\Omega
\psi\,d\Omega.
\]

---

# 8. Deep Energy Neural Network

The displacement field is represented using a multilayer perceptron

```text
Inputs: (x, y)
      ↓
64 neurons + Tanh
      ↓
64 neurons + Tanh
      ↓
64 neurons + Tanh
      ↓
Outputs: (ux, uy)
```

Xavier initialization is used for the linear layers.

Unlike the 1D PINN, the network predicts two displacement components simultaneously.

---

## Hard enforcement of the fixed boundary

The cantilever's essential boundary condition is embedded directly into the trial displacement field.

The raw network outputs are multiplied by \(x\):

\[
u_x(x,y)=x\,N_x(x,y),
\]

\[
u_y(x,y)=x\,N_y(x,y).
\]

Therefore,

\[
u_x(0,y)=0,
\qquad
u_y(0,y)=0
\]

for any neural-network parameters.

This is an example of **hard boundary-condition enforcement**, because the constraint is satisfied by construction rather than through an additional penalty term.

---

# 9. Variational Loss

The internal strain energy is numerically approximated over the two-dimensional domain as

\[
\mathcal{U}
\approx
\operatorname{mean}(\psi)
\times
L H b.
\]

The distributed traction acting on the beam surface contributes external work,

\[
\mathcal{W}
=
\int_{\Gamma_t}
\mathbf{t}\cdot\mathbf{u}\,d\Gamma.
\]

The implementation approximates this contribution using the predicted vertical displacement along the loaded upper surface.

The training objective is then

\[
\boxed{
\mathcal{L}_{DEM}
=
\Pi
=
\mathcal{U}-\mathcal{W}
}
\]

rather than a pointwise equilibrium-equation residual.

Optimization is performed using L-BFGS.

---

# 10. DEM Verification

The predicted displacement along the neutral axis is compared against both Euler–Bernoulli and Timoshenko beam theory.

The stored notebook results are

| Method | Maximum / tip deflection |
|---|---:|
| Euler–Bernoulli beam | \(-0.048000~\mathrm{m}\) |
| Timoshenko beam | \(-0.048499~\mathrm{m}\) |
| Deep Energy Method | \(-0.042279~\mathrm{m}\) |

Corresponding errors are

| Reference solution | DEM relative error |
|---|---:|
| Euler–Bernoulli | 11.92% |
| Timoshenko | 12.83% |

These results show that the current DEM implementation captures the qualitative bending deformation but still exhibits a noticeable quantitative discrepancy.

Accordingly, this notebook should be interpreted as a **proof-of-concept implementation of variational neural mechanics**, rather than a fully converged replacement for conventional finite-element analysis.

Potential contributors to the remaining error include numerical integration accuracy, optimization convergence, spatial sampling density, neural-network approximation error, and differences between the 2D continuum formulation and one-dimensional beam-theory reference models.

---

# 11. Strong-Form PINN vs. Deep Energy Method

The repository therefore contains two distinct ways of incorporating mechanics into neural networks.

| | Strong-form PINN | Deep Energy Method |
|---|---|---|
| Physical constraint | Governing differential equation | Energy functional |
| Training objective | PDE residual | Total potential energy |
| Required derivatives | Up to fourth order for Euler–Bernoulli beam | First displacement derivatives for linear elasticity |
| Boundary conditions | Typically penalty or hard constraints | Essential BCs imposed explicitly; natural BCs arise variationally |
| Primary example | 1D beam | 2D continuum |
| Main advantage | Direct connection to governing equations | Lower derivative order and natural mechanics formulation |
| Main challenge | Optimization of high-order derivatives | Accurate numerical integration and energy minimization |

This comparison is one of the central motivations of the project.

---

# 12. Numerical Results Summary

| Problem | Main result |
|---|---|
| Simply supported 1D PINN | Maximum absolute displacement error \(8.73\times10^{-5}\) |
| Simply supported 1D PINN | Mean absolute displacement error \(2.86\times10^{-5}\) |
| Cantilever forward PINN | Tip-deflection relative error \(0.0038\%\) |
| Inverse PINN | \(E=209.58~\mathrm{GPa}\) |
| Inverse ground truth | \(E=210.00~\mathrm{GPa}\) |
| Inverse modulus error | approximately \(0.20\%\) |
| 2D DEM | Tip deflection \(-0.042279~\mathrm{m}\) |
| 2D DEM vs. Euler–Bernoulli | 11.92% error |
| 2D DEM vs. Timoshenko | 12.83% error |

---

# 13. How to Run

The notebooks are written in Python using PyTorch and can be executed in Google Colab or a local Jupyter environment.

## Core dependencies

```bash
pip install torch numpy matplotlib jupyter
```

CUDA acceleration is optional. Each notebook automatically uses a GPU when one is available.

A recommended order is

```text
1. 簡支梁1_Beam_Deflection_PINN.ipynb
2. 正反問題.ipynb
3. DEM.ipynb
```

This order follows the conceptual progression from a basic 1D PINN to inverse identification and finally two-dimensional variational mechanics.

---

# 14. Reproducibility

Random seeds are explicitly specified in the notebooks for NumPy and PyTorch.

Examples include

```python
torch.manual_seed(2026)
np.random.seed(2026)
```

and

```python
torch.manual_seed(42)
np.random.seed(42)
```

Exact numerical results may nevertheless vary slightly across PyTorch versions, CPU/GPU hardware, CUDA implementations, and optimizer behavior.

---

# 15. Current Limitations

This repository is an educational and research-oriented prototype. Several limitations should be considered when interpreting the results.

### 1. Synthetic inverse data

The inverse Young's-modulus experiment uses analytically generated displacement measurements with artificial Gaussian noise.

Therefore, the reported identification accuracy does not yet establish performance under real experimental noise, imperfect boundary conditions, model discrepancy, or sensor bias.

### 2. Analytical information in the introductory beam problem

The simply supported beam notebook supplies the analytical bending-moment distribution and several slope constraints.

It is therefore intended primarily as a pedagogical PINN implementation and should not be interpreted as a completely data-free solution of the full fourth-order beam boundary-value problem.

### 3. DEM numerical integration

The current DEM approximates domain and boundary integrals using structured sampling and mean-value integration.

More accurate quadrature, adaptive sampling, or higher-order integration may improve accuracy.

### 4. Optimization sensitivity

Physics-informed optimization is generally non-convex. Results can depend on

- initialization,
- relative loss weighting,
- network architecture,
- collocation-point distribution,
- optimizer parameters,
- scaling and non-dimensionalization.

### 5. Beam-theory versus continuum comparison

The DEM solves a two-dimensional elasticity problem, whereas Euler–Bernoulli and Timoshenko theories are one-dimensional reduced-order models.

Their predictions are therefore useful verification references but are not mathematically identical formulations.

---

# 16. Possible Extensions

Several research directions naturally follow from the current implementation:

- identification of multiple unknown material parameters;
- parameter estimation from experimental displacement or strain measurements;
- uncertainty quantification under noisy sensor data;
- spatially varying elastic properties;
- nonlinear material behavior;
- geometrically nonlinear beam deformation;
- damage and stiffness degradation identification;
- comparison with finite-element solutions;
- adaptive collocation or quadrature;
- weak-form and variational PINNs;
- plate and shell mechanics;
- fracture and phase-field formulations;
- physics-informed surrogate modeling for repeated structural analyses.

These extensions would move the framework from analytical benchmark problems toward experimental and engineering-scale inverse mechanics.

---

# 17. Academic Context

This project is based on two related classes of physics-informed computational methods.

### Physics-Informed Neural Networks

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).  
**Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.**  
*Journal of Computational Physics, 378*, 686–707.  
DOI: `10.1016/j.jcp.2018.10.045`

### Deep Energy Method

Samaniego, E., Anitescu, C., Goswami, S., Nguyen-Thanh, V. M., Guo, H., Hamdia, K., Zhuang, X., & Rabczuk, T. (2020).  
**An energy approach to the solution of partial differential equations in computational mechanics via machine learning: Concepts, implementation and applications.**  
*Computer Methods in Applied Mechanics and Engineering, 362*, 112790.  
DOI: `10.1016/j.cma.2019.112790`

These works motivate the use of neural networks not simply as regression models, but as continuous function approximators constrained by governing physical principles.

---

# 18. Project Perspective

The main purpose of this project is not to demonstrate that neural networks outperform classical analytical or finite-element methods for simple beam problems.

For these benchmark cases, conventional mechanics methods remain substantially more direct.

Instead, the beam problems provide controlled environments for investigating a more general question:

> **Can neural networks represent physical fields while simultaneously satisfying governing mechanics, incorporating sparse observations, and identifying unknown physical parameters?**

The forward PINN demonstrates physics-constrained field approximation.

The inverse PINN demonstrates parameter discovery from sparse noisy observations.

The two-dimensional Deep Energy Method extends the same idea from beam theory toward continuum mechanics using a variational formulation.

Together, these examples form a foundation for future work in physics-informed computational mechanics and inverse structural analysis.
