# Physics-Informed Neural Networks for Beam Deflection

## Abstract

This project investigates **Physics-Informed Neural Networks (PINNs)** for computational solid mechanics through beam deformation problems. Instead of learning solely from labeled data, the neural networks are constrained by governing equations and boundary conditions from structural mechanics.

The project progresses from forward prediction of structural deformation to **inverse identification of Young's modulus**, and further explores the **Deep Energy Method (DEM)** as a variational alternative to conventional residual-based PINNs.

Analytical solutions from Euler–Bernoulli and Timoshenko beam theories are used for verification.

---

## Research Problems

This repository focuses on three computational mechanics problems:

### 1. Forward Beam Deflection

Given the geometry, loading condition, and material properties, PINNs are trained to approximate the displacement field of beams governed by Euler–Bernoulli beam theory.

The repository includes:

- a simply supported beam under uniformly distributed loading;
- a cantilever beam under uniformly distributed loading.

For the cantilever benchmark, the governing equation is

\[
EI\frac{d^4w}{dx^4}=q,
\]

with the fixed- and free-end boundary conditions incorporated into the physics-informed loss.

The trained model achieved a **0.0038% tip-deflection error** relative to the analytical solution.

### 2. Inverse Identification of Material Properties

The forward formulation is extended into an inverse problem:

\[
\text{sparse displacement measurements}
\rightarrow
\text{Young's modulus } E.
\]

Young's modulus is treated as a trainable physical parameter and optimized simultaneously with the displacement field using physics constraints and noisy synthetic sensor measurements.

With a true modulus of \(210\,\text{GPa}\), the inverse PINN identified

\[
E_{\mathrm{PINN}} = 209.58\,\text{GPa}.
\]

This demonstrates the potential of PINNs for parameter identification and structural-condition inference from limited observations.

### 3. Deep Energy Method

The project further extends the formulation from one-dimensional beam theory to a **two-dimensional linear-elastic continuum** using the Deep Energy Method.

Instead of minimizing a pointwise PDE residual, DEM minimizes the total potential energy

\[
\Pi = U-W,
\]

where \(U\) is the internal strain energy and \(W\) is the external work.

The displacement field

\[
\mathbf{u}(x,y)=[u_x,u_y]
\]

is represented by a neural network, while the fixed boundary condition is enforced directly through a trial-function ansatz.

The numerical solution is compared with Euler–Bernoulli and Timoshenko beam predictions to evaluate the variational neural formulation.

---

## Methodology

The main computational framework is implemented in **PyTorch** and combines:

- fully connected neural networks with `tanh` activation;
- automatic differentiation for displacement derivatives;
- physics-based residual loss;
- boundary-condition constraints;
- Adam and L-BFGS optimization;
- non-dimensionalization for improved numerical conditioning;
- sparse-data loss for inverse parameter identification;
- total-potential-energy minimization for DEM.

The three notebooks represent the progression

```text
PINNs-for-Beam-Deflection/
│
├── 簡支梁1_Beam_Deflection_PINN.ipynb
│   └── Simply supported beam PINN
│
├── 正反問題.ipynb
│   ├── Cantilever forward problem
│   └── Inverse identification of Young's modulus
│
├── DEM.ipynb
│   └── 2D linear elasticity using the Deep Energy Method
│
├── PINNs Beam Deflection Final Project Presentation.pdf
│
└── README.mdBeam PINN
    ↓
Forward mechanics
    ↓
Inverse parameter identification
    ↓      
2D continuum mechanics
    ↓
Deep Energy Method
