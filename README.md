# Physics-Informed Neural Networks for Beam Deflection

## Abstract

This project explores **Physics-Informed Neural Networks (PINNs)** for computational solid mechanics through beam-deformation problems. Instead of relying only on labeled data, the neural networks are constrained by governing equations, boundary conditions, and energy principles from structural mechanics.

The project progresses from **forward prediction of beam deflection**, to **inverse identification of Young's modulus**, and finally to a **two-dimensional Deep Energy Method (DEM)** formulation. Analytical solutions from Euler–Bernoulli and Timoshenko beam theories are used for verification.

---

## Research Problems

### 1. Forward beam deflection

PINNs are used to approximate structural displacement fields under prescribed geometry, loading, and material properties.

Two beam problems are considered:

- simply supported beam under uniformly distributed load;
- cantilever beam under uniformly distributed load.

For the cantilever beam, the Euler–Bernoulli governing equation is

\[
EI\frac{d^4w}{dx^4}=q,
\]

with fixed-end and free-end boundary conditions incorporated into the physics-informed formulation.

The forward cantilever model achieved a **0.0038% tip-deflection error** relative to the analytical solution.

### 2. Inverse identification of Young's modulus

The forward problem is extended to an inverse problem:

\[
\text{sparse displacement measurements}
\rightarrow
\text{Young's modulus } E.
\]

Young's modulus is treated as a trainable physical parameter and optimized jointly with the displacement field using governing physics and noisy synthetic measurements.

The model identified

\[
E_{\mathrm{PINN}} = 209.58~\mathrm{GPa},
\]

compared with the true value

\[
E_{\mathrm{true}} = 210.00~\mathrm{GPa}.
\]

This demonstrates the potential of physics-informed learning for parameter identification and structural-condition inference from limited observations.

### 3. Deep Energy Method

The project further extends from one-dimensional beam theory to a **two-dimensional linear-elastic continuum** using the Deep Energy Method.

Instead of minimizing a pointwise PDE residual, DEM minimizes the total potential energy

\[
\Pi = U-W,
\]

where \(U\) is the internal strain energy and \(W\) is the external work.

A neural network approximates the two-dimensional displacement field

\[
\mathbf{u}(x,y)=[u_x,u_y].
\]

The fixed boundary condition is enforced through a trial-function ansatz, and the DEM prediction is compared with Euler–Bernoulli and Timoshenko beam solutions.

The final presentation reports a DEM tip deflection of **−0.046111 m**, corresponding to errors of **3.94%** relative to Euler–Bernoulli theory and **4.92%** relative to Timoshenko theory.

---

## Methodology

The project is implemented in **PyTorch** and combines:

- fully connected neural networks;
- `tanh` activation functions;
- automatic differentiation;
- physics-based residual losses;
- boundary-condition constraints;
- Adam and L-BFGS optimization;
- non-dimensionalization;
- sparse-data loss for inverse parameter identification;
- total-potential-energy minimization for DEM.

The overall progression is

```text
Beam PINN
    ↓
Forward mechanics
    ↓
Inverse parameter identification
    ↓
2D continuum mechanics
    ↓
Deep Energy Method
```

---

## Repository Structure

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
└── README.md
```

---

## How to Use

The notebooks can be executed in **Google Colab** or a local Jupyter environment.

Install the core dependencies:

```bash
pip install torch numpy matplotlib jupyter
```

Recommended execution order:

```text
1. 簡支梁1_Beam_Deflection_PINN.ipynb
2. 正反問題.ipynb
3. DEM.ipynb
```

CUDA acceleration is optional; the notebooks automatically use a GPU when available.

---

## Main Contributions

This project demonstrates a progressive application of physics-informed learning to structural mechanics:

1. **Forward mechanics** — reproduced beam deformation using governing equations rather than purely data-driven regression.
2. **Inverse mechanics** — extended the PINN formulation to identify an unknown material parameter from sparse and noisy displacement measurements.
3. **Variational mechanics** — implemented the Deep Energy Method using total potential energy as the optimization objective.
4. **Verification** — compared neural-network predictions with classical Euler–Bernoulli and Timoshenko beam theories.

The purpose of this project is not to replace analytical solutions for simple beam problems, but to establish a framework that can be extended toward more complex **inverse problems, heterogeneous materials, complex geometries, and continuum-mechanics problems where analytical solutions are unavailable**.

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707.
- Nguyen-Thanh, V. M., Zhuang, X., & Rabczuk, T. (2020). *A deep energy method for finite deformation hyperelasticity*. European Journal of Mechanics - A/Solids, 80, 103874.
