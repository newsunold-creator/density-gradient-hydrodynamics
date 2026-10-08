# Density Gradient Hydrodynamics Simulator

A numerical simulator of an alternative density gradient-based hydrodynamics model that demonstrates the absence of singularities in fluid dynamics.

## Description

This simulator demonstrates that a hydrodynamics model in which velocity is defined as a function of the density gradient avoids the singularities inherent in the classical Navier-Stokes equations. The model transforms the governing equations into a parabolic (diffusive) system, ensuring that solutions remain bounded and smooth for all time.

### Key Features

- **Linear diffusion model**: Classical equation ∂ρ/∂t = D∇²ρ with explicit finite difference scheme
- **Nonlinear porous medium equation**: ∂ρ/∂t = ½∇²(ρ²) with conservative discretization
- **Validation against exact Barenblatt-Pattle self-similar solution**
- **GPU acceleration** via PyTorch/CUDA
- **Automated screenshot generation** at key time steps
- **L1 error tracking** demonstrating numerical accuracy (~10⁻⁴)

## Mathematical Model

### Fundamental Postulate

Fluid velocity is defined as a function of the density gradient:

**v(x,t) = -D(ρ)∇ρ**

This leads to the general diffusion equation:

**∂ρ/∂t = ∇·(ρD(ρ)∇ρ)**

For the nonlinear case D(ρ) = ρ, this becomes the porous medium equation:

**∂ρ/∂t = ½∇²(ρ²)**

### Maximum Principle

The model satisfies the maximum principle:

**max(ρ(x,t)) ≤ max(ρ(x,0)) for all t > 0**

This guarantees that singularities (infinite values) are mathematically impossible.

## Installation

```bash
pip install torch matplotlib numpy scipy
