# Random Feature Method for PDEs

This repository contains implementations of the Random Feature Method (RFM) for solving partial differential equations. RFM is a meshfree approach that combines ideas from classical numerical methods and machine learning to solve PDEs efficiently.

## What is RFM?

The Random Feature Method uses random feature functions to approximate solutions to PDEs without requiring mesh generation. Unlike traditional finite difference or finite element methods, RFM can handle complex geometries more easily while maintaining the linear optimization framework that makes classical methods reliable.

Main benefits:
- No mesh generation needed
- Works well with complex geometries  
- Avoids non-convex optimization problems
- Can achieve spectral convergence rates
- Scales better to high dimensions

## Implemented Problems

I've implemented RFM for several different types of PDEs:

**Schrödinger Equation** (`RFM_Schodinger_equation.ipynb`)  
Solves the nonlinear Schrödinger equation using complex-valued networks. Includes validation against analytical bright soliton solutions.

**2D Elasticity** (`RFM_2dElasticity_Ⅰ_calculate.py`)  
Linear elasticity problems in 2D domains. Computes stress and displacement fields with proper handling of Neumann boundary conditions.

**Fluid Dynamics** (`RFM_2dFluid_ex2_analytic_pytorch.py`)  
2D Stokes flow equations for incompressible viscous flow. Handles domains with circular obstacles and computes velocity/pressure fields.

**Timoshenko Beams** (`RFM_Timoshenko_beam_pytorch.py`)  
Beam bending analysis using Timoshenko theory. Calculates deflections and stresses for structural engineering applications.

**Viscous Burgers' Equation** (`RFM_Viscous_Burger_2d_pytorch.py`)  
Nonlinear convection-diffusion equation with time evolution. Demonstrates shock wave propagation in 2D.

## Implementation Details

The code is built on PyTorch for automatic differentiation. Each implementation follows a similar pattern:
- Domain decomposition with local neural networks
- Random feature generation for basis functions
- Assembly of linear system Au = f
- Solution using scipy's linear solvers

Most implementations include analytical test cases for validation and error analysis using L² and L∞ norms.

## Usage

Requirements: `torch`, `numpy`, `scipy`, `matplotlib`

Run individual examples:
```bash
python src/RFM_2dElasticity_Ⅰ_calculate.py
python src/RFM_2dFluid_ex2_analytic_pytorch.py  
python src/RFM_Viscous_Burger_2d_pytorch.py
```

The Jupyter notebooks can be opened directly for interactive exploration.


