# Random Feature Method (RFM) for Partial Differential Equations

This repository contains my research work on the Random Feature Method (RFM), a novel numerical approach for solving partial differential equations that combines the advantages of classical numerical methods with machine learning techniques.

## 🔍 Overview

The Random Feature Method is a meshfree numerical approach that bridges classical numerical methods and machine learning for solving PDEs. Unlike traditional finite difference or finite element methods that rely on mesh generation, RFM uses random feature functions to construct numerical solutions while maintaining the linear optimization framework of classical methods.

### Key Advantages
- **Meshfree**: No complex mesh generation required, handles complex geometries easily
- **Linear Optimization**: Avoids non-convex optimization issues common in neural network approaches
- **Spectral Accuracy**: Achieves high convergence rates
- **Scalable**: Suitable for high-dimensional problems
- **Stable**: Maintains theoretical convergence guarantees

## 📚 Implemented Applications

This repository demonstrates RFM implementations for various physical problems:

### 1. **Schrödinger Equation** (`RFM_Schodinger_equation.ipynb`)
- Nonlinear Schrödinger equation solver
- Complex-valued neural networks
- Bright soliton validation

### 2. **2D Elasticity Problems** (`RFM_2dElasticity_Ⅰ_calculate.py`)
- Linear elasticity in 2D domains
- Stress and displacement field calculations
- Neumann boundary condition handling

### 3. **Fluid Dynamics** (`RFM_2dFluid_ex2_analytic_pytorch.py`)
- 2D Stokes flow equations
- Velocity and pressure field computation
- Complex domain geometries with circular obstacles

### 4. **Timoshenko Beam Theory** (`RFM_Timoshenko_beam_pytorch.py`)
- Structural mechanics beam analysis
- Deflection and stress calculations
- Engineering applications

### 5. **Viscous Burgers' Equation** (`RFM_Viscous_Burger_2d_pytorch.py`)
- Nonlinear convection-diffusion equation
- 2D implementation with time evolution
- Shock wave propagation

## 🛠️ Technical Implementation

### Core Architecture
- **PyTorch-based**: Leverages automatic differentiation for gradient calculations
- **Modular Design**: Separate implementations for different equation types
- **Local Networks**: Domain decomposition with local neural networks
- **Linear Solver**: Uses `scipy.linalg.lstsq` for efficient linear system solution


## 📊 Results and Validation

The implementations include:
- **Analytical Comparisons**: Validation against known exact solutions
- **Error Analysis**: L² and L∞ norm computations
- **Convergence Studies**: Rate analysis with varying parameters
- **Visualization**: Error distribution and solution field plots


