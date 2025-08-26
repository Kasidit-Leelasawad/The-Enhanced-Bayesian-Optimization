# Mixed Integer Bayesian Optimisation in High-Dimensional Search Spaces

A novel Enhanced Bayesian Optimization (BO) framework designed to overcome the curse of dimensionality in high-dimensional spaces (20+ dimensions) and handle mixed-integer optimization problems - two significant challenges that limit standard BO and other optimization algorithms.

## ✨ Key Features

- **Multi-Trust Regions**: Efficiently explores high-dimensional search spaces through localized optimization
- **Hybrid Acquisition Function**: Combines Upper Confidence Bound (UCB) and Thompson Sampling (TS) for balanced exploration-exploitation
- **NSGA-II Integration**: Multi-objective optimization capability for complex trade-offs
- **Adaptive Parameters**: Dynamic adjustment of:
  - Batch size
  - Number of trust regions
  - Acquisition function weights
- **Mixed-Integer Handling**: Native support for both continuous and discrete variables
- **High Performance**: Consistently achieves performance scores above 0.8 in experimental validations

## 🔄 Algorithm Workflow

![Enhanced BO Workflow](https://github.com/Kasidit-Leelasawad/The-Enhanced-Bayesian-Optimization/blob/main/docs/workflow.png)

The optimization process consists of three main phases:

### 1. Initialization Phase
- **Setup Variables**: Identify integer/continuous variables, create value mappings, initialize bounds and scales
- **Initialize Components**: GP surrogate (Matérn 5/2 kernel), Trust Region Manager (n=3), NSGA-II evolutionary optimizer, Adaptive parameters (batch, weights)

### 2. Main Optimization Loop
- **Suggest Phase**: 
  - Early iterations use Random sampling (Sobol)
  - Later iterations generate candidates from each Trust Region, handle integer variables, and calculate multi-objective acquisition (UCB + TS)
- **Evaluate Phase**: 
  - Map integer indices to actual values
  - Evaluate objective function
  - Track improvements
- **Update Phase**: 
  - Update GP model with new data
  - Update Trust Region centers if improvement found
  - Adjust Trust Region lengths (expand/shrink)
  - Trigger adaptive adjustments if needed

## 📊 Experimental Validation

Our Enhanced BO has been extensively tested on three categories of problems:

### 1. Benchmark Functions
Validated on five standard optimization benchmarks:
- Rosenbrock Function
- Levy Function
- Rastrigin Function
- Ackley Function
- 1-Norm Function

All functions demonstrated excellent convergence and solution quality, particularly in high-dimensional settings where traditional methods struggle.

### 2. CSTR Model (Continuous Stirred-Tank Reactor)
Successfully optimized complex chemical reactor parameters including:
- Temperature profiles
- Residence time
- Catalyst concentration
- Feed composition

The algorithm showed robust performance in handling the nonlinear dynamics and constraints typical of chemical processes.

### 3. Latent Function Problem
Demonstrated strong capability in optimizing systems with hidden variables and uncertain parameters, effectively quantifying and managing uncertainty throughout the optimization process.

## 🚀 Installation

### Requirements
```
Python >= 3.8
numpy >= 1.20.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
torch >= 1.9.0
botorch >= 0.6.0
matplotlib >= 3.3.0
```

### Install from GitHub
```bash
git clone https://github.com/Kasidit-Leelasawad/The-Enhanced-Bayesian-Optimization.git
cd The-Enhanced-Bayesian-Optimization
pip install -r requirements.txt
```

## 📝 Citation

The corresponding manuscript for this repository is currently under preparation. In the interim, please cite this work as follows:

```bibtex
@article{enhanced_bo_2024,
  title={Mixed Integer Bayesian Optimisation in High-Dimensional Search Spaces},
  author={Leelasawad, K. and del Rio Chanona, A.},
  journal={Manuscript in preparation},
  year={2024}
}
```

## 👥 Authors

- **Kasidit Leelasawad** - *Main Developer*
- **Prof. Antonio del Rio Chanona** - *Principal Investigator*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and support:
- Repository: [https://github.com/Kasidit-Leelasawad/The-Enhanced-Bayesian-Optimization](https://github.com/Kasidit-Leelasawad/The-Enhanced-Bayesian-Optimization)
- Issues: [GitHub Issues](https://github.com/Kasidit-Leelasawad/The-Enhanced-Bayesian-Optimization/issues)
