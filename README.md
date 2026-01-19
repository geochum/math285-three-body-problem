# Module 1 Final Project - Three-Body Problem

Numerical solutions to the three-body problem using various numerical integration methods.

## Project Structure

```
Module 1/
├── src/                    # Source code
│   └── main.py            # Main simulation script
├── outputs/                # Generated visualizations
│   ├── *.png              # Static plots
│   └── *.gif              # Animated visualizations
├── docs/                   # Documentation
│   ├── Final Project.pdf  # Project report
│   └── Module 1 Final Project.nb  # Mathematica notebook
├── final_project.wpr      # Wing IDE workspace configuration
└── README.md              # This file
```

## Description

This project implements numerical methods to solve the three-body problem in celestial mechanics. The code uses several numerical integration schemes:

- **Explicit Euler Method**: First-order explicit method
- **Adams-Bashforth (2-step)**: Second-order multi-step method
- **Runge-Kutta 4**: Fourth-order explicit method

## Features

- Multiple numerical integration schemes
- 3D visualization of trajectories
- Animated GIF generation
- Support for various initial conditions (Figure 8, Dragonfly, Yin-Yang, etc.)

## Usage

1. **Activate the virtual environment** (if using one):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run the simulation** from the project root:
   ```bash
   python src/main.py
   ```

Outputs (PNG images and GIF animations) will be saved to the `outputs/` directory.

## Requirements

- Python 3.x
- numpy
- matplotlib

## Setup

### Using Virtual Environment (Recommended)

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Without Virtual Environment

If you prefer to install packages globally:
```bash
pip install -r requirements.txt
```

## Initial Conditions

The script includes several pre-configured initial conditions for different periodic orbits:
- Ovals with flourishes
- Figure 8
- Dragonfly
- Yin-Yang 1a/1b
- Yarn
- Goggles
- Skinny pineapple

Uncomment the desired configuration in `src/main.py` to run different simulations.
