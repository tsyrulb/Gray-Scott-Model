# Gray-Scott Reaction-Diffusion Simulation

This repository contains Python scripts for simulating and visualizing Gray-Scott reaction-diffusion patterns using different methods: finite-difference for 2D parameter sweep, FEniCS for 2D finite-element simulation, and basic 3D reaction-diffusion.

## Project Structure

- **`gray_scott_enhanced.py`**: Python script containing all simulation and visualization functions.
- **`pattern_results_2d/`**: Directory to store 2D parameter sweep results (generated images).
- **`README.md`**: This file, providing an overview of the project and instructions.

## Features

1. **2D Parameter Sweep**:
   - Sweep over different \( F \) (feed rate) and \( k \) (kill rate) values.
   - Visualize resulting patterns (spots, stripes, labyrinths) using Matplotlib.

2. **2D Finite-Element Solver (FEniCS)**:
   - Implement the Gray-Scott model using FEniCS for higher accuracy.
   - Solve reaction-diffusion equations on a 2D mesh.

3. **3D Reaction-Diffusion** (Basic):
   - Basic implementation of Gray-Scott in 3D.
   - Generate volumetric patterns using NumPy and Matplotlib or advanced visualization libraries.

## Usage

1. **Setup**:
   - Ensure Python 3.x and required libraries (`numpy`, `matplotlib`, `fenics`) are installed.

2. **Running the Simulations**:
   - To run the 2D parameter sweep: `python gray_scott_enhanced.py parameter_sweep_2d`
   - To run the 2D FEniCS solver (requires FEniCS installed): `python gray_scott_enhanced.py run_grayscott_fenics_2d`
   - 3D simulation is embedded in the `gray_scott_enhanced.py` script; modify as needed.

3. **Viewing Results**:
   - Check the `pattern_results_2d/` directory for generated pattern images.
   - Modify visualization parameters and solver settings in the script as required.

## Credits

- This project was inspired by the classic Gray-Scott reaction-diffusion model.
- Developed by tsyrulb for educational and exploratory purposes.

## License

This project is licensed under the MIT License.
