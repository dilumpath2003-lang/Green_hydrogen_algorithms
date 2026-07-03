# Green Hydrogen Generation: Optimized Algorithms & Simulation

An advanced simulation and optimization framework designed to maximize green hydrogen generation efficiency. This project implements evolutionary and metaheuristic optimization algorithms to balance energy supply, cost, and electrolyzer performance constraints.

## 🚀 Key Features
* **Metaheuristic Optimization**: Implementations of Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and Genetic Algorithms (GA).
* **Efficiency Analysis**: Mathematical modeling of electrolyzer efficiency curves relative to variable renewable energy inputs.
* **Data Visualization**: Automated generation of convergence behavior charts and industrial performance metrics.

## 📁 Repository Structure
* `final_docs/` — Core production-ready Python optimization scripts.
  * `CMA_ES.py` — Covariance Matrix Adaptation Evolution Strategy optimization loop.
  * `Genotictest.py` — Genetic Algorithm test bed for tracking population convergence.
  * `Efficiency_Plot.py` — Script to render electrolyzer efficiency curves.
  * `Industry.py` — Industrial baseline data and processing constraints.
* `Hygrogen generation algorithm/` — Directory containing algorithm convergence plots and performance benchmarks.

## 📊 Sample Visualizations
The optimization algorithms generate and save high-resolution benchmark plots directly to the workspace, tracking parameters across iterations to verify algorithmic stability:
* **CMA-ES Convergence Logs**: Iteration vs. objective function performance.
* **Genetic Algorithm Stability**: Evaluation of generation-over-generation fitness improvements.
* **Electrolyzer Efficiency Profiles**: Dynamic tracking of system efficiency against fluctuating input power supply profiles.

## 🛠️ Requirements & Installation
Ensure you have Python installed, then set up the required numerical processing and plotting packages:

```bash
pip install numpy matplotlib scipy cma
```

## 💻 Usage
To run the primary CMA-ES optimization strategy and observe the generation convergence metrics:
```bash
python final_docs/CMA_ES.py
```
