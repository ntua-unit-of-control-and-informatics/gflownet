

[![Build-and-Test](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml)
[![Code Quality](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# gflownet

GFlowNet-related training and environment code on graphs.

**Primer**

GFlowNet [[1]](https://yoshuabengio.org/2022/03/05/generative-flow-networks/), [[2]](https://www.gflownet.org/), [[3]](https://github.com/zdhNarsil/Awesome-GFlowNets), short for Generative Flow Network, is a novel generative modeling framework, particularly suited for discrete, combinatorial objects. Here in particular it is implemented for graph generation.

The idea behind GFN is to estimate flows in a (graph-theoretic) directed acyclic network*. The network represents all possible ways of constructing objects, and so knowing the flow gives us a policy which we can follow to sequentially construct objects. Such a sequence of partially constructed objects is a _trajectory_. *Perhaps confusingly, the _network_ in GFN refers to the state space, not a neural network architecture.

The main focus of this library (although it can do other things) is to construct graphs (e.g. graphs of atoms), which are constructed node by node. To make policy predictions, we use a graph neural network. This GNN outputs per-node logits (e.g. add an atom to this atom, or add a bond between these two atoms), as well as per-graph logits (e.g. stop/"done constructing this object").

This library supports a variety of GFN algorithms (as well as some baselines), and supports training on a mix of existing data (offline) and self-generated data (online), the latter being obtained by querying the model sequentially to obtain trajectories.


## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package wheels, the `--find-links` arguments must be specified as well:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```
Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

To install or [depend on](https://matiascodesal.com/blog/how-use-git-repository-pip-dependency/) a specific tag, for example here `v0.0.10`, use the following scheme:
```bash
pip install git+https://github.com/recursionpharma/gflownet.git@v0.0.10 --find-links ...
```

If package dependencies seem not to work, you may need to install the exact frozen versions listed `requirements/`, i.e. `pip install -r requirements/main-3.10.txt`.

## Getting started

A good place to get started immediately is with the [sEH fragment-based MOO task](src/gflownet/tasks/seh_frag_moo.py). The file `seh_frag_moo.py` is runnable as-is (although you may want to change the default configuration in `main()`).

For a gentler introduction to the library, see [Getting Started](docs/getting_started.md). For a more in-depth look at the library, see [Implementation Notes](docs/implementation_notes.md).

## Repo overview

- [algo](src/gflownet/algo), contains GFlowNet algorithms implementations ([Trajectory Balance](https://arxiv.org/abs/2201.13259), [SubTB](https://arxiv.org/abs/2209.12782), [Flow Matching](https://arxiv.org/abs/2106.04399)), as well as some baselines. These implement how to sample trajectories from a model and compute the loss from trajectories.
- [data](src/gflownet/data), contains dataset definitions, data loading and data sampling utilities.
- [envs](src/gflownet/envs), contains environment classes; the base environment is agnostic to what kind of graph is being made, and context classes specify mappings from graphs to objects (e.g. molecules) and torch geometric Data.
- [models](src/gflownet/models), contains model definitions.
- [tasks](src/gflownet/tasks), contains training code.
    -  [qm9](src/gflownet/tasks/qm9/qm9.py), temperature-conditional molecule sampler based on QM9's HOMO-LUMO gap data as a reward.
    -  [seh_frag](src/gflownet/tasks/seh_frag.py), reproducing Bengio et al. 2021, fragment-based molecule design targeting the sEH protein
    -  [seh_frag_moo](src/gflownet/tasks/seh_frag_moo.py), same as the above, but with multi-objective optimization (incl. QED, SA, and molecule weight objectives).
- [utils](src/gflownet/utils), contains utilities (multiprocessing, metrics, conditioning).
- [`trainer.py`](src/gflownet/trainer.py), defines a general harness for training GFlowNet models.
- [`online_trainer.py`](src/gflownet/online_trainer.py), defines a typical online-GFN training loop.

See [implementation notes](docs/implementation_notes.md) for more.


## Developing & Contributing

External contributions are welcome.

To install the developers dependencies
```
pip install -e '.[dev]' --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

We use `tox` to run tests and linting, and `pre-commit` to run checks before committing.
To ensure that these checks pass, simply run `tox -e style` and `tox run` to run linters and tests, respectively.

For more information, see [Contributing](docs/contributing.md).


## How to run the repo

This repository is intended to be run with **Python 3.10** inside a virtual environment (recommended: Conda).
All configuration is handled via **command-line arguments** — source files do **not** need to be edited.

You can run all commands either from a terminal (Anaconda Prompt / PowerShell / bash)
or from the integrated terminal in **Visual Studio Code**.
The same rules apply in both cases.

---

#### 1) Clone the repository

Choose a directory on your system where you want the code to live, then run:

```bash
git clone -b full-example --single-branch https://github.com/ntua-unit-of-control-and-informatics/jaqpot-gflownet-model.git
cd jaqpot-gflownet-model
```

The repository is downloaded into a folder named `jaqpot-gflownet-model`
inside your current directory.

---

#### 2) Create and activate a virtual environment (one time only)

The environment needs to be **created only once**.
Every time you want to run the code, you only need to **activate** it.

```bash
conda create -n gflownet_env python=3.10
conda activate gflownet_env
```

Make sure the environment is activated **before running any Python commands**.

> **Optional**: If you plan to run the analysis notebook (`analyze_results.ipynb`) in
**Visual Studio Code or Jupyter**, you need to install `ipykernel`
**once for this environment**.
>
> With the environment activated:
>
>```bash
>conda install ipykernel
>```
>This step is required only once per environment. Afterwards, gflownet_env will be available as a selectable Jupyter kernel
in VS Code and other notebook interfaces.

---

#### 3) Install dependencies (from the repository root)

Always run installation commands from the repository root.

**CPU-only installation (recommended default):**
```bash
cd path/to/jaqpot-gflownet-model
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
pip install numpy==1.26.4
```

**GPU installation (optional, NVIDIA GPUs only):**
```bash
cd path/to/jaqpot-gflownet-model
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install numpy==1.26.4
```

Use the GPU option only if you have an NVIDIA GPU with compatible drivers.
If you are unsure, use the CPU installation above.

**GPU usage note for task examples:**
- The following scripts **require a GPU by default** (they explicitly set `device="cuda"`):
  - `toy_seq.py`
  - `make_rings.py`
- The following scripts **auto-detect** and run on CPU or GPU:
  - `example.py`
  - `seh_frag.py`
  - `seh_frag_moo.py`
  - `qm9.py`
  - `qm9_moo.py`

---

#### 4) Running from Visual Studio Code

Before opening Visual Studio Code, **ensure that the correct environment is already activated**.
If the environment is active when VS Code is launched, the integrated terminal will inherit it.

From the repository root, with the environment activated:

```bash
cd path/to/jaqpot-gflownet-model
code .
```

Inside VS Code:
- Open a terminal (**Terminal → New Terminal**)
- Verify that `gflownet_env` is active
- Run the commands below exactly as shown

You do **not** need to re-activate the environment if it is already active.

---

#### 5) Train a predictive (proxy) model

Navigate explicitly to the proxy folder:

```bash
cd path/to/jaqpot-gflownet-model/src/gflownet/proxy
```

Train a proxy model using the default dataset and hyperparameters:

```bash
python train.py
```

To inspect all available command-line options and dataset requirements, run:
```bash
python train.py --help
```

To use a **custom dataset or change training settings**, run:

```bash
python train.py   --data_url "C:\path\to\dataset.csv"   --target_col logKOW   --learning_rate 0.001   --batch_size 64   --gnn_layers 2   --gnn_channels 64   --heads 4   --mlp_layers 2   --dropout_proba 0.2   --best_model_out best_model.pt   --params_out model_params.txt
```

**Command-line arguments for `train.py`:**
- `--data_url`  : Path or URL to a CSV dataset. The dataset must contain a `smiles` column.
- `--target_col` : Name of the target column in the dataset to be predicted (e.g. `logKOW`).
- `--best_model_out` : Filename where the best-performing model checkpoint is saved.
- `--params_out` : Filename where all training hyperparameters are stored for reproducibility.
- `--learning_rate` : Learning rate used by the Adam optimizer.
- `--batch_size` : Batch size for training the proxy model.
- `--gnn_layers` : Number of graph neural network layers.
- `--gnn_channels` : Hidden dimension size of the GNN layers.
- `--heads` : Number of attention heads in the graph attention layers.
- `--mlp_layers` : Number of fully connected layers after the GNN.
- `--dropout_proba` : Dropout probability used for regularization.

---

#### 6) Train a GFlowNet using the proxy model

Navigate explicitly to the tasks folder:

```bash
cd path/to/jaqpot-gflownet-model/src/gflownet/tasks
```

Run GFlowNet training with default settings:

```bash
python example.py
```

To inspect all available optimization and proxy-related options, run:
```bash
python example.py --help
```

To customize the optimization task and proxy model used:

```bash
python example.py   --objective min   --min_logp -13.71   --max_logp 2.41   --param_file ../proxy/model_params.txt   --model_file ../proxy/best_model.pt   --log_dir ./logs/min_run
```

**Command-line arguments for `example_inputs.py`:**
- `--objective` : Optimization direction for the reward.  Use `min` to minimize the proxy output or `max` to maximize it.
- `--min_logp`, `--max_logp` : Minimum and maximum values used to scale the proxy output into a normalized reward.
- `--param_file` : Path to the proxy model parameter file generated during proxy training.
- `--model_file` : Path to the trained proxy model weights.
- `--log_dir` : Directory where GFlowNet logs, checkpoints, and generated samples are saved.

---

#### 7) Analyze results (interactive notebook)

After GFlowNet training completes, navigate back to the repository root and open:

```bash
analyze_results.ipynb
```

This notebook provides an interactive interface to:

- load a trained GFlowNet run,
- generate new molecular designs,
- visualize results (tables, plots, molecule drawings),
- export generated molecules to CSV.

All inputs are provided through widgets — no source code edits are required.

>Make sure the notebook kernel is set to `gflownet_env (Python 3.10)`.

Inside the notebook you will find:
- input fields for:
  - GFlowNet log ID (training run to analyze),
  - proxy model parameter file,
  - proxy model weights,
  - number of molecules to generate,
  - number of sampling rounds,
  - random seed,
- a Run generation button with status feedback,
- interactive plots 
- a molecule viewer with dropdown selection,
- a Download CSV button to export generated designs.

The notebook is fully self-contained and designed for exploration, evaluation, and visualization of trained GFlowNet models.

> To run the notebook on JupyterLab instead of VS code, first make sure you have installed jupyterlab inside the environment:
> ```bash
> conda activate gflownet_env
> conda install jupyterlab
> conda install -c conda-forge matplotlib ipywidgets
> pip install matplotlib
> ```
> You only need to install this once. Then launch:
> ```bash
> jupyter lab
>```
> Open analyze_results.ipynb and use the preselected kernel.
---
