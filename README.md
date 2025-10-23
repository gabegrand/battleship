# Shoot First, Ask Questions Later: Building Rational Agents That Explore and Act Like People

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/downloads/release/python-3100/)

This is the official code repository for the paper [Shoot First, Ask Questions Later: Building Rational Agents That Explore and Act Like People](https://openreview.net/forum?id=XvCBtm5PgF) by Gabriel Grand, Valerio Pepe, Joshua B. Tenenbaum, and Jacob Andreas.

![DisCIPL Framework Overview](docs/static/images/battleship-splash-2.png)

## Getting Started

```
git clone git@github.com:gabegrand/battleship.git
```

### With Poetry (Recommended)

This codebase uses [Poetry](https://python-poetry.org/) to manage dependencies. If you don't have Poetry installed, you can do so by following the instructions [here](https://python-poetry.org/docs/#installation).

```bash
cd battleship
poetry install
```

> [!NOTE]
> If you also want to install optional development dependencies (e.g., for running unit tests, code formatting, and plotting), you can do so with:

```bash
poetry install --with dev
```

Once the installation is complete, you can [activate the virtual environment](https://python-poetry.org/docs/managing-environments/#activating-the-environment):

```bash
# Default with Poetry v2.0
eval $(poetry env activate)

# Alternative with Poetry v1.x or with the poetry-shell plugin
poetry shell
```

## With pip

For convenience, we also provide a [build-system] section in `pyproject.toml`, so you can install the package with pip. We recommend using a virtual environment (e.g., via `venv` or `conda`) to avoid dependency conflicts.

```bash
cd battleship
pip install -e .
```