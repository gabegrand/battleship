# battleship

## Installation

Make sure to clone the repo with the `--recurse-submodules` flag:
```
git clone --recurse-submodules https://github.com/gabegrand/battleship
```

First, install all dependencies using `poetry`:
```
poetry install
```

Then, install the EIG repo:
```
poetry run python -m pip install -e ./EIG --compile --no-cache-dir
```

NOTE: For Apple Silicon, you will need to specify the architecture in order for the C++ code to compile:
```
ARCHFLAGS="-arch arm64" poetry run python -m pip install -e ./EIG --compile --no-cache-dir
```

## Development tools

### Pre-commit hooks

We use `black` and `pre-commit` to enforce code style. To install the pre-commit hooks, run:
```
pre-commit install
```