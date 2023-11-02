# battleship

## Installation

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