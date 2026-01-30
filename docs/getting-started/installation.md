# Installation

## Requirements

- Python 3.9+
- pip or conda

## Basic Installation

```bash
# Clone repository
git clone https://github.com/kohr2/mindprint-model
cd mindprint-model

# Install package
pip install -e .
```

## Optional Dependencies

### MLX Backend (Apple Silicon)

```bash
pip install -e ".[mlx]"
```

### Experiment Tracking

```bash
pip install -e ".[tracking]"
```

### Development Tools

```bash
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import src.core.losses; print('Core modules OK')"
python -c "import src.adapters.tracking; print('Adapters OK')"
```
