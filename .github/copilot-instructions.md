# MLC LLM Copilot Instructions

## Project Overview
MLC LLM is a machine learning compiler and high-performance deployment engine for large language models. It provides universal LLM deployment across multiple platforms including Linux, macOS, iOS, Android, and web browsers with OpenAI-compatible APIs.

## Build System
- **Python packaging**: Uses `setuptools` via `python/setup.py`
- **C++ build**: CMake-based system for platform-specific backends (Vulkan, Metal, CUDA, OpenCL)
- **Build commands**:
  - `pip install -e .` - Install in development mode
  - `python -m build --wheel` - Build wheel package
  - CMake builds for native components

## Test Framework
- Uses `pytest` with custom markers defined in `python/conftest.py`
- Test categorization via `@pytestmark = [pytest.mark.category_name]`

# Testing
- Run tests: `python -m pytest tests/python/ -m unittest`
- Common markers: `unittest`, `category_name`, etc.

## Key Components
1. **MLCEngine**: Core inference engine with OpenAI-compatible API
2. **Platform backends**: Vulkan, Metal, CUDA, OpenCL implementations
3. **Quantization pipelines**: AWQ, group quantization support
4. **Tensor parallelism**: Multi-GPU execution via tensor sharding
5. **REST server**: Python and JavaScript interfaces

## Architecture Patterns
- Three-phase workflow: download weights → compile model → execute
- TVM-based compiler infrastructure for model optimization
- Entry points defined in `pyproject.toml` for CLI tools
- Platform-specific implementations with shared API contracts

## Development Conventions
- Python type hints are encouraged (see `python/pyproject.toml`)
- Use relative imports within python packages
- C++ code follows Apache TVM conventions
- Documentation uses Sphinx with reStructuredText format

## Common Tasks
- **Run unit tests**: `pytest tests/python/ -m unittest`
- **Build wheel`: `python -m build --wheel`
- **Install dev mode**: `pip install -e .`
- **Generate docs**: Sphinx-based in `/home/kristoffer/mlc-llm/docs/`

## Platform Support Matrix
- Linux/Win: Vulkan, ROCm, CUDA
- macOS: Metal (dGPU and iGPU)
- Web Browser: WebGPU and WASM
- iOS/iPadOS: Metal on Apple A-series GPU
- Android: OpenCL on Adreno/Mali GPUs


## Documentation Structure
- User guides in `/docs/get_started/`
- API reference auto-generated from docstrings
- Deployment guides in `/docs/deploy/`
- Installation instructions in `/docs/install/`
