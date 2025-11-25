# Contributing to Universal 4D Viewer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

## Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/universal_4d_viewer.git --recursive
   cd universal_4d_viewer
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

   # Install PyTorch for your CUDA version
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   # Install package in editable mode
   uv pip install -e .
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- **Type Hints**: Use Python 3.10+ type syntax for all function signatures
- **CLI Arguments**: Use `tyro` for type-safe CLI parsing
- **Logging**: Use `logging` module instead of print statements
- **Execution**: Always use `uv run` for script execution
- **Imports**: Use absolute imports (e.g., `from src.domain.entities import GSTensor`)

### Clean Architecture

Follow the dependency rule: code can only depend on layers below it.

```
domain/          <- Core business logic (no external dependencies)
  |
models/ & infrastructure/ <- Application & I/O layers
  |
viewer/          <- Presentation layer
```

See [CLAUDE.md](../CLAUDE.md) for detailed architecture documentation.

### Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/test_ply_loader.py -v
```

Write tests for:
- New features and functionality
- Bug fixes
- Edge cases and error handling

## Pull Request Process

1. **Make Your Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update docstrings and comments

2. **Run Tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

3. **Update Documentation**
   - Update README.md for user-facing changes
   - Update CLAUDE.md for architecture changes
   - Add docstrings to new functions/classes

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

   Commit message format:
   - Start with a verb (Add, Fix, Update, Remove)
   - Keep first line under 72 characters
   - Add details in the body if needed

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

   In your PR:
   - Provide clear title and description
   - Reference related issues
   - Include screenshots/videos for UI changes
   - List any breaking changes

## Areas for Contribution

### High Priority
- Performance optimization (PLY loading, rendering, GPU utilization)
- Documentation improvements (tutorials, examples, API docs)
- Test coverage expansion
- Better error handling and messages

### Feature Ideas
- Support for additional formats (OBJ, COLMAP, etc.)
- Recording/export functionality
- Quality presets for different hardware
- Multi-GPU support
- Web streaming optimization

### Bug Fixes
Check [GitHub Issues](https://github.com/OpsiClear/universal_4d_viewer/issues) for known bugs.

## Common Gotchas

1. **PyTorch First**: Install PyTorch before running `uv pip install -e .`
2. **Hard Refresh**: If you see "viser Version mismatch", do Ctrl+Shift+R in browser
3. **Clean Architecture**: Never import from outer layers into inner layers

See [CLAUDE.md](../CLAUDE.md) for complete list of conventions and gotchas.

## Questions?

- **Architecture & Conventions**: See [CLAUDE.md](../CLAUDE.md)
- **User Documentation**: See [README.md](../README.md)
- **Issues & Discussions**: [GitHub Issues](https://github.com/OpsiClear/universal_4d_viewer/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
