# Contributing to gsplay

Thank you for your interest in contributing!

## Attribution

This project builds upon excellent open-source work:

- **[nerfview](https://github.com/hangg7/nerfview)** - The original NeRF viewer (MIT License)
- **[viser](https://github.com/nerfstudio-project/viser)** - Web-based 3D visualization framework (MIT License)

## Copyright Agreement

By submitting a pull request or any contribution to this project, you agree that:

1. **Copyright Assignment**: All contributions become the exclusive property of **OpsiClear LLC**
2. **License Grant**: Your contributions will be licensed under AGPL-3.0 as part of this project
3. **Original Work**: You certify that your contribution is your original work or you have the right to submit it

## Quick Start

1. **Fork and Clone**

   ```bash
   git clone https://github.com/opsiclear/gsplay.git
   cd gsplay
   ```

2. **Set Up Environment**

   ```bash
   uv venv && source .venv/bin/activate
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   uv pip install -e .
   ```

3. **Create Branch, Make Changes, Submit PR**

   ```bash
   git checkout -b feature/your-feature
   # Make changes, add tests
   git commit -m "Add your feature"
   git push origin feature/your-feature
   # Open PR on GitHub
   ```

## Guidelines

- **Python 3.12+** with type hints
- **Use `uv run`** for script execution
- **Use logging** instead of print statements
- **Follow Clean Architecture** - see [CLAUDE.md](../CLAUDE.md)
- **Add tests** for new functionality

## Questions?

- Architecture: [CLAUDE.md](../CLAUDE.md)
- Issues: [GitHub Issues](https://github.com/opsiclear/gsplay/issues)
