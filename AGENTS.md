# Repository Guidelines

## Project Structure & Module Organization

`src/` implements the Clean Architecture split: `domain/` (entities), `infrastructure/` (streaming, LUTs, file I/O), `models/`, `viewer/`, and `shared/`. Tests live under `tests/architecture`, `tests/functional`, and the root unit specs with fixtures in `tests/conftest.py`. Runtime configs go in `module_config/`; sample captures live in `datasets/` and `examples/`; automation resides in `scripts/`; docs and deep dives sit in `docs/`. External or native dependencies live in `third_party/`; run `git submodule update --init --recursive` after cloning.

## Build, Test, and Development Commands

- `uv sync && uv pip install gsplat` - create the virtual environment, install pyproject dependencies, and add the CUDA splatting extension that remains out-of-tree.
- `uv run viewer --config ./module_config/local_ply.json` - launch the packaged CLI entry point against a local PLY scene; swap the config for Jellyfin presets when streaming.
- `uv run python src/viewer/main.py --config ./datasets/sample_scene/config.json` - run the same entry module directly while iterating on experimental configs.
- `uv run pytest` (full suite) / `uv run pytest -m "not slow and not gpu"` (quick CPU smoke) - cover either the complete matrix or a fast regression slice.

## Coding Style & Naming Conventions

Use Python 3.12, four-space indentation, type annotations on public surfaces, and docstrings for modules that cross layers. Run `black` (88 columns) before committing and order imports as stdlib, third-party, then local. Modules and config files stay snake_case, classes use PascalCase, constants use UPPER_SNAKE_CASE, and config JSON names describe their module. Respect Clean Architecture boundaries: domain stays dependency free and infrastructure never imports viewer components.

## Testing Guidelines

Pytest discovers `test_*.py`, `Test*`, and `test_*` as defined in `pytest.ini`. Apply markers (`unit`, `integration`, `slow`, `gpu`) so collaborators can slice runs, e.g. `uv run pytest tests/functional -m "integration and not slow"`. Reuse fixtures in `tests/conftest.py` for sample frames and LUT paths instead of hardcoding file access. For risky or performance-sensitive work, collect coverage with `uv run pytest --cov=src --cov-report=term-missing` and document GPU requirements inside the test docstring.

## Commit & Pull Request Guidelines

Commits follow `<type>: <imperative summary>` as seen in `docs: comprehensive codebase cleanup and documentation fixes` and `chore: remove obsolete files from refactoring to Clean Architecture`; keep scope narrow and ensure each commit passes `uv run pytest`. Pull requests should link an issue, describe behavior changes, mention configs or assets touched, and include screenshots or short clips when viewer output changes. Highlight migrations, environment variables, or downloads in the PR checklist so reviewers can reproduce.

## Security & Configuration Tips

Store secrets in an untracked `.env` or local config JSON; never commit Jellyfin tokens. Keep `meta_bundle.json`, Gaussian caches, and other large artifacts in `datasets/`, `results/`, or external storage instead of Git history. Before sharing instructions, verify absolute paths inside `module_config/` samples and refresh them whenever parameters change so downstream agents do not ship stale configs.
