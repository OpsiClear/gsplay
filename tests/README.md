# Test Suite Documentation

This directory contains all tests for the Universal 4D Viewer project.

## Test Organization

Tests are organized into categories for easy navigation and execution:

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for feature workflows
├── architecture/            # Architecture and refactoring tests
└── functional/              # Functional and end-to-end tests
```

### Unit Tests (`unit/`)

Tests for individual components and utilities in isolation:

- `test_activation_lut.py` - Tests for activation lookup tables
- `test_color_lut.py` - Tests for color lookup tables
- `test_frame_cache.py` - Tests for frame caching system
- `test_natural_sort.py` - Tests for natural sorting utilities

**Run unit tests**:
```bash
pytest tests/unit/ -v
```

### Integration Tests (`integration/`)

Tests for feature integrationand workflows:

- `test_integration.py` - Integration tests for OptimizedPlyModel
- `test_14prop_loading.py` - Tests for 14-property PLY loading
- `test_pyminiply_gaussian.py` - Tests for pyminiply Gaussian loading
- `test_gsply.py` - Tests for gsply library integration

**Run integration tests**:
```bash
pytest tests/integration/ -v
```

### Architecture Tests (`architecture/`)

Tests for architecture refactoring and flexibility improvements:

- `test_backward_compatibility.py` - Tests for backward compatibility after refactoring
- `test_flexibility.py` - Tests for flexibility improvements (components, events, etc.)
- `test_improvements.py` - Tests for Phase 1 improvements
- `test_phase2_improvements.py` - Tests for Phase 2 improvements
- `test_phase3_improvements.py` - Tests for Phase 3 improvements

**Run architecture tests**:
```bash
pytest tests/architecture/ -v
```

### Functional Tests (`functional/`)

End-to-end functional tests:

- `test_ply_roundtrip.py` - Tests for PLY export/import roundtrip

**Run functional tests**:
```bash
pytest tests/functional/ -v
```

## Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Architecture tests only
pytest tests/architecture/ -v

# Functional tests only
pytest tests/functional/ -v
```

### Run Specific Test File

```bash
pytest tests/unit/test_frame_cache.py -v
```

### Run Specific Test Function

```bash
pytest tests/unit/test_frame_cache.py::test_cache_hit -v
```

## Test Configuration

Test configuration is managed in `pytest.ini` at the project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## Writing Tests

### Test Structure

Follow these conventions when writing tests:

1. **File naming**: `test_<module_name>.py`
2. **Class naming**: `Test<FeatureName>` (for grouping related tests)
3. **Function naming**: `test_<specific_behavior>`
4. **Assertions**: Use clear assertion messages

Example:
```python
class TestFrameCache:
    """Tests for frame caching system."""

    def test_cache_hit_returns_cached_data(self):
        """Test that cache returns correct data on hit."""
        cache = FrameCache(max_size=10)
        # ... test implementation
        assert result == expected, "Cache should return correct data"
```

### Test Categories

- **Unit tests**: Test single functions/classes in isolation
  - Mock external dependencies
  - Fast execution (< 100ms per test)
  - No I/O operations

- **Integration tests**: Test multiple components working together
  - May use real dependencies
  - Moderate execution time
  - May include I/O operations

- **Architecture tests**: Test architectural properties
  - Backward compatibility
  - Component structure
  - Import patterns

- **Functional tests**: Test complete workflows
  - End-to-end scenarios
  - Real data and operations
  - Longer execution time acceptable

## Test Fixtures

Common fixtures are defined in `conftest.py` files:

- `tests/conftest.py` - Project-wide fixtures
- `tests/unit/conftest.py` - Unit test fixtures
- `tests/integration/conftest.py` - Integration test fixtures

## Coverage Goals

- **Unit tests**: > 80% coverage
- **Integration tests**: > 60% coverage
- **Critical paths**: 100% coverage

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

## Contributing Tests

When contributing:
1. Add tests for new features
2. Add tests for bug fixes
3. Update tests when modifying existing functionality
4. Ensure all tests pass before submitting PR

See [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) for more details.

## Test Status

**Last Updated**: 2025-11-13
**Total Tests**: 50+
**Test Coverage**: ~75%

All tests are passing as of the last reorganization.
