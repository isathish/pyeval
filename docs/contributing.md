# Contributing to PyEval

Thank you for your interest in contributing to PyEval! This guide will help you get started.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

---

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Minimal code example

```markdown
### Bug Report

**Description:** Brief description of the bug

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:** What should happen

**Actual Behavior:** What actually happens

**Environment:**
- Python: 3.x.x
- PyEval: x.x.x
- OS: [Windows/macOS/Linux]

**Minimal Example:**
```python
from pyeval import metric_name
# Code that demonstrates the bug
```
```

### Suggesting Features

1. **Check existing issues** for similar suggestions
2. **Create a feature request** with:
   - Clear use case
   - Proposed API design
   - Examples of how it would work

### Contributing Code

#### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/pyeval.git
cd pyeval

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests to verify setup
pytest
```

#### Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run the test suite**:
   ```bash
   pytest
   pytest --cov=pyeval  # With coverage
   ```

5. **Run linting**:
   ```bash
   ruff check pyeval/
   ruff format pyeval/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new metric for X"
   ```

7. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

---

## Coding Standards

### Style Guide

- Follow **PEP 8** conventions
- Use **type hints** for all function signatures
- Maximum line length: **88 characters** (Black default)
- Use **descriptive variable names**

### Code Structure

```python
"""Module docstring describing the module's purpose."""

from typing import List, Optional, Union

__all__ = ['public_function', 'PublicClass']


def public_function(
    param1: List[int],
    param2: str,
    *,
    optional_param: Optional[float] = None
) -> float:
    """
    Brief description of the function.
    
    Longer description if needed. Explain the algorithm,
    edge cases, or any important details.
    
    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.
        optional_param: Description of optional parameter.
            Defaults to None.
    
    Returns:
        Description of the return value.
    
    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not a string.
    
    Examples:
        >>> public_function([1, 2, 3], "test")
        0.75
        
        >>> public_function([1, 2], "test", optional_param=0.5)
        0.80
    
    Notes:
        Any additional notes about the implementation.
    
    References:
        - Paper or source reference if applicable
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    # Implementation
    result = 0.0
    
    return result
```

### Docstring Format

We use Google-style docstrings:

```python
def example_metric(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the example metric.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    
    Returns:
        The computed metric value between 0 and 1.
    
    Raises:
        ValueError: If inputs have different lengths.
    
    Examples:
        >>> example_metric([1, 0, 1], [1, 0, 0])
        0.667
    """
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Union, Tuple, Callable

def metric(
    y_true: List[int],
    y_pred: List[int],
    *,
    average: str = 'binary',
    sample_weight: Optional[List[float]] = None
) -> Union[float, Dict[str, float]]:
    ...
```

---

## Adding New Metrics

### Metric Implementation Checklist

- [ ] Function follows the standard signature pattern
- [ ] Has comprehensive docstring with examples
- [ ] Includes type hints
- [ ] Has input validation
- [ ] Has unit tests (aim for >95% coverage)
- [ ] Added to `__all__` in module
- [ ] Added to module's `__init__.py`
- [ ] Added to main `pyeval/__init__.py`
- [ ] Documented in API reference

### Example: Adding a New Metric

```python
# In pyeval/ml/classification.py

from typing import List

__all__ = [..., 'new_metric_score']


def new_metric_score(
    y_true: List[int],
    y_pred: List[int],
    *,
    normalize: bool = True
) -> float:
    """
    Calculate the new metric score.
    
    The new metric measures... [description]
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        normalize: Whether to normalize the result.
            Defaults to True.
    
    Returns:
        The metric value. If normalize=True, returns
        a value between 0 and 1.
    
    Raises:
        ValueError: If y_true and y_pred have different lengths.
        ValueError: If inputs contain values other than 0 and 1.
    
    Examples:
        >>> new_metric_score([1, 0, 1, 1], [1, 0, 0, 1])
        0.75
        
        >>> new_metric_score([1, 0, 1], [1, 1, 1], normalize=False)
        2.0
    
    References:
        - Author et al. "Paper Title". Conference/Journal, Year.
          https://doi.org/...
    """
    # Validation
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )
    
    if not y_true:
        raise ValueError("Inputs cannot be empty")
    
    # Implementation
    score = sum(t == p for t, p in zip(y_true, y_pred))
    
    if normalize:
        score = score / len(y_true)
    
    return score
```

### Writing Tests

```python
# In tests/test_ml/test_classification.py

import pytest
from pyeval import new_metric_score


class TestNewMetricScore:
    """Tests for new_metric_score function."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        assert new_metric_score(y_true, y_pred) == 1.0
    
    def test_all_wrong_predictions(self):
        """Test with all incorrect predictions."""
        y_true = [1, 0, 1, 0]
        y_pred = [0, 1, 0, 1]
        assert new_metric_score(y_true, y_pred) == 0.0
    
    def test_partial_correct(self):
        """Test with partially correct predictions."""
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        assert new_metric_score(y_true, y_pred) == 0.75
    
    def test_normalize_false(self):
        """Test with normalize=False."""
        y_true = [1, 0, 1]
        y_pred = [1, 1, 1]
        result = new_metric_score(y_true, y_pred, normalize=False)
        assert result == 2.0
    
    def test_empty_input_raises(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            new_metric_score([], [])
    
    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            new_metric_score([1, 0], [1, 0, 1])
    
    def test_single_element(self):
        """Test with single element inputs."""
        assert new_metric_score([1], [1]) == 1.0
        assert new_metric_score([1], [0]) == 0.0
    
    @pytest.mark.parametrize("y_true,y_pred,expected", [
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),
        ([0, 0, 0, 0], [0, 0, 0, 0], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0, 1], [1, 0, 0], 2/3),
    ])
    def test_various_cases(self, y_true, y_pred, expected):
        """Test various input combinations."""
        result = new_metric_score(y_true, y_pred)
        assert abs(result - expected) < 1e-10
```

---

## Documentation

### Updating Documentation

1. **API Reference**: Update `docs/api/` when adding new functions
2. **Examples**: Add to `docs/examples/` for complex features
3. **Changelog**: Update `docs/changelog.md` for all changes

### Building Documentation Locally

```bash
# Install MkDocs and dependencies
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Build static site
mkdocs build
```

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(ml): add balanced accuracy score metric
fix(nlp): handle empty reference in BLEU score
docs(api): add examples for fairness metrics
test(rag): add tests for context precision
refactor(core): simplify validation logic
```

---

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**: `pytest`
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

### PR Title Format

```
type(scope): Brief description
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] All tests passing
- [ ] Code follows style guidelines

## Related Issues
Fixes #123
```

---

## Questions?

- Open a [Discussion](https://github.com/yourusername/pyeval/discussions)
- Check existing [Issues](https://github.com/yourusername/pyeval/issues)
- Review the [Documentation](https://yourusername.github.io/pyeval/)

Thank you for contributing! 🎉
