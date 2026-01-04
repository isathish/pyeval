# Contributing to PyEval

We love your input! We want to make contributing to PyEval as easy and transparent as possible.

## Quick Links

- 📖 [Full Contributing Guide](docs/contributing.md)
- 🐛 [Report Bug](https://github.com/yourusername/pyeval/issues/new?template=bug_report.md)
- 💡 [Request Feature](https://github.com/yourusername/pyeval/issues/new?template=feature_request.md)
- 📚 [Documentation](https://yourusername.github.io/pyeval/)

## Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/pyeval.git
cd pyeval
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check pyeval/
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new code
3. Update documentation if needed
4. Ensure tests pass: `pytest`
5. Submit your PR!

## Commit Messages

We use [Conventional Commits](https://conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
