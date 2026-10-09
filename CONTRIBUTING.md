# Contributing to Titanic Classifier

Thank you for your interest in contributing to Titanic Classifier! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed and expected
- Include screenshots or animated GIFs if possible
- Include details about your environment (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Explain why this enhancement would be useful
- List similar implementations if any

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Follow the project's code style
- Include tests when adding new features
- Ensure all tests pass before submitting
- Document new features in the README
- Keep changes focused and minimal

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. Make your changes and commit
6. Run tests:
   ```bash
   pytest
   ```
7. Push and create a Pull Request

## Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep functions focused and small
- Use meaningful variable names

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

Example:
```
feat: add gradient boosting classifier
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=titanic_classifier tests/
```