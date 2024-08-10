# Contributing to Py-FSRS

Welcome to Py-FSRS!

In this short guide, you will get a quick overview of how you can contribute to the Py-FSRS project.

## Reporting issues

If you encounter an issue with Py-FSRS and would like to report it, you'll first want to make sure you're using the latest version of Py-FSRS.

The latest version of py-fsrs can be found under [releases](https://github.com/open-spaced-repetition/py-fsrs/releases) and you can verify the version of your current installation with the following command:
```
pip show fsrs
```

Once you've confirmed your version, please report your issue in the [issues tab](https://github.com/open-spaced-repetition/py-fsrs/issues).

## Contributing code

### Local setup

**Step 1**: Start by forking this repo, then cloning it to your local machine.

**Step 2**: Create a new local branch where you will implement your changes.

### Develop

Install the `fsrs` python package locally in editable mode from the src with
```
pip install -e .
```

Now you're ready to make changes to `src/fsrs` and see your changes reflected immediately!

### Bump the version number

This project follows [semantic versioning](https://semver.org/), so please make sure to increment the version number in [pyproject.toml](pyproject.toml) when contributing new code.

### Lint, format and test

Py-FSRS uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting and uses [pytest](https://docs.pytest.org) to run its tests. In order for your contribution to be accepted, your code must pass linting/formatting checks and be able to pass the tests.

You can install these packages with
```
pip install ruff pytest
```

Lint your code with:
```
ruff check --fix
```

Format your code with:
```
ruff format
```

Run the tests with:
```
pytest
```

Additionally, you are encouraged to contribute your own tests to [tests/test_fsrs.py](tests/test_fsrs.py) to help make Py-FSRS more reliable!

### Submit a pull request

To submit a pull request, commit your local changes to your branch then push the branch to your fork. You can now open a pull request.