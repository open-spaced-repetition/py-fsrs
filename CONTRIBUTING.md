# Contributing to Py-FSRS

## Reporting issues

If you encounter an issue with Py-FSRS and would like to report it, you'll first want to make sure you're using the latest version of Py-FSRS.

The latest version of py-fsrs can be found under [releases](https://github.com/open-spaced-repetition/py-fsrs/releases) and you can verify the version of your current installation with the following command:
```
pip show fsrs
```

Once you've confirmed your version, please report your issue in the [issues tab](https://github.com/open-spaced-repetition/py-fsrs/issues).

## Contributing code

### Set up local environment

After cloning this repo, install `fsrs` locally in editable mode along with the dev dependencies
```
pip install -e ".[dev]"
```

Now you're ready to make changes to files in the `fsrs` directory and see your changes reflected immediately.

### Pass the checks

In order for your contribution to be accepted, your code must pass the linting checks and unit tests.

Lint your code with:
```
ruff check --fix
```

Run the tests with:
```
pytest
```

Also, don't forget to format your code with:
```
ruff format
```

Additionally, you are strongly encouraged to contribute your own tests to [tests/test_fsrs.py](tests/test_fsrs.py) to help make Py-FSRS more reliable.