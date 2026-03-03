# Releasing

## First release

```sh
# Build (generate dist/*.whl and dist/*.tar.gz)
uv build

# Publish in TestPyPI first
uv publish --publish-url https://test.pypi.org/legacy/ --token TU_TOKEN_TESTPYPI

# Verify installation from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ howmuch

# Publish in real PyPI
uv publish --token TU_TOKEN_PYPI
```