# Releasing

The release process is fully automated via GitHub Actions. There are two workflows:

- **CI** (`.github/workflows/ci.yml`): runs on every push to `main` and on pull requests. Executes lint and tests across Python 3.11, 3.12 and 3.13.
- **Release** (`.github/workflows/release.yml`): triggered by pushing a `v*.*.*` tag. Builds, publishes to TestPyPI, smoke-tests the install, publishes to PyPI, and creates a GitHub Release.

## Prerequisites (one-time setup)

### 1. Configure PyPI Trusted Publishing (OIDC)

Trusted Publishing lets GitHub Actions publish to PyPI without storing tokens as secrets.

**On TestPyPI** (`https://test.pypi.org/manage/account/publishing/`):
- Publisher: GitHub Actions
- Owner: `rmescandon`
- Repository: `modelcost`
- Workflow: `release.yml`
- Environment: `testpypi`

**On PyPI** (`https://pypi.org/manage/account/publishing/`):
- Same settings but environment: `pypi`

### 2. Create GitHub Environments

In the repository settings (`Settings > Environments`), create two environments:

| Environment | Protection rules (recommended) |
|---|---|
| `testpypi` | none |
| `pypi` | Required reviewer (manual approval before publishing to production) |

---

## Release workflow

### 1. Bump the version

Edit `pyproject.toml` and update the `version` field:

```toml
[project]
version = "0.2.0"
```

### 2. Commit and push to main

```sh
# 1. Bump version en pyproject.toml
# 2. Commit + push
git tag 0.2.0
git push origin 0.2.0
```

### 3. Tag the release

The tag must match the version in `pyproject.toml` exactly:

```sh
git tag 0.2.0
git push origin 0.2.0
```

### 4. Automated pipeline

Pushing the tag triggers the release workflow, which runs these jobs in order:

```
validate-tag → build → publish-testpypi → smoke-test → publish-pypi → github-release
```

| Job | What it does |
|---|---|
| `validate-tag` | Asserts the tag version matches `pyproject.toml` |
| `build` | Runs `uv build`, produces wheel + sdist |
| `publish-testpypi` | Publishes to TestPyPI via OIDC (no token needed) |
| `smoke-test` | Installs from TestPyPI and runs `modelcost --help` |
| `publish-pypi` | Publishes to PyPI via OIDC (waits for `pypi` environment approval if configured) |
| `github-release` | Creates a GitHub Release with auto-generated notes and attaches the build artifacts |

### 5. Verify

Once the pipeline finishes, verify the release:

```sh
pip install modelcost==0.2.0
modelcost --help
```
