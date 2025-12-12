# Setup Notes

## Initial Setup Completed

Virtual environment created at `.venv` and all dependencies installed successfully.

### Python Version
- Python 3.14.0

### Known Issues

#### pandas_ta Compatibility
`pandas_ta` requires `numba==0.61.2` which does not support Python 3.14+. This package was skipped during installation. If you need `pandas_ta`, consider:
- Using Python 3.13 or earlier, OR
- Waiting for numba to release Python 3.14 support

All other packages from requirements.txt were installed successfully.

### Configuration
- `.env` file created from `.env.example`
- Remember to update `.env` with your actual credentials before running the application

### Verification
Run tests with: `.venv\Scripts\pytest tests\`
