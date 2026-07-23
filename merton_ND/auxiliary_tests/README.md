# Auxiliary tests

These files are regression and interface checks used while preparing the
Merton experiment code. They are separated from the runnable experiment and
post-processing scripts.

Run the full suite from the `merton_ND` directory:

```bash
MPLCONFIGDIR=/tmp/merton_mpl \
python3 -m unittest discover \
  -s auxiliary_tests \
  -t . \
  -p 'test_*.py' \
  -q
```

Tests that require PyTorch are skipped automatically when PyTorch is not
installed.
