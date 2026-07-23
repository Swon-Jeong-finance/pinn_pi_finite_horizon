# Auxiliary regression tests

The production training and post-processing scripts are in the parent
directory. These tests are kept here so they do not clutter the working
directory used for experiments.

Run the complete CPU regression suite from the extracted package root:

```bash
bash auxiliary_tests/run_tests.sh
```

The runner sets the package root on `PYTHONPATH`. Tests that require optional
packages such as PyTorch skip themselves when those packages are unavailable.
