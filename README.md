# step-perf

```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 cargo test --package step-perf --lib -- ramulator::ramulator_context::test::ramulator_e2e_small --exact --show-output 
```


Template for future use

```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 **cargo test --package step-perf --lib -- test::<test_file_name>::test::<name_of_the_test_fn> --exact --show-output
```