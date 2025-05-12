# step-perf

To run the tests that use Ramulator:
```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 **cargo test --package step-perf --lib -- test::<test_file_name>::test::<name_of_the_test_fn> --exact --show-output
```
<br/>

Example:
```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 cargo test --package step-perf --lib -- ramulator::ramulator_context::test::ramulator_e2e_small --exact --show-output 
```
<br/>

To log data with MongoDB:
* Running MongoDB in foreground
    ```bash
    sudo mongod --config /etc/mongod.conf
    ```