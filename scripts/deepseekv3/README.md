moe_v0 implements the parallelization strategy.
1. Create .npy files for weights and input feature
2. Create .npy file for index tensor that controls the expert workload pattern
3. Instantiate an MoE instance from the weights and config

```
export PYTHONPATH="/scratch/zgh23/step-perf/scripts:$PYTHONPATH"
```