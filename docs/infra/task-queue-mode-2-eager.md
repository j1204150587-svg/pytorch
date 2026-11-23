# Task Queue mode 2 on an eager workload

Compare mode 1 with mode 2 on the same eager workload. Set the variable before process startup and run the same command and arguments in each leg.

```sh
env TASK_QUEUE_ENABLE=1 python run_workload.py
env TASK_QUEUE_ENABLE=2 python run_workload.py
```

The source implementation of [`Repository::ReadQueue`](../../torch_npu/csrc/core/npu/NPUQueue.cpp) briefly polls an empty queue in mode 2. Confirm the behavior in the installed wheel and keep device task order and launch count fixed. Measure enqueue completion to dequeue start, CPU utilization, device idle time and synchronized step time.

Status: the recorded validation was inconclusive because another process used the device.
