# Disabling Task Queue with graph replay

Compare mode 1 with mode 0 using the same graph capture and replay configuration. Run the same command and arguments in each leg.

```sh
env TASK_QUEUE_ENABLE=1 python run_workload.py
env TASK_QUEUE_ENABLE=0 python run_workload.py
```

The wheel used in the comparison accepted modes 0 and 1 during graph capture and rejected mode 2. Verify the installed wheel when testing another version. Keep graph capture, data and run settings identical in both legs.

Status: the recorded graph workload showed higher device idle time and longer synchronized steps with mode 0.
