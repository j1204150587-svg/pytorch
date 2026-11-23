# Process CPU affinity on Linux

Compare the same workload with its inherited CPU affinity and with a restricted CPU list. The restriction is applied to the launched process and inherited by its children. Set `CPU_LIST` to eight allowed CPUs from one NUMA node, selected for low load on the system under test.

```sh
python run_workload.py
taskset --cpu-list "${CPU_LIST:?Set CPU_LIST to an allowed CPU set}" python run_workload.py
```

Use `lscpu -e=CPU,NODE,ONLINE` and the process's effective affinity to verify the selection. Confirm device locality separately if the PCI device exposes a NUMA node. Keep Task Queue settings and workload execution identical in both legs. Compare device idle time and synchronized step time.

Status: two profiles showed less device idle time with the restriction, while synchronized steps became slower in the recorded comparison.
