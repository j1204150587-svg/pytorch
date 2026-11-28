# CANN 显式流的 Runtime 查询成本

原训练第 4、5 步（[1,19]、[2,14]）的消费者分别执行 9,892、9,893 次 `aclrtGetStreamAttribute`，服务时间 3.234、3.095 ms。与同一 dequeue 后继空档的名义交集为 2.351、1.451 ms。候选针对其中的非空输入流查找。触发时，在 `ApiErrorDecorator::GetStreamCacheOpInfoSwitch` 显式流直接沿用该指针，空流仍进原 `Runtime::GetCurStream`。流与输出指针的检查、错误路径、实际属性读取和所有 setter 均保留。证伪条件是局部可省时间乘实测频次后，远小于训练波动。[整段属性 trace 分解](../windows.json)

### 从训练容器核对函数

实装 `libnnopbase.so` 的 `op::internal::GetCacheOpInfoSwitch` 与 `NnopbaseReportCacheOpInfo` 在两个对应调用点都检查流非空，并传常量 4。实装头文件将 4 定义为 `ACL_STREAM_ATTR_CACHE_OP_INFO`。实装 ACL wrapper 下传 `rtsStreamGetAttribute`。训练 CPU 采样确认消费者载入 `libruntime_v100.so`。这两个 libnnopbase 调用点不能给每个 trace 标记分别指认上游调用点；下文按所有属性查询均合格作乐观投影。[实装与公开构建对照](layout.txt) · [只读复核命令](layout.sh)

v9.1.0 公开构建（commit `0b4af8a31c75a1f1f43638a8f75b9e3c6182e031`）的 ApiImpl 与 ApiErrorDecorator 对象都把 GetStreamCacheOpInfoSwitch 放在 vtable 起始偏移 0xf80。通过两个对应的 type_info 名称及重定位，实装映射至相同槽。装饰器读取 Runtime 单例指针并调用 GetCurStream，后者对非空指针返回原值；出入函数还保存寄存器和验证 stack canary。装饰器对所得流、输出指针均作空指针检查，再经虚表转发。实装 ApiImpl 的四条机器指令与公开构建逐字节相同，从真实 stream 的对应成员取值、赋给调用者并返回成功。公开源码的空流路径还需要查芯片特性及当前上下文，这条路径未纳入方案。[公开源码](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/core/src/api_impl/api_error.cc#L5922)

### 无卡 CPU 局部计时

用同一镜像的独立 runc 容器，在容器内检查不存在 NPU 设备。A 的 ARM 汇编复现实装非空路径的 Runtime 指针读取、一次非内联 GetCurStream 调用及寄存器、canary 开销，B 的匹配函数直接通过非空检查返回同一指针。两腿有相同外层函数帧，每次交替传入两个有效的 CPU 标记地址，仅在无须解引用 Stream 的分支计时。此微测没有运行 CANN 设备 API、全量装饰器或真实 Stream 的 setter。按每个 CPU 51 轮，轮内 ABBA、BAAB 交替，分 4,096、16,384、65,536 次三个批长。以两 A 批平均减两 B 批平均，取各轮配对差的中位数。[微测代码](bench.c) · [非空路径](lookup.S) · [实测反汇编](bench.asm) · [原始数据 CPU 100](cpu-100.csv) · [CPU 230](cpu-230.csv)

| CPU | 65,536 次批的 A / B ns/次，中位 | 配对节省 ns/次（p05 / p50 / p95） | 按每步约 9,893 次折算 |
|---|---:|---:|---:|
| 100 | 2.632 / 2.015 | 0.548 / 0.609 / 0.765 | 0.0060 ms/步 |
| 230 | 2.656 / 2.015 | 0.577 / 0.640 / 0.921 | 0.0063 ms/步 |

两个 CPU 在不同物理包。全六个配置的中位节省为 0.603–0.642 ns/次；最大中位节省投影约 0.0064 ms/步。这里测的是热点数据驻留时的局部差值；没有观测训练内每次 Runtime 查找，不能给训练冷缓存、调度长尾或所有 ACL 子阶段设置绝对上界。该候选的局部量级不足以解释已有 1–2 ms 的整 API 空档交集，也远小于实测训练交叉序波动，归「尝试无优化路径」。未修改训练镜像或提交完整训练 A/B。代码与实机参数可按下述命令复核。[分位与投影数据](timing.json) · [复算脚本](summarize.py) · [原始执行命令](run-command.txt)
