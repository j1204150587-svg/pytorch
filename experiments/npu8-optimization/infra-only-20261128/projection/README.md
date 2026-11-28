# 线性投影描述符

前向 `op_api::addmm` 对受守卫的只读 bias 和转置 weight 回绑实际字节地址，反向 `op_api::mm` 对 dInput 的连续 RHS 权重做相同处理。两次实测使用各自的独立库。缓存键写入独立标记，storage offset 归零，并使用当前实际视图的长度。缓存命中时所绑定的地址与未命中时构造 ACL 张量的 base、offset 采用同一规则。原 ATen 张量继续拥有底层 storage。dtype、形状、地址对齐、容量、Mode 1 和非 capture 守卫均由各自原生实现检查。[Addmm 实现](native/addmm_rebase.inc) · [Mm 实现](native_mm/mm_rebase.inc)

|算子|开关|实测 libtorch_npu.so SHA-256|
|---|---|---|
|Addmm|`NPU_DYNAMIC_ADDMM_REBASE_128`|`1e794df7fb438c9d0e0c9646a348cc9497bf850c73a88eb0b55710b3a98d3230`|
|Mm|`NPU_DYNAMIC_MM_REBASE_128`|`11d285bbf42f5852d8f02f53622f2073fecba42840ed2dfb058be1bfd535339b`|

构建分别运行 native/build.py 和 native_mm/build.py，`--study` 指向已有的 dynamic-infra-20261127 构建目录，`--output` 指向独立的新目录。三方编译规则、前置对象与既有库按 build_native.py 的收据固定。训练对照两腿加载相同实测库，开关取 0 或 1。诊断使用库自带 counter，在进程退出时读取，正式计时单独运行。[Addmm 构建](native/build.py) · [Mm 构建](native_mm/build.py) · [训练命令与哈希核验](../复现与证据.md)

