# 固定形状台账与动态训练的前序材料

下列文件位于本地实验父目录 `outputs/recsys-gr-ds019-host88/dispatch-study/20261119-npu8/`。本轮沿用这些文件中的配置、调用链和历史结论。Git 上的本轮代码、数据和优化编号使用独立目录。

| 相对于实验父目录的文件 | SHA-256 |
|---|---|
| `NPU8优化台账.md` | `c1b7b7e0bee1336d581a386e6cea78fb2811298879ba1ff61cfa76be61198474` |
| `原始HSTU训练调用链.md` | `71ec2cf0cb1818908941c950d28f9d95c3d0d11434c6bae3c2b896fd8b533deb` |
| `dynamic-shapes-20261127/RESULTS.md` | `be44ed1f629f168daeab02a862dd5b372cf429f875f910eff54b38ef99e5aea1` |
| `infra-discovery-20261127/evidence/42/audit/report.md` | `dd1f22f7df3e8ebae18b3b6e1f926b754a235932b261f15c5a0a6b1dff7e5b36` |
| `pta08-queued-replay-20261125/queued-copy-replay.patch` | `794170784eef3a7883ef22abaf3fdcf4d302c56c2ce70259d2b82e599b27e1a1` |

固定负载的早期优化和动态训练 runner 分别在以下两份提交中归档。[固定负载实测代码](https://github.com/j1204150587-svg/pytorch/tree/76b73864dde200076db4751780e1f35d84595100/experiments/npu8-optimization) · [动态训练 runner 与形状](https://github.com/j1204150587-svg/pytorch/tree/04165b25f3774301451bdebbd8a816df18ece443/experiments/npu8-optimization/dynamic-shapes-20261127)
