class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s27)", arg1_1: "Sym(s53)", arg2_1: "bf16[1, s27, s53]", arg3_1: "bf16[1, s27, s53]"):
        # File: /artifacts/dynamic-infra-20261127/pr42634/eligibility/compiled_control_compat.py:50 in pointwise, code: return x * 2.0 + y
        mul: "bf16[1, s27, s53]" = torch.ops.aten.mul.Tensor(arg2_1, 2.0);  arg2_1 = None
        add_3: "bf16[1, s27, s53]" = torch.ops.aten.add.Tensor(mul, arg3_1);  mul = arg3_1 = None
        return (add_3,)
        