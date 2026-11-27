class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s17)", arg1_1: "Sym(s27)", arg2_1: "Sym(s53)", arg3_1: "bf16[s17, s27, s53]", arg4_1: "bf16[s17, s27, s53]"):
        # File: /artifacts/dynamic-infra-20261127/pr42634/eligibility/compiled_control_compat.py:50 in pointwise, code: return x * 2.0 + y
        mul: "bf16[s17, s27, s53]" = torch.ops.aten.mul.Tensor(arg3_1, 2.0);  arg3_1 = None
        add_4: "bf16[s17, s27, s53]" = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        return (add_4,)
        