"""Stage RecSDK under one switch in the existing dynamic eager training runner."""
import shutil
import sys
from pathlib import Path


def once(src, anchor, replacement):
    if src.count(anchor) != 1:
        raise ValueError("Unexpected model or runner source: " + anchor)
    return src.replace(anchor, replacement)


def main(ws):
    model, trainer = ws/"model.py", ws/"train_deepspeed.py"
    src = model.read_text(encoding="utf8")
    src = once(src, "import math\n", "import math\nfrom recsdk_attention import Experiment\n")
    init = "        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    src = once(src, init, init + "        self.recsdk = Experiment()\n")
    start = "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha\n"
    end = "        attn = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)\n"
    section = src[src.index(start):src.index(end,src.index(start))+len(end)]
    replacement = """        fused_output = self.recsdk(q, k, v, self.attn_alpha)
        if fused_output is not None:
            attn = fused_output.reshape(batch_size, seq_len, -1)
        else:
""" + "".join("    " + line for line in section.splitlines(keepends=True))
    src = once(src, section, replacement)
    train = trainer.read_text(encoding="utf8")
    anchor = "        end = time.perf_counter()\n"
    train = once(train, anchor, anchor + """        from recsdk_attention import receipt
        receipt(model.blocks, rank_dir, args.steps)
""")
    compile(src,str(model),"exec")
    compile(train,str(trainer),"exec")
    model.write_text(src,encoding="utf8")
    trainer.write_text(train,encoding="utf8")
    shutil.copyfile(Path(__file__).with_name("adapter.py"),ws/"recsdk_attention.py")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
