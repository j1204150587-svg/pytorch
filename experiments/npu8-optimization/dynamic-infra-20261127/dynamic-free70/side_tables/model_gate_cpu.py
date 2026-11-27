"""Exercise the model overlay, independent leaves, tied weight, and a guarded fallback."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import torch
from stage import stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        shutil.copyfile(args.workspace / "model.py", root / "model.py")
        stage(root)
        sys.path.insert(0,str(root))
        from model import ModelConfig, MultiTableHSTU
        cfg = ModelConfig(**json.loads((args.workspace / "configs/model_many_tables.json").read_bytes()))
        models = []
        for switch in ("0","1"):
            os.environ["NPU_DYNAMIC_GROUPED_SIDE"] = switch
            torch.manual_seed(1234)
            models.append(MultiTableHSTU(cfg).to(torch.bfloat16))
        a,b = models
        assert a.state_dict().keys() == b.state_dict().keys()
        for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters(),strict=True):
            assert name == other and torch.equal(x,y)
        assert all(m.output.weight is m.item_embedding.weight for m in models)
        assert len(list(a.parameters())) == len(list(b.parameters())) == 1052
        rng = torch.Generator().manual_seed(789)
        report = []
        for index,(n,l) in enumerate(((1,19),(2,14),(3,13))):
            tokens = torch.randint(1,cfg.vocab_size,(n,l+1),generator=rng)
            groups = torch.randint(cfg.num_user_groups,(n,),generator=rng)
            ids = torch.randint(cfg.table_rows,(n,l,cfg.num_tables,4),generator=rng)
            if index == 2:
                # A real forward hook must continue to see the table lookup.
                calls = [0,0]
                def count(observed_index):
                    def hook(*_):
                        calls[observed_index] += 1
                    return hook
                for i,m in enumerate(models):
                    m.side_embeddings[12].register_forward_hook(count(i))
            outputs = []
            for m in models:
                m.zero_grad(set_to_none=True)
                logits = m(tokens[:,:-1].contiguous(),groups,ids)
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1,cfg.vocab_size),
                                                         tokens[:,1:].reshape(-1))
                loss.backward()
                outputs.append((logits.detach(),loss.detach()))
            assert torch.equal(outputs[0][0],outputs[1][0]) and torch.equal(outputs[0][1],outputs[1][1])
            for p,q in zip(a.parameters(),b.parameters(),strict=True):
                assert p.grad is not None and q.grad is not None and torch.equal(p.grad,q.grad)
            report.append(dict(shape=[n,l],loss=outputs[0][1].item(),logits_equal=True,matching_gradients=1052))
        assert calls == [1,1] and b._side_calls == 3 and b._side_hits == 2
        args.out.write_text(json.dumps(dict(passed=True,torch=str(torch.__version__),device="cpu",
                                            cases=report,qualified_hits=b._side_hits,fallbacks=1), indent=2))


if __name__ == "__main__":
    main()
