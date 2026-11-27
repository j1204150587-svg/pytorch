"""Keep each step's generation, copies, and lifetimes inside its original input scope."""
from pathlib import Path
import shutil
import sys
import textwrap


def stage(workspace):
    entry = workspace / "train_deepspeed.py"
    source = entry.read_text()
    marker = '                with record_function("input_generation_and_h2d"):\n'
    next_marker = '                with record_function("hstu_forward_and_loss"):\n'
    if source.count(marker) != 1 or source.count(next_marker) != 1:
        raise ValueError("expected one input scope and one forward scope")
    prefix, tail = source.split(marker)
    stock, suffix = tail.split(next_marker)
    if stock.count("torch.randint(") != 3 or stock.count(".to(device)") != 4:
        raise ValueError("unexpected original ID generation")
    init = '    metrics = []\n'
    record = '        end = time.perf_counter()\n'
    if prefix.count(init) != 1 or suffix.count(record) != 1:
        raise ValueError("unexpected training timer or metrics initialization")
    prefix = prefix.replace(init, '    from packed_inputs import inputs as packed_inputs\n'
        '    packed_transfer = os.environ.get("NPU_DYNAMIC_INPUT_TRANSFER") == "1"\n'
        '    input_count = 0\n' + init)
    new = ('                    if packed_transfer:\n'
           '                        items, targets, groups, features = packed_inputs(\n'
           '                            config, batch_size, sequence_length, generator, device)\n'
           '                        input_count += 1\n'
           '                    else:\n')
    suffix = suffix.replace(record, record + '        (rank_dir / "input_transfer.json").write_text(\n'
          '            json.dumps({"enabled": packed_transfer, "packed_batches": input_count}))\n')
    source = prefix + marker + new + textwrap.indent(stock, "    ") + next_marker + suffix
    compile(source, str(entry), "exec")
    entry.write_text(source)
    shutil.copyfile(Path(__file__).with_name("packed_inputs.py"), workspace / "packed_inputs.py")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
