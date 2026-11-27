"""Submit one H2D for a dense batch of four immutable int64 input fields."""
import torch


def inputs(config, batch, length, generator, device):
    field_shapes = ((batch, length, config.num_tables, config.features_per_table),
                    (batch, length), (batch, length), (batch,))
    n = batch * length
    counts = [n * config.num_tables * config.features_per_table, n, n, batch]
    host = torch.empty(sum(counts), dtype=torch.int64, device="cpu")
    feature_ids, items, targets, groups = [
        part.view(shape) for part, shape in zip(host.split(counts), field_shapes)
    ]
    # The generator is advanced in exactly the same order as the stock path.
    tokens = torch.randint(1, config.vocab_size, (batch, length + 1), generator=generator)
    items.copy_(tokens[:, :-1])
    targets.copy_(tokens[:, 1:])
    torch.randint(0, config.num_user_groups, (batch,), generator=generator, out=groups)
    torch.randint(0, config.table_rows, field_shapes[0], generator=generator, out=feature_ids)
    device_batch = host.to(device)
    features, items, targets, groups = [
        part.view(shape) for part, shape in zip(device_batch.split(counts), field_shapes)
    ]
    return items, targets, groups, features
