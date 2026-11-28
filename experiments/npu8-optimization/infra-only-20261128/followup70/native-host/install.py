"""Apply the PTA-only low-rank view change to a pinned isolated source tree."""
from pathlib import Path
import sys

SOURCE = "torch_npu/csrc/aten/common/TensorShape.cpp"


def apply(source):
    include = "#include <algorithm>"
    anchor = "    auto inferred_size = at::infer_size(size, self.numel());"
    if source.count(anchor) != 1 or source.count(include) != 1:
        raise ValueError("unexpected TensorShape source")
    return source.replace(include, include + "\n" + Path(__file__).with_name("config.inc").read_text()).replace(
        anchor, '''    const bool eligible = size.size() <= c10::kDimVectorStaticSize && !self.is_quantized();
    const bool active = viewInlineDimsEnabled() && eligible;
    if (viewInlineDimsAudit()) {
        recordInlineDims(active, eligible);
    }
    if (active) {
        const auto inferred_size = at::infer_size_dv(size, self.numel());
        const auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
        TORCH_CHECK(
            stride.has_value(),
            "view size is "
            "not compatible with input tensor's size and stride (at least one dimension"
            " spans across two contiguous subspaces). Use .reshape(...) instead.", OPS_ERROR(ErrCode::PARAM));
        return alias_with_sizes_and_strides_npu(self, inferred_size, *stride);
    }
''' + anchor)


if __name__ == "__main__":
    target = Path(sys.argv[1]) / SOURCE
    target.write_text(apply(target.read_text()))
