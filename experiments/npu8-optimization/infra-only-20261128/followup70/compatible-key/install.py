"""Generate the single-TU patch from the pinned PTA source."""
import difflib
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
original = (here / "pinned-input.cpp").read_bytes()
assert hashlib.sha256(original).hexdigest() == "9dc7faf6f7f3fe6a4542cc10fbbf335de521e94836f6baf9d951a2bb7e809039"
source = original.decode()
name = "torch_npu/csrc/framework/interface/EnvVariables.cpp"
line = 'REGISTER_OPTION_BOOL_FUNCTION(CheckCompatibleImpl, TORCH_NPU_USE_COMPATIBLE_IMPL, "0", "1")'
assert source.count(line) == 1
impl = """
namespace {
bool UseStableOptionKey(const char *name)
{
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

struct OptionQueryAudit {
    uint64_t queries = 0;
    uint64_t nanoseconds = 0;
    ~OptionQueryAudit()
    {
        std::fprintf(stderr, "NPU_COMPATIBLE_QUERY calls=%llu ns=%llu\\n",
            static_cast<unsigned long long>(queries), static_cast<unsigned long long>(nanoseconds));
    }
};
}

bool CheckCompatibleImpl()
{
    static const bool stableKey = UseStableOptionKey("TORCH_NPU_STABLE_COMPATIBLE_KEY");
    static const bool audit = UseStableOptionKey("TORCH_NPU_COMPATIBLE_QUERY_AUDIT");
    auto query = [] {
        if (stableKey) {
            static const std::string key("TORCH_NPU_USE_COMPATIBLE_IMPL");
            return c10_npu::option::GetOption(key).value_or("0") == "1";
        }
        return c10_npu::option::GetOption("TORCH_NPU_USE_COMPATIBLE_IMPL").value_or("0") == "1";
    };
    if (!audit) {
        return query();
    }
    static thread_local OptionQueryAudit stats;
    const auto start = std::chrono::steady_clock::now();
    const bool result = query();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    ++stats.queries;
    stats.nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    return result;
}
""".strip()
patched = "#include <chrono>\n#include <cstdio>\n#include <cstring>\n" + source.replace(line, impl)
patch = "".join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True), fromfile="a/"+name, tofile="b/"+name))
(here/"compatible-key.patch").write_text(patch)
