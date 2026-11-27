"""Wait for the shared card while other accepted pairs finish."""
from pathlib import Path
import sys

def main(path):
    code = path.read_text()
    begin = code.index("    deadline = time.monotonic() + 900", code.index("def wait_for_card_lock():"))
    end = code.index("\n\ndef ", begin)
    code = code[:begin] + """    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
""" + code[end:]
    compile(code, str(path), "exec")
    path.write_text(code)

if __name__ == "__main__":
    main(Path(sys.argv[1]))

