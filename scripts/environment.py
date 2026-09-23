"""Print a JSON environment manifest; missing GPU tools are reported, not fatal."""

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture(argv: list[str], cwd: Path = ROOT) -> dict:
    if not shutil.which(argv[0]):
        return {"available": False}
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": str(exc)}
    return {
        "available": True,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    upstream = ROOT / "third_party" / "llama.cpp"
    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "project_commit": capture(["git", "rev-parse", "--verify", "HEAD"]),
        "project_status": capture(["git", "status", "--porcelain"]),
        "submodules": capture(["git", "submodule", "status", "--recursive"]),
        "llama_commit": capture(["git", "rev-parse", "HEAD"], upstream)
        if (upstream / "CMakeLists.txt").exists()
        else {"available": False},
        "cmake": capture(["cmake", "--version"]),
        "cxx": capture(["c++", "--version"]),
        "cuda": capture(["nvcc", "--version"]),
        "nvidia": capture(["nvidia-smi"]),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
