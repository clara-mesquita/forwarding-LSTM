"""Fetch pinned upstream implementations, preserving upstream licenses."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "benchmark_sources.json").read_text(encoding="utf-8"))


def main():
    for name, source in SOURCES.items():
        target = ROOT / "vendor" / name
        if not target.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            source["url"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "sparse-checkout", "init", "--cone"], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", source["commit"]], check=True)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", *source["paths"]], check=True)
        commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if commit != source["commit"]:
            raise RuntimeError(f"{name}: expected {source['commit']}, found {commit}; use a separate checkout.")
        changes = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if changes.strip():
            raise RuntimeError(f"{name}: checkout has local changes; refusing to overwrite them.")
        print(f"{name}: {commit}")


if __name__ == "__main__":
    main()
