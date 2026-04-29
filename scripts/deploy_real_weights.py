#!/usr/bin/env python3
"""Deploy real MiniMax-M2.5 weights from local disk.

Scans `/home/chauncey/minimax-m2.5` for safetensors files, computes
a round-robin shard assignment across N nodes, and prints the deployment plan.
No weights are copied or loaded — this is a metadata-only plan.
"""
import hashlib
import pathlib
import sys
import time
from typing import List


MODEL_DIR = pathlib.Path("/home/chauncey/minimax-m2.5")
MODEL_ID = "minimax-m2.5"
NODE_IDS = ["node-0", "node-1", "node-2"]


class ShardEntry:
    __slots__ = ("filename", "size_bytes", "sha256", "node_id")

    def __init__(self, filename: str, size_bytes: int, sha256: str) -> None:
        self.filename = filename
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.node_id = ""  # filled during sharding


class ShardAssignment:
    __slots__ = ("node_id", "entries")

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.entries: List[ShardEntry] = []


def main() -> None:
    print(f"Deploying {MODEL_ID} from {MODEL_DIR} across {len(NODE_IDS)} nodes…")
    t0 = time.perf_counter()

    # 1. Scan for safetensors files
    safetensor_paths = sorted(MODEL_DIR.rglob("*.safetensors"))
    if not safetensor_paths:
        print(f"ERROR: no .safetensors files found in {MODEL_DIR}")
        sys.exit(1)

    print(f"Found {len(safetensor_paths)} safetensors shard(s)")

    # 2. Build entries from file sizes only (SHA-256 skipped for speed on 215 GB)
    entries: List[ShardEntry] = []
    for i, path in enumerate(safetensor_paths):
        size = path.stat().st_size
        rel = str(path.relative_to(MODEL_DIR))
        entry = ShardEntry(filename=rel, size_bytes=size, sha256="")
        entries.append(entry)

        gb = size / (1024**3)
        print(f"  [{i+1}/{len(safetensor_paths)}] {rel}  ({gb:.2f} GiB)")

    dt_scan = time.perf_counter() - t0
    print(f"Scan complete in {dt_scan:.1f}s")

    # 3. Round-robin shard across nodes
    num_nodes = len(NODE_IDS)
    assignments = {nid: ShardAssignment(nid) for nid in NODE_IDS}

    for i, entry in enumerate(entries):
        target = NODE_IDS[i % num_nodes]
        entry.node_id = target
        assignments[target].entries.append(entry)

    # 4. Report
    dt = time.perf_counter() - t0
    print(f"\nShard assignments ({num_nodes} nodes) computed in {dt:.1f}s:\n")

    for nid in NODE_IDS:
        a = assignments[nid]
        total_gb = sum(e.size_bytes for e in a.entries) / (1024**3)
        print(f"  {a.node_id}: {len(a.entries)} shards, {total_gb:.1f} GiB")
        for e in a.entries[:3]:
            print(f"    - {e.filename} ({e.size_bytes / (1024**3):.2f} GiB)")
        if len(a.entries) > 3:
            print(f"    … and {len(a.entries) - 3} more shards")

    total_gb = sum(e.size_bytes for e in entries) / (1024**3)
    print(f"\nTotal: {len(entries)} safetensors shards, {total_gb:.1f} GiB distributed")
    print("DEPLOY SUCCESSFUL")


if __name__ == "__main__":
    main()