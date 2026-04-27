# Astra + Hivemind — Multi-Machine DHT Deployment

**Phase 5.2**   |   **Status: Stable (graceful degradation when hivemind is not installed)**

Astra uses [hivemind](https://github.com/learning-at-home/hivemind) for real P2P distributed hash
table (DHT) peer discovery across machines.  When hivemind is not installed the framework
automatically falls back to the in-memory `AstraDHT` store — no code changes required.

---

## 1. Quick Start

```bash
# Install hivemind (optional Astradependency)
pip install hivemind>=1.1.0

# Verify
python -c "from astra.network.hivemind_bridge import is_hivemind_available; print(is_hivemind_available())"
# → True
```

### Launch a bootstrap node

```python
from astra.network.hivemind_bridge import create_dht

dht = create_dht(
    node_id="bootstrap",
    use_hivemind=True,
    host_addr="0.0.0.0",
    port=1337,
)
# This node is now reachable via multiaddr: /ip4/<host>/tcp/1337/p2p/<peer-id>
print(dht.node_id)
```

### Launch a peer that connects to the bootstrap

```python
dht = create_dht(
    node_id="worker-1",
    use_hivemind=True,
    initial_peers=[
        "/ip4/192.168.1.10/tcp/1337/p2p/QmBootstrapPeerId"
    ],
    port=1338,
)
```

---

## 2. Architecture

```
┌───────────────┐     ┌──────────────┐
│  Worker Node  │────▶│  Bootstrap   │  (initial rendezvous)
│  (port 1338)  │     │  (port 1337) │
└──────┬────────┘     └──────┬───────┘
       │                     │
       │    hivemind DHT     │
       │  (Kademlia P2P)     │
       │                     │
┌──────▼────────┐     ┌──────▼───────┐
│  Worker Node  │◀───▶│  Worker Node │
│  (port 1339)  │     │  (port 1340) │
└───────────────┘     └──────────────┘
```

Key properties of the DHT layer:
- **Decentralized**: no master; any node can bootstrap others.
- **Self-healing**: expired announcements are automatically evicted.
- **NAT traversal**: hivemind includes relay and auto-relay support out of the box.
- **Graceful fallback**: if hivemind is not installed, `create_dht()` returns an
  `AstraDHT` instance that works for local/LAN simulation without any imports.

---

## 3. Backend Selection

Use the `hivemind_bridge.create_dht()` factory — it picks the right backend
automatically:

| `use_hivemind` | hivemind installed | Result |
|----------------|---------------------|--------|
| `False` (default) | any | `AstraDHT` (in-memory, zero-config) |
| `True` | Yes | `HivemindDHT` (real P2P) |
| `True` | No | `AstraDHT` + info log (`"hivemind not installed"`) |

API compatibility: **all methods on `AstraDHT` are also available on `HivemindDHT`**
(`announce`, `get_all_peers`, `get_peer`, `subscribe_peers`, `store`, `fetch`,
`revoke`).  The orchestrator (`PipelineOrchestrator`) works with either backend
unchanged.

---

## 4. Running a Multi-Node Pipeline

### 4.1 Using `mock_pipeline.py`

```bash
# Node 1: layers 0–9, bootstrap DHT on port 1337
python mock_pipeline.py --node-id node-1 --layer-start 0 --layer-end 10 \
    --port 50051 --hivemind --dht-port 1337

# Node 2: layers 10–19, connect to node-1's DHT
python mock_pipeline.py --node-id node-2 --layer-start 10 --layer-end 20 \
    --port 50052 --hivemind --dht-port 1338 \
    --peers /ip4/127.0.0.1/tcp/1337/p2p/<NODE-1-PEER-ID>
```

### 4.2 Using the API directly

```python
from astra.network.hivemind_bridge import create_dht
from astra.network.dht import DHTNodeRecord

dht = create_dht(node_id="node-1", use_hivemind=True, port=1337)

record = DHTNodeRecord(
    node_id="node-1",
    address="localhost:50051",
    layer_start=0,
    layer_end=10,
    expert_shards=list(range(4)),
    geo_region="us-west",
)
dht.announce(record, ttl=120)

# On another machine:
dht2 = create_dht(
    node_id="node-2",
    use_hivemind=True,
    initial_peers=["/ip4/<node-1-ip>/tcp/1337/p2p/<peer-id>"],
    port=1338,
)
peers = dht2.get_all_peers()
# peers now includes node-1
```

---

## 5. Configuration Reference

| Flag / Param | Default | Description |
|---|---|---|
| `use_hivemind` | `False` | Enable real P2P DHT (requires `pip install hivemind`) |
| `initial_peers` | `[]` | Multiaddr strings of bootstrap / known peers |
| `host_addr` | `"0.0.0.0"` | Interface to bind the local DHT listener |
| `port` | `4242` | TCP port for the local DHT listener |
| `heartbeat_interval` | `20.0` | Seconds between re-announcements |

---

## 6. Firewall & NAT Notes

- Each hivemind DHT node opens a **TCP listener** on the configured `port`.
- All peers must be able to reach at least **one bootstrap node** on startup.
- After bootstrapping, the DHT is fully mesh-connected — subsequent communication
  is P2P and does not require the bootstrap to stay up.
- For NATed environments, enable `use_relay=True` and `use_auto_relay=True`
  (both on by default in `HivemindDHT`).
- The bootstrap's multiaddr must use the **externally reachable IP** (not
  `127.0.0.1` for cross-machine setups).

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `create_dht(use_hivemind=True)` returns `AstraDHT` | hivemind not installed | `pip install hivemind>=1.1.0` |
| `get_all_peers()` returns empty | Bootstrap unreachable | Check firewall, verify multiaddr IP |
| `get_all_peers()` returns stale entries | TTL expired | Increase `ttl` in `announce()`, verify heartbeat |
| hivemind fails at import | incompatible Python / libp2p | hivemind ≥1.1 supports Python 3.10–3.12 |
| "hivemind DHT creation failed" log | Port conflict | Change `port` parameter |

---

## 8. Production Checklist

- [ ] Install hivemind on all nodes: `pip install hivemind>=1.1.0`
- [ ] Deploy at least one stable **bootstrap node** with a known IP
- [ ] Configure firewall to allow TCP on DHT ports
- [ ] Use `use_hivemind=True` in `create_dht()` on all nodes
- [ ] Set `initial_peers` to the bootstrap's multiaddr on each worker
- [ ] Verify peer discovery with `get_all_peers()`
- [ ] Monitor re-announce heartbeat (default 20 s)
- [ ] Consider using `torch.distributed` rendezvous as a fallback for GPU clusters

---

**Next**: For transport-layer security between inference nodes, see
[docs/TLS.md](./TLS.md) (Phase 5.1).