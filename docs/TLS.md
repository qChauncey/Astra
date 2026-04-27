# Astra gRPC mTLS — Transport Security Guide

**Phase 5.1**   |   **Status: Stable (self-signed + TOFU trust model)**

Astra secures inter-node gRPC communication with mutual TLS (mTLS) using
self-signed X.509 certificates and a Trust-On-First-Use (TOFU) pinning model
— no central CA required.

---

## 1. Quick Start

```bash
# Generate certificates for every node (run once per node)
python -c "
from astra.rpc.tls import generate_self_signed_cert_bundle
bundle = generate_self_signed_cert_bundle(node_id='node-1')
cert_path, key_path = bundle.write('/etc/astra/certs')
print(f'Cert: {cert_path}')
print(f'Key:  {key_path}')
"
```

Then configure your server/client with the certificate paths:

```python
from astra.rpc.tls import TLSConfig

tls_config = TLSConfig(
    enabled=True,
    cert_path="/etc/astra/certs/node-1.cert.pem",
    key_path="/etc/astra/certs/node-1.key.pem",
    ca_cert_path="/etc/astra/certs/ca.pem",   # shared across cluster
)
```

---

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│                  Astra Node A                     │
│  ┌───────────┐        ┌──────────────────────┐  │
│  │ Self-signed│        │  InferenceServer      │  │
│  │ X.509 cert │───────▶│  (mTLS on port 50051) │  │
│  │ + RSA key  │        └──────────┬───────────┘  │
│  └───────────┘                   │               │
└──────────────────────────────────┼───────────────┘
                                   │ gRPC over TLS
                                   │ (mutual auth)
┌──────────────────────────────────┼───────────────┐
│                  Astra Node B    │                │
│  ┌───────────┐        ┌──────────▼───────────┐  │
│  │ Self-signed│        │  InferenceClient      │  │
│  │ X.509 cert │◀───────│  (mTLS to node A)     │  │
│  │ + RSA key  │        └──────────────────────┘  │
│  └───────────┘                                    │
└──────────────────────────────────────────────────┘

TOFU Trust Model:
  - First time node B sees node A's cert → pin it
  - Subsequent connections from node A MUST present the same cert
  - If cert changes → connection rejected (potential MITM)
```

**Key design decisions:**

| Mechanism | Choice | Rationale |
|---|---|---|
| Certificate type | Self-signed X.509 | No CA infrastructure; suitable for P2P |
| Key algorithm | RSA 2048-bit | Broad gRPC compatibility |
| Signature | SHA-256 | Current best practice |
| Trust model | TOFU (Trust-On-First-Use) | Pins cert on first contact; detects key rotation |
| Certificate lifetime | 365 days (default) | Configurable via `days_valid` |
| Client authentication | Required (mTLS) | Both ends must present valid certs |

---

## 3. Certificate Generation

```python
from astra.rpc.tls import generate_self_signed_cert_bundle

# Generate a self-signed cert + RSA key pair
bundle = generate_self_signed_cert_bundle(
    node_id="worker-3",
    common_name="astra.worker-3",   # optional, defaults to astra.{node_id}
    days_valid=365,
    key_size=2048,
)

# Access PEM strings directly
print(bundle.cert_pem)  # -----BEGIN CERTIFICATE-----
print(bundle.key_pem)   # -----BEGIN PRIVATE KEY-----

# Write to disk
cert_path, key_path = bundle.write("/etc/astra/certs")
# → /etc/astra/certs/worker-3.cert.pem
# → /etc/astra/certs/worker-3.key.pem
```

### Cluster-wide CA setup (optional but recommended)

For a shared trust root across all nodes:

```python
# Generate a "CA" cert (just another self-signed cert used as trust anchor)
ca_bundle = generate_self_signed_cert_bundle(node_id="astra-ca", days_valid=3650)
ca_bundle.write("/etc/astra/certs")
# Copy /etc/astra/certs/astra-ca.cert.pem to ALL nodes
```

**Important:** All nodes in the cluster must use the **same CA cert** as
`ca_cert_path` for mTLS to work.  Distribute the CA cert via ansible, scp,
or shared filesystem.

---

## 4. Server Configuration

The `InferenceServer` constructor accepts a `TLSConfig`:

```python
from astra.rpc.tls import TLSConfig
from astra.rpc.server import InferenceServer

tls = TLSConfig(
    enabled=True,
    cert_path="/etc/astra/certs/node-1.cert.pem",
    key_path="/etc/astra/certs/node-1.key.pem",
    ca_cert_path="/etc/astra/certs/astra-ca.cert.pem",
)

server = InferenceServer(
    node_id="node-1",
    layer_start=0,
    layer_end=30,
    port=50051,
    tls_config=tls,
)
server.serve()
```

Without `ca_cert_path` the server uses **server-only TLS** (encryption without
client certificate verification).  With `ca_cert_path` it requires **mutual TLS**
— clients must present a valid certificate signed by the same CA.

---

## 5. Client Configuration

The `InferenceClient` also accepts a `TLSConfig`:

```python
from astra.rpc.tls import TLSConfig
from astra.rpc.client import InferenceClient

tls = TLSConfig(
    enabled=True,
    cert_path="/etc/astra/certs/node-2.cert.pem",
    key_path="/etc/astra/certs/node-2.key.pem",
    ca_cert_path="/etc/astra/certs/astra-ca.cert.pem",
)

client = InferenceClient(
    address="node-1:50051",
    node_id="node-2",
    tls_config=tls,
)
```

---

## 6. TOFU Trust Store

For fully decentralized P2P without any shared CA, use the TOFU trust store:

```python
from astra.rpc.tls import TofuTrustStore, generate_self_signed_cert_bundle

store = TofuTrustStore()

# When you first connect to a peer, pin its certificate
peer_cert_pem = get_peer_cert_from_handshake()  # your handshake logic
store.add("peer-node-5", peer_cert_pem)

# On subsequent connections, verify the cert hasn't changed
is_trusted = store.verify("peer-node-5", peer_cert_pem)
# → True if same cert as first seen, False if changed (possible MITM)

# Persist the trust store for restarts
with open("/etc/astra/trust_store.json", "wb") as f:
    f.write(store.serialize())

# Restore after restart
with open("/etc/astra/trust_store.json", "rb") as f:
    store = TofuTrustStore.deserialize(f.read())
```

**TOFU guarantees:**
- First cert seen for a `node_id` is pinned permanently
- Any subsequent cert change for the same `node_id` → rejected
- Manual intervention required to rotate (call `store.remove(node_id)` then `store.add(node_id, new_cert)`)
- Thread-safe for concurrent access

---

## 7. Run with mock_pipeline.py

```bash
# Phase 1: Generate certs (one-time setup)
python -c "
from astra.rpc.tls import generate_self_signed_cert_bundle
for name in ['node-a', 'node-b', 'astra-ca']:
    b = generate_self_signed_cert_bundle(node_id=name)
    b.write('/tmp/astra-certs')
    print(f'{name}: done')
"

# Phase 2: Run with TLS enabled
python mock_pipeline.py --tls
```

The `--tls` flag auto-generates temporary certificates for the mock run.

---

## 8. TLS Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `False` | Enable TLS for this node |
| `cert_path` | `""` | Path to PEM-encoded X.509 certificate |
| `key_path` | `""` | Path to PEM-encoded RSA private key |
| `ca_cert_path` | `""` | Path to CA certificate for verifying peers |

### Security Notes

- **Private key protection**: Set file permissions to `600` (owner read/write only)
- **Certificate rotation**: Re-generate certs before expiry; update TOFU store
- **Key size**: 2048-bit RSA is the default; consider 4096-bit for higher security
- **No hardcoded secrets**: All paths are configurable; no keys in source code
- **In transit only**: TLS protects data in transit; use TEE for data at rest

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ssl.SSLError: [SSL] PEM lib` | Cert/key file missing or corrupt | Verify paths, check file permissions |
| `ssl.SSLError: certificate verify failed` | CA cert mismatch between client/server | Ensure all nodes use same CA cert |
| `ssl.SSLError: unknown ca` | CA cert not trusted by peer | Copy CA cert to both nodes' `ca_cert_path` |
| `grpc.RpcError: UNAVAILABLE` after enabling TLS | Client using insecure channel to TLS server | Add `tls_config` to both sides |
| TOFU verify returns `False` | Certificate was rotated | Remove old pin, add new cert |
| `ValueError: CA cert required for mTLS` | `ca_cert_path` empty when `enabled=True` | Set `ca_cert_path` to shared CA PEM file |

---

## 10. Production Checklist

- [ ] Generate unique cert+key pair for every node
- [ ] Generate a shared CA cert and distribute to all nodes
- [ ] Set `TLSConfig(enabled=True, ...)` for all servers and clients
- [ ] Set file permissions: `chmod 600` for key files, `chmod 644` for cert files
- [ ] Configure firewall to allow gRPC ports
- [ ] Test connection with `openssl s_client -connect host:port`
- [ ] Monitor certificate expiration (default 365 days)
- [ ] Set up certificate rotation procedure
- [ ] Consider using a real CA in production (e.g., cert-manager, Vault PKI)

---

**Next**: For multi-machine DHT peer discovery, see
[docs/HIVEMIND.md](./HIVEMIND.md) (Phase 5.2).