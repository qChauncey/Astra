# Astra TEE (Trusted Execution Environment) Deployment Guide

> **Phase 4 — TEE Integration**
>
> This document describes how to deploy Astra inference inside hardware-isolated
> Trusted Execution Environments (Intel SGX via Gramine, AMD SEV-SNP).
> It covers hardware requirements, toolchain setup, attestation verification,
> and operational best practices.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Intel SGX via Gramine](#intel-sgx-via-gramine)
3. [AMD SEV-SNP](#amd-sev-snp)
4. [Attestation & Verification](#attestation--verification)
5. [Sealing Model Weights](#sealing-model-weights)
6. [Production Hardening Checklist](#production-hardening-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Astra's TEE layer provides an abstract interface (`astra.tee.TEEBackend`) with two
concrete backends:

| Backend | Technology | Isolation Boundary | Hardware Required |
|---------|-----------|-------------------|-------------------|
| `GramineBackend` | Intel SGX | Per-process enclave | 6th-gen Core+ with SGX |
| `SevBackend` | AMD SEV-SNP | Full VM encryption | EPYC 7002 "Rome"+ |

Both backends implement the same five primitives:

```
status()  → TEEStatus       # Hardware detection
attest()  → AttestationReport  # Cryptographic proof of identity
seal()    → SealedBlob      # Encrypt data to TEE
unseal()  → bytes           # Decrypt inside same TEE
get_quote() → bytes         # Raw attestation quote
```

### How Astra Uses TEE

```
┌──────────────────────────────────────────┐
│              Astra Inference              │
│  ┌────────────┐  ┌────────────────────┐  │
│  │ Heterogen. │  │  Differential       │  │
│  │ Engine     │  │  Privacy Injector   │  │
│  └─────┬──────┘  └────────┬───────────┘  │
│        │                  │               │
│  ┌─────▼──────────────────▼───────────┐  │
│  │         TEE Abstraction Layer      │  │
│  │  ┌──────────┐  ┌──────────────┐    │  │
│  │  │ Gramine  │  │   AMD SEV    │    │  │
│  │  │ (SGX)    │  │   (SEV-SNP)  │    │  │
│  │  └──────────┘  └──────────────┘    │  │
│  └────────────────────────────────────┘  │
│              │              │             │
│        ┌─────▼─────┐  ┌────▼────────┐    │
│        │ Intel CPU │  │ AMD EPYC    │    │
│        │ SGX Encl  │  │ SEV-Enc VM  │    │
│        └───────────┘  └─────────────┘    │
└──────────────────────────────────────────┘
```

---

## Intel SGX via Gramine

### Hardware Requirements

- **CPU**: Intel 6th-gen Core (Skylake) or newer
- **Chipset**: Must support SGX (not all SKUs do — check `ark.intel.com`)
- **BIOS**: SGX set to **"Software Controlled"** or **"Enabled"**
- **RAM**: Enclave Page Cache (EPC) allocated in BIOS (128 MB for dev, 512 MB+ for production)
- **OS**: Linux with SGX driver (kernel ≥ 5.11 has in-kernel driver)

### Check SGX Availability

```bash
# Check CPUID for SGX flag
grep -o 'sgx' /proc/cpuinfo | head -1

# Check SGX device nodes
ls -la /dev/sgx_enclave /dev/sgx_provision

# Check EPC size
dmesg | grep -i "sgx.*epc"
```

### Install Gramine

```bash
# Ubuntu 22.04+
sudo apt-get update
sudo apt-get install -y gramine

# Verify installation
gramine-sgx --version
```

### Install Intel SGX SDK/PSW (for Quote Generation)

```bash
# Add Intel SGX repository
echo "deb [trusted=yes arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/intel-sgx.list
sudo apt-get update

# Install SGX SDK and PSW
sudo apt-get install -y sgx-aesm-service libsgx-quote-ex libsgx-dcap-ql
```

### Generate the Gramine Manifest

```python
from astra.tee.gramine import GramineBackend, GramineConfig

config = GramineConfig(
    build_dir="/opt/astra/enclave",
    python_path="/usr/bin/python3",
    enclave_size="8G",
    thread_count=8,
    sgx_debug=False,  # MUST be False in production
)
sgx = GramineBackend(config=config)
sgx.generate_manifest("/opt/astra/enclave/astra.manifest.template")

# Build the signed enclave
# gramine-sgx-sign \
#   --manifest /opt/astra/enclave/astra.manifest.template \
#   --output /opt/astra/enclave/astra.manifest.sgx
```

### Run Astra Inside SGX

```bash
cd /opt/astra
gramine-sgx ./astra-runner.py
```

The Python program runs unchanged — Gramine intercepts all system calls
and enforces SGX memory protection transparently.

### SGX Debug vs. Production

| Setting | `sgx_debug` | Use Case |
|---------|-------------|----------|
| **Debug** | `True` | Development, debugging with GDB, EDBGWR allowed |
| **Production** | `False` | Release enclave, no debugger attach, full attestation |

**CRITICAL**: Never deploy `sgx_debug = True` in production. A debug enclave
allows an attacker to read all enclave memory!

---

## AMD SEV-SNP

### Hardware Requirements

- **CPU**: AMD EPYC 7002 "Rome" (SEV), 7003 "Milan" (SEV-SNP), or 9004 "Genoa"
- **BIOS**: SEV and SEV-SNP enabled (may be called "Secure Memory Encryption" or "SME")
- **OS**: Linux kernel ≥ 6.0 for full SNP host/guest support
- **Hypervisor**: QEMU ≥ 7.0 or KVM with SEV-SNP patches

### Check SEV Availability

```bash
# Check KVM SEV parameters
cat /sys/module/kvm_amd/parameters/sev       # Should be "1" or "Y"
cat /sys/module/kvm_amd/parameters/sev_es    # SEV-ES support
cat /sys/module/kvm_amd/parameters/sev_snp   # SEV-SNP support

# Verify SEV device
ls -la /dev/sev

# Check SEV CPU capabilities
cat /sys/devices/system/cpu/caps/sev
```

### Launch a SEV-SNP Protected VM

```bash
# Example QEMU command line (simplified)
qemu-system-x86_64 \
  -machine q35,confidential-guest-support=sev0 \
  -object sev-snp-guest,id=sev0,cbitpos=51,reduced-phys-bits=1 \
  -cpu EPYC-Milan \
  -m 8G \
  -drive file=/opt/astra/images/astra-sev.qcow2,format=qcow2 \
  ...
```

### Inside the SEV VM

```python
from astra.tee import register_backend
from astra.tee.amd_sev import SevBackend

sev = SevBackend()
register_backend("sev", sev)

print(sev.status())           # TEEStatus.AVAILABLE
print(sev.is_guest)           # True
print(sev.measurement[:32])   # First 32 hex chars of launch measurement
```

### SEV-SNP Attestation Flow

```
1. Guest requests attestation report from PSP via /dev/sev-guest
2. PSP generates REPORT signed with VCEK (Versioned Chip Endorsement Key)
3. Guest sends REPORT to remote verifier
4. Verifier queries AMD KDS (Key Distribution Service) for VCEK certificate
5. Verifier validates:
   - REPORT signature against VCEK
   - MEASUREMENT matches known-good hash
   - Platform TCB version is acceptable
6. Verifier returns OK → secrets may be provisioned
```

---

## Attestation & Verification

### SGX Attestation Code

```python
from astra.tee import register_backend, get_tee_backend
from astra.tee.gramine import GramineBackend

sgx = GramineBackend()
register_backend("sgx", sgx)

tee = get_tee_backend("sgx")
if tee and tee.status() == TEEStatus.AVAILABLE:
    report = tee.attest()
    print(f"MRENCLAVE: {report.measurement}")
    print(f"Valid: {report.is_valid}")
    print(f"Quote type: {report.details['quote_type']}")

    # Verify against known-good measurement
    EXPECTED_MRENCLAVE = "a1b2c3d4..."  # Pre-computed hash
    assert report.measurement == EXPECTED_MRENCLAVE
```

### SEV Attestation Code

```python
from astra.tee.amd_sev import SevBackend

sev = SevBackend()
if sev.status() == TEEStatus.AVAILABLE:
    report = sev.attest()
    print(f"SEV Measurement: {report.measurement}")
    print(f"SEV Type: {report.details['sev_type']}")

    # Verify measurement
    EXPECTED_MEASUREMENT = "e5f6a7b8..."  # From OVMF hash
    assert report.measurement == EXPECTED_MEASUREMENT
```

### Remote Attestation Service (Future)

Phase 5+ will add a REST endpoint for remote attestation:

```
POST /astra/v1/attest
Response: {
  "tee_type": "sgx",
  "measurement": "a1b2c3...",
  "quote": "base64...",
  "timestamp": 1719000000
}
```

---

## Sealing Model Weights

Sealing encrypts model weights so they can only be decrypted inside the
same TEE instance (same MRENCLAVE for SGX, same MEASUREMENT for SEV).

```python
from astra.tee import get_tee_backend

tee = get_tee_backend("sgx")

# Seal model weights at rest
with open("model.safetensors", "rb") as f:
    model_bytes = f.read()

sealed = tee.seal(model_bytes)
with open("model.sealed", "wb") as f:
    f.write(sealed.ciphertext)

# Unseal inside the enclave at runtime
with open("model.sealed", "rb") as f:
    blob = SealedBlob(ciphertext=f.read(), tee_type="sgx")

model_bytes = tee.unseal(blob)
# model_bytes is now only accessible inside the enclave
```

### Security Properties

| Property | SGX | SEV-SNP |
|----------|-----|---------|
| Memory encryption | AES-GCM with EPC key | AES-XEX with VM-specific key |
| Integrity protection | Hash-based (EPC page) | Reverse Map Table (RMP) |
| Sealing key binding | MRENCLAVE (code identity) | MEASUREMENT (VM identity) |
| Key hierarchy | CPU → Enclave → Sealing | CPU → VM → Sealing |
| Side-channel mitigation | Intel microcode updates | AMD SNP-specific hardening |

---

## Production Hardening Checklist

### Pre-Deployment

- [ ] **SGX**: Verify `sgx_debug = False` in Gramine manifest
- [ ] **SEV**: Confirm SEV-SNP is active (not plain SEV)
- [ ] **Measurements**: Pre-compute expected MRENCLAVE / MEASUREMENT
- [ ] **Attestation**: Verify attestation works from a remote machine
- [ ] **Sealing**: Test seal/unseal roundtrip with production model weights
- [ ] **TLS**: Use TLS 1.3 for all network connections in/out of TEE
- [ ] **Logging**: Ensure logs never leak plaintext model data

### Runtime

- [ ] **Monitoring**: Alert if TEE status changes from AVAILABLE
- [ ] **Secrets**: Never log or export sealing keys
- [ ] **Updates**: Keep Intel SGX PSW / AMD SEV firmware current
- [ ] **Quotes**: Validate quotes against Intel IAS / AMD KDS before
  provisioning secrets
- [ ] **Network**: Restrict TEE VM to minimal necessary network access

### Post-Deployment

- [ ] **Audit Logs**: Store attestation reports for compliance
- [ ] **Rotate Keys**: Periodically re-seal weights with fresh keys
- [ ] **Penetration Test**: Engage a third-party to test TEE isolation

---

## Troubleshooting

### SGX: "sgx device not found"

```bash
# Check BIOS settings — SGX must be "Enabled" or "Software Controlled"
# Check kernel has SGX driver:
lsmod | grep sgx

# Install SGX driver if missing:
sudo apt-get install linux-modules-extra-$(uname -r)
```

### SGX: "enclave_size too large"

Reduce `sgx.enclave_size` in the manifest. The EPC is a shared system
resource with a hard ceiling (e.g., 256 MB on desktop, 512 GB on Xeon).

### SEV: "/dev/sev not found"

```bash
# Load KVM AMD module with SEV support
sudo modprobe kvm_amd sev=1 sev_es=1 sev_snp=1

# Verify
cat /sys/module/kvm_amd/parameters/sev
```

### SEV: "Cannot launch SEV-SNP guest"

Ensure QEMU version ≥ 7.2 and kernel ≥ 6.0. Check `dmesg` for SEV-SNP
initialization messages.

### Attestation Fails: Measurement Mismatch

- **SGX**: Rebuild the enclave with identical code and configuration.
  Even a single byte difference in the Python script changes MRENCLAVE.
- **SEV**: Rebuild the OVMF firmware and disk image with identical content.
  The measurement covers the entire initial VM state.

### Both: "cryptography package not installed"

```bash
pip install cryptography
```

The `seal()`/`unseal()` methods use `cryptography` for AES-GCM.
Without it, they fall back to a base64 wrapper (insecure — for testing only).

---

## References

- [Gramine Documentation](https://gramine.readthedocs.io/)
- [Intel SGX Developer Guide](https://download.01.org/intel-sgx/)
- [AMD SEV-SNP ABI Specification](https://www.amd.com/en/developer/sev.html)
- [AMD KDS (Key Distribution Service)](https://kdsintf.amd.com/)
- [Intel IAS (Attestation Service)](https://api.trustedservices.intel.com/)
- [Confidential Computing Consortium](https://confidentialcomputing.io/)