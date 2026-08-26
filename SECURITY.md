# 🛡️ Tantra-LLM Security Policy & Bug Bounty Program

Tantra-LLM is built on a **privacy-first, local-first, local-compute** security model. We take system security, numerical stability, and AI safety seriously. We welcome security researchers, AI red-teamers, and the open-source community to identify and responsibly disclose vulnerabilities.

---

## 🎯 1. Bug Bounty Program & Scope

### In-Scope Targets

| Target Component | Key Surface / Asset | Scope Description |
| :--- | :--- | :--- |
| **Model Engine & Architecture** | `Tantra/model.py`, `Tantra/bitnet.py`, `Tantra/moe.py` | Weight corruption, gradient norm / tensor explosions, unhandled NaNs/Infs, layer state bypass. |
| **Codec & Decompression** | `Tantra/codec.py` (`.dna`, ZSTD dictionary) | Deserialization exploits, parity check bypass, arbitrary memory allocation / decompression bombs. |
| **Tool Router & Sandbox** | `Tantra/tool_router.py` | Remote Code Execution (RCE) escapes, unauthorized file reads, path traversal (`../`), math parser injection. |
| **WebUI & API Server** | `webui/server.py`, FastAPI endpoints | Authentication bypass, unauthorized checkpoint/dataset manipulation, CORS/SSRF, prompt injection bypass. |
| **Tokenizer & Data Ingestion** | `Tantra/tokenizer.py`, `Tantra/dataset.py` | Out-of-bounds token IDs, memory exhaust vectors, malformed JSONL parser crashes. |

### Out-of-Scope Targets & Prohibited Actions
- **Destructive Denial of Service (DoS)** targeting hosted demo infrastructure.
- **Social Engineering / Phishing** against Tantra-LLM maintainers or contributors.
- **Physical Attacks** against developer hardware.
- **Upstream Third-Party Issues** (in PyTorch, FastAPI, etc.) that do not have direct exploitability within Tantra-LLM.
- **Non-reproducible hallucination claims** that do not bypass explicit safety guardrails or access control.

---

## 🏆 2. Severity Classification & Rewards

Reports are evaluated under the **Common Vulnerability Scoring System (CVSS v3.1)** and our AI-specific threat matrix.

| Severity Tier | CVSS Score | Example Vulnerabilities | Recognition & Reward |
| :--- | :--- | :--- | :--- |
| 🔴 **Critical** | 9.0 – 10.0 | • Remote Code Execution (RCE) in `tool_router` / sandbox escape<br>• Arbitrary code execution via checkpoint or `.dna` deserialization<br>• Unauthorized administrative command execution | **Hall of Fame (Tier 1)** + Lead Contributor Advisory Credit |
| 🟠 **High** | 7.0 – 8.9 | • Path traversal reading arbitrary files outside repo boundaries<br>• Authentication bypass on administrative WebUI endpoints<br>• Model weight corruption or state poison via crafted API payload | **Hall of Fame (Tier 2)** + Security Advisory Co-Author |
| 🟡 **Medium** | 4.0 – 6.9 | • Server crash / resource exhaustion via crafted token sequences<br>• AST evaluation bypass in safe calculator math<br>• Telemetry / local metadata exposure | **Hall of Fame (Tier 3)** + Release Note Acknowledgement |
| 🟢 **Low** | 0.1 – 3.9 | • Non-sensitive path disclosure<br>• Minor edge-case unhandled exception with no memory corruption<br>• UI-side rendering / XSS edge-cases in local WebUI | **Project Contributor Credit** |

---

## 📜 3. Safe Harbor Policy

We consider security research conducted under this policy to be **authorized, lawful, and in good faith**. If you adhere to these guidelines:
1. We will **not** pursue legal action against you.
2. We will work with you to understand and resolve the issue quickly.
3. We will recognize your contribution publicly in our Security Hall of Fame (unless you request anonymity).

**Researcher Guidelines**:
- Play by the rules: do not view, modify, or destroy data belonging to other users.
- Give us reasonable time (minimum 48 hours for triage, 30 days for fix deployment) before public disclosure.
- Act in good faith to avoid privacy violations, data destruction, and service interruption.

---

## 📬 4. Reporting a Security Vulnerability

If you discover a potential vulnerability:

1. **Do NOT open a public GitHub issue.**
2. Report privately via **GitHub Security Advisories**:
   - Navigate to the repository's **Security** tab $\rightarrow$ **Advisories** $\rightarrow$ **Report a vulnerability**.
3. **Include the following information**:
   - Detailed description of the vulnerability and attack vector.
   - Exact steps or proof-of-concept (PoC) script to reproduce the issue.
   - Impact assessment and proposed mitigation (if known).
   - Affected system components, Python version, and OS environment.

### ⏱️ SLA & Response Timelines
- **Initial Acknowledgment**: Within **48 hours**.
- **Triage & Severity Assessment**: Within **5 business days**.
- **Patch Release & Advisory Publication**: Coordinated with researcher within **14–30 days**.

---

## 🔒 5. Core Security Practices for Developers

- **Zero Secret Exposure**: Never commit `.env`, private keys, or API tokens.
- **Safe Serialization**: Use `weights_only=True` for PyTorch model checkpoint loading.
- **Strict Path Validation**: Verify all filesystem reads/writes resolve strictly within `REPO_ROOT`.
- **Sandboxed Execution**: Keep code execution tools disabled (`TANTRA_ENABLE_SANDBOX=0`) by default unless isolated in containerized microVMs.
