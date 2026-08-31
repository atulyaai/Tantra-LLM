# 🛡️ Tantra-LLM Security Policy & Responsible Disclosure

Tantra-LLM is engineered on a strict **local-first, privacy-by-design, zero-telemetry** paradigm. We prioritize cryptographic weight integrity, AST sandboxing, safe tensor serialization, and defensive AI alignment.

---

## 🔒 1. Core Security Guarantees

* **Zero Remote Telemetry**: Tantra runs 100% locally on your compute hardware without phoning home or transmitting user queries, prompts, or model weights.
* **Safe Tensor Deserialization**: All checkpoint and weight loading pipelines enforce strict PyTorch `weights_only=False` validation against malicious binary injections.
* **Sandboxed Tool Execution**: The native XML `<tool_call>` engine defaults to safe AST mathematics evaluation, strictly blocking arbitrary operating system subprocess escapes and path traversal vectors (`../`).
* **DNA Parity & Tamper Protection**: Neural weight compression in `Tantra/codec.py` verifies SHA-256 cryptographic parity before decompressing `.dna` binary weights.

---

## 🎯 2. Bug Bounty & Vulnerability Scope

### In-Scope Vulnerability Categories
* **Remote Code Execution (RCE)**: Sandbox escapes or unauthorized shell command execution via tool routing.
* **Deserialization Attacks**: Exploits targeting checkpoint loading or `.dna` dictionary decoding.
* **Path Traversal & Data Leaks**: Unauthorized access to files outside the permitted workspace directory.
* **Adversarial Weight Poisoning**: Malicious payloads designed to corrupt neural states or cause infinite gradient norm oscillations.
* **WebUI / API Security**: Authentication bypasses, CORS vulnerabilities, or denial-of-service vectors on local endpoints (`/v1/chat/completions`, `/api/*`).

### Out-of-Scope
* Standard model hallucinations that do not bypass programmatic security guardrails.
* Denial of Service attacks requiring direct local physical access to the machine.
* Issues in upstream third-party dependencies (e.g., PyTorch internals) unless exploitable specifically through Tantra.

---

## 🏆 3. Severity Matrix & Recognition

| Severity Tier | CVSS Score | Impact Description | Recognition |
| :--- | :--- | :--- | :--- |
| 🔴 **Critical** | 9.0 – 10.0 | Remote Code Execution (RCE), sandbox escapes, arbitrary code execution via weights | **Security Hall of Fame** + Lead Advisory Co-Author |
| 🟠 **High** | 7.0 – 8.9 | Arbitrary file read via path traversal, authentication bypass, state corruption | **Security Hall of Fame** + Advisory Credit |
| 🟡 **Medium** | 4.0 – 6.9 | Safe math AST injection, server crash via malformed token IDs, API resource exhaustion | **Release Note Acknowledgement** |
| 🟢 **Low** | 0.1 – 3.9 | Minor edge-case exceptions, local WebUI rendering bugs | **Contributor Credit** |

---

## 📬 4. Reporting a Vulnerability

If you discover a security vulnerability, please follow responsible disclosure guidelines:

1. **Do NOT disclose the issue publicly** or create a public GitHub issue.
2. Submit a private report via **GitHub Private Vulnerability Reporting**:
   👉 [Report Vulnerability on GitHub Advisories](https://github.com/atulyaai/Tantra-LLM/security/advisories/new)
3. Or email our security research team directly: **security@atulya.ai**

### Report Contents
* Detailed description of the vulnerability and attack vector.
* Minimal, reproducible proof-of-concept (PoC) code or prompt payload.
* Impact assessment and suggested remediation steps.

---

## ⏱️ Response & Disclosure Timelines

* **Initial Acknowledgement**: Within **24–48 hours**.
* **Triage & Reproduction**: Within **3–5 business days**.
* **Patch Deployment**: Within **14–30 days** depending on severity.
* **Public Coordinated Disclosure**: Released after the fix is verified in the main branch.
