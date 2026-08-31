# Security Policy & AI Safety Guidance

## Overview

Tantra-LLM is designed with a **privacy-first, local-first, local-compute** security model. The codebase contains no embedded analytics, telemetry, or remote backdoors. All execution, hardware profiling, tokenization, and expert loading take place on local hardware.

## Security Practices for Developers

### 1. Secret Protection
- **Never commit `.env` files, API keys, or private tokens.**
- Use environment variables or local key vaults for cloud integrations.
- Always check `git status` before pushing changes to remote repositories.

### 2. Model & Dataset Safety
- Do not commit large binary model checkpoints (`*.pt`, `*.bin`, `*.dna`) or raw datasets (`*.jsonl`) directly to Git. Use release attachments, HuggingFace Hub, or local directory links instead.
- Sanitize training datasets for PII (Personally Identifiable Information) before running training loops.

### 3. Execution Safety
- Hardware auto-detection queries hardware specifications using standard system interfaces (`psutil`, `py-cpuinfo`). No administrative privileges are required or requested.
- Expert networks are decompressed in-memory from `.dna` binary streams with CRC/SHA-256 integrity verification to protect against file corruption.

## Reporting a Security Vulnerability

If you discover a security vulnerability or potential exploit within Tantra-LLM:

1. **Do NOT open a public issue.**
2. Report the vulnerability privately by submitting a Security Advisory on GitHub or contacting the project maintainers directly.
3. Include detailed steps to reproduce the issue, environment information, and proposed remediation if available.
4. Maintainers will acknowledge reports within 48 hours and provide status updates as fixes are investigated and deployed.
