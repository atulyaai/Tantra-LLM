# Security and Release Guidance

## Private by Default

Keep these files out of public commits unless they are intentionally sanitized for release:

- raw training datasets
- private prompts or memory exports
- checkpoints and generated model artifacts
- logs
- local archives
- credentials, tokens, API keys, and `.env` files

## Agent Tools

Review every enabled agent tool before shared or public use. Tools that access the network, write memory, read files, or evaluate user-provided input should have an explicit policy, timeout, and audit path.

## Reporting

If you find a security issue, open a private report with reproduction steps, affected files, and expected impact.
