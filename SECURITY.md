# Security

## Supported version

Security fixes are applied to the latest release on `main`.

## Deployment boundary

Evaluated agents must use the remote API and the client-only wheel. Do not run
untrusted agent code inside the server container or provide it with repository
access. The server-only ticket bank and grading rubric are deliberately excluded
from the client wheel and must be treated as evaluation secrets.

Production deployments require `SUPPORT_ENV_API_KEY` and a private,
high-entropy `SUPPORT_ENV_SEED_SALT`, each at least 32 characters. Send the API key through either the
`X-API-Key` header or `Authorization: Bearer <key>`. Rotate the key if it is
exposed. The client refuses to transmit API keys over remote plaintext WebSocket
connections by default; production endpoints must use TLS (`https://`/`wss://`).
The in-process rate limiter is suitable for a single-worker demo; use a
shared gateway-backed limiter for multi-worker or multi-replica deployments.

The interactive playground is disabled by default in production. Production only
exempts `/health` from API-key authentication. If the playground is explicitly
enabled, place the entire service behind an authenticated reverse proxy.

Authentication failures count toward the per-client rate limit. The in-process
limiter bounds tracked client identities with `SUPPORT_ENV_MAX_RATE_LIMIT_CLIENTS`
(default `10000`) to prevent memory exhaustion. Configure stricter limits at the
edge for internet-facing deployments.

For scored evaluation, mount a held-out JSON ticket bank outside the repository
and set `SUPPORT_ENV_TICKET_BANK` to its path. It must use the same fields as the
reference bank. Never publish the held-out bank, seed salt, or evaluation seeds.

## Reporting a vulnerability

Open a private GitHub security advisory rather than a public issue. Include the
affected revision, reproduction steps, impact, and any proposed mitigation.
