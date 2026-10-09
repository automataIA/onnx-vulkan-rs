# Security policy

## Supported versions

The project is pre-release (0.0.x). Only the latest commit on `main` receives
fixes.

## Reporting a vulnerability

Report privately through GitHub:
[Security → Report a vulnerability](https://github.com/automataIA/onnx-vulkan-rs/security/advisories/new).
Do not open a public issue for a suspected vulnerability.

Include the affected crate or script, the commit, and a minimal reproduction
(for example the ONNX file that triggers the problem). Reports are answered on
a best-effort basis.

## Scope

In scope: memory safety in the runtime and the ONNX Runtime plugin, parsing of
untrusted ONNX files and their external weights, and the download scripts in
`scripts/`.

Out of scope: the GPU driver and the Vulkan loader, ONNX Runtime itself, and
third-party models fetched by the scripts. Report those upstream.
