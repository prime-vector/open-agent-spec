# OA Conformance Matrix

| Case | python-reference 1.5.0 | npm 1.5.1 |
|---|---|---|
| schema/invalid-engine | ✅ PASS | ✅ PASS |
| schema/invalid-version | ✅ PASS | ✅ PASS |
| schema/missing-agent | ✅ PASS | ✅ PASS |
| schema/missing-intelligence | ✅ PASS | ✅ PASS |
| schema/missing-tasks | ✅ PASS | ✅ PASS |
| schema/valid-minimal | ✅ PASS | ✅ PASS |
| prompt-resolution/cli-override | ✅ PASS | ✅ PASS |
| prompt-resolution/global-fallback | ✅ PASS | ✅ PASS |
| prompt-resolution/independent-resolution | ✅ PASS | ✅ PASS |
| prompt-resolution/legacy-task-map | ✅ PASS | ✅ PASS |
| prompt-resolution/per-task-inline | ✅ PASS | ✅ PASS |
| depends-on/cycle-detection | ✅ PASS | ✅ PASS |
| depends-on/linear-chain | ✅ PASS | ✅ PASS |
| depends-on/merge-order | ✅ PASS | ✅ PASS |
| depends-on/no-chain-key-without-deps | ✅ PASS | ✅ PASS |
| depends-on/unknown-dependency | ✅ PASS | ✅ PASS |
| response-format/fence-stripping-no-lang | ✅ PASS | ✅ PASS |
| response-format/fence-stripping | ✅ PASS | ✅ PASS |
| response-format/json-default | ✅ PASS | ✅ PASS |
| response-format/output-schema-validation | ✅ PASS | ⬜ UNSUPPORTED |
| response-format/text-mode | ✅ PASS | ✅ PASS |
| delegation/default-task-name | ✅ PASS | ✅ PASS |
| delegation/local-path | ✅ PASS | ✅ PASS |
| delegation/missing-task | ✅ PASS | ✅ PASS |
| errors/chain-cycle | ✅ PASS | ✅ PASS |
| errors/chain-input-missing | ✅ PASS | ✅ PASS |
| errors/contract-violation | ✅ PASS | ⬜ UNSUPPORTED |
| errors/error-structure | ✅ PASS | ✅ PASS |
| errors/task-not-found | ✅ PASS | ✅ PASS |

## Summary

| Runtime | Pass | Fail | Unsupported | Adapter errors |
|---|---|---|---|---|
| python-reference 1.5.0 | 29 | 0 | 0 | 0 |
| npm 1.5.1 | 27 | 0 | 2 | 0 |

Legend: ✅ PASS · ❌ FAIL · ⬜ UNSUPPORTED (capability not declared) · 💥 adapter error
