# Xiaoyaner v0.19 clean release candidate

Created: 2026-07-23T08:44:26Z
Base: `v2026.7.20` / `3ef6bbd201263d354fd83ec55b3c306ded2eb72a`
Branch: `xiaoyaner/v019-clean-release`
Live cutover: not performed by this candidate.

## Scope

This release candidate keeps the clean Hermes v0.19.0 release base and ports only the approved parent-task surfaces:

1. Alias explicit-provider guard from the upstream alias lane:
   - `7a0b843edc30c125b1bf7522459f704813335c21`
   - `7cb1c69a61633933503dbbf8b852726881120c3b`
2. Shared-session trusted sender envelope from the upstream sender lane:
   - `a09e6db230b064d61c7d34b1630bce0e09872f51`
   - `131d447c9ca3633927b0abd56338200aa73718d0`
3. Standalone plugin compatibility only, with no Graphiti or Fish Audio code vendored into Hermes core:
   - Graphiti memory plugin `xiaoyaner0201/hermes-graphiti-memory@fe4056ef4e2a393291770f4c2b81555327d97fd4`
   - Fish Audio TTS plugin `xiaoyaner0201/hermes-fishaudio-tts@8870c6a64b9c71bb68e0cb04bfc2c9451e60c091`

No `upstream/main` merge was performed. The branch remains based on `3ef6bbd201263d354fd83ec55b3c306ded2eb72a`.

## Changed core files

- `hermes_cli/model_switch.py`
  - Reverse direct-alias lookup now prefers the alias whose provider matches the requested/current provider.
  - Explicit `--provider` switching treats the requested provider as authoritative and ignores reverse aliases belonging to another provider.
  - Alias base URL reuse preserves the existing `api_mode` when the resolved alias base URL is unchanged.
- `gateway/run.py`
  - Shared multi-user inbound turns now strip user-supplied forged `[Verified sender: ...]` envelopes before adding Hermes-authenticated sender metadata.
  - Shared turns with trusted `user_id` / `user_id_alt` include a canonical verified sender envelope; Slack preserves `<@U...>` mention syntax.
  - Shared turns without trusted IDs fall back to a display-name prefix after stripping forged verified-sender text.
- `tests/hermes_cli/test_ollama_cloud_auth.py`
- `tests/gateway/test_image_input_routing_runtime.py`

## Compatibility lock

See `release-candidates/v2026.7.20-xiaoyaner/plugin-lock.json` for the pinned external plugin commits and install mode. These plugins stay outside this repo and should be installed into the target profile's isolated `$HERMES_HOME/plugins/` after human review.

## Isolated plugin smoke

Smoke root used during validation: `/tmp/hermes-v019-plugins.uGoZI7`.

The smoke used a temporary `HERMES_HOME` and did not modify the live profile. It symlinked Graphiti as `$HERMES_HOME/plugins/graphiti`, copied Fish Audio as `$HERMES_HOME/plugins/fishaudio-tts`, enabled only `fishaudio-tts` in config, and used a dummy Graphiti URL plus dummy Fish reference. No production credentials were read or printed.

Result:

- Graphiti + Fish Audio plugin unit suites: `23 passed in 0.51s`.
- Hermes `PluginManager` loaded `fishaudio-tts`: `plugin_loaded True`.
- Hermes TTS registry discovered provider: `tts_provider True`.
- Graphiti provider exposed schemas: `['graphiti_memory', 'recall']`.

## Verification record

Commands run from `/Users/dongsjoa/.hermes/worktrees/hermes-v019-clean-release`:

- `scripts/run_tests.sh tests/hermes_cli/test_ollama_cloud_auth.py tests/gateway/test_image_input_routing_runtime.py tests/gateway/test_shared_group_sender_prefix.py`
  - First run: `39 tests passed, 0 failed`.
  - Final rerun after conflict cleanup: `39 tests passed, 0 failed`.
- `scripts/run_tests.sh tests/agent/test_memory_provider.py tests/run_agent/test_memory_provider_init.py tests/tools/test_memory_tool.py tests/tools/test_tts_plugin_dispatch.py tests/tools/test_tts_command_providers.py tests/agent/test_tts_registry.py tests/hermes_cli/test_plugins_tts_registration.py`
  - `327 tests passed, 0 failed`.
- Isolated plugin command from the temporary plugin smoke root:
  - `.venv/bin/python -m pytest -q "$TMP_ROOT/repos/graphiti/tests" "$TMP_ROOT/repos/fishaudio/tests" --import-mode=importlib --rootdir="$TMP_ROOT/repos"`
  - `23 passed in 0.51s` after adding both standalone repo roots to `PYTHONPATH`.
- CLI smoke with temporary `HERMES_HOME`:
  - `.venv/bin/hermes --version`
  - `.venv/bin/hermes config path`
  - `.venv/bin/hermes tools list`

## Rollback

Before live cutover, rollback is just branch-level:

```bash
git switch xiaoyaner/v019-clean-release
git reset --hard v2026.7.20
```

After installing standalone plugins into a profile, rollback is profile-local and does not require changing Hermes core:

```bash
# In the affected profile only
rm -f "$HERMES_HOME/plugins/graphiti"
rm -rf "$HERMES_HOME/plugins/fishaudio-tts"
# Restore previous config.yaml from backup, or set provider choices back explicitly.
hermes config set memory.provider builtin
hermes config set tts.provider edge
```

If a release-candidate commit has been created and needs to be removed without rewriting reviewed history, use `git revert <candidate-commit>` instead of merging upstream or resetting a shared branch.
