# HCU CI

The HCU workflows follow the test layering used by the pinned upstream VERL
baseline while running only the test cases owned by this repository.

## Repository variables

Configure these variables before enabling the workflows:

| Variable | Purpose |
| --- | --- |
| `VERL_HCU_CI_RUNNER_LABEL` | Custom label of the self-hosted HCU runner |
| `VERL_HCU_PR_IMAGE` | HCU image used by the PR smoke job |
| `VERL_HCU_VLLM_IMAGE` | HCU image used by the vLLM nightly job |
| `VERL_HCU_SGLANG_IMAGE` | HCU image used by the SGLang nightly job |
| `VERL_HCU_MODEL_ROOT` | Read-only model root mounted into nightly containers |
| `VERL_HCU_DATA_ROOT` | Read-only dataset root mounted into nightly containers |

All image values must use an immutable digest:

```text
registry.example.com/project/image@sha256:<64 hexadecimal characters>
```

The PR configuration check requires only the runner label and PR image.
Nightly configuration additionally requires the vLLM and SGLang images plus
the model and dataset roots.

The current PR smoke and nightly model baselines run on an eight-card BW1000
runner. The runner must have the `bw1000` label and provide Docker,
`/opt/hyhal`, and the model and dataset roots configured above. Runtime
dependencies belong in the pinned images; the workflows do not install the
floating dependency from the product `requirements.txt`. BW1100 coverage will
be added as a separate labeled test lane when a runner is available.

The PR gate uses `pull_request_target`, so its authorization and runner
dispatch logic always come from the default branch rather than the PR. HCU
execution is limited to same-repository branches opened by an owner,
organization member, or repository collaborator. The HCU runner group must be
restricted to this private repository, and users with write access must be
treated as trusted to execute code on that runner.

During initial bootstrap, merge the reviewed workflow framework before relying
on PR HCU checks. The first follow-up PR is the earliest change that can execute
the default branch's trusted `pull_request_target` workflow.

## Workflows

- `PR Test (HCU)` always runs the quality gate. Changes under `hcu_verl/`
  additionally run the fixed-submodule, HCU device, patch, worker, and Ray
  smoke checks. Other paths do not occupy the HCU runner.
- `Nightly Test (HCU)` runs at 03:00 Asia/Shanghai. The vLLM and SGLang cases
  run serially on the same eight-card runner. Manual runs can select one case.

Model and dataset downloads are forbidden in both workflows. Nightly tests use
only the configured local roots and fail with a clear message if an input is
missing.

After the initial workflows are stable, configure only `PR Test (HCU) / Finish`
as the required branch-protection check. The finish job already evaluates every
required upstream job and prevents skipped runtime checks from being treated as
success.
