# HCU CI

The HCU workflows follow the test layering used by the pinned upstream VERL
baseline while running only the test cases owned by this repository.

## Repository variables

Configure these variables before enabling the workflows:

| Variable | Purpose |
| --- | --- |
| `VERL_HCU_CI_RUNNER_LABEL` | Custom label of the self-hosted HCU runner |
| `VERL_HCU_PR_IMAGE` | HCU image used by all PR test layers |
| `VERL_HCU_VLLM_IMAGE` | HCU image used by the vLLM nightly job |
| `VERL_HCU_SGLANG_IMAGE` | HCU image used by the SGLang nightly job |
| `VERL_HCU_MODEL_ROOT` | Read-only model root mounted into PR E2E and nightly containers |
| `VERL_HCU_DATA_ROOT` | Parent of the read-only `gsm8k/` dataset directory used by PR E2E and nightly jobs |

All image values must use an immutable digest:

```text
registry.example.com/project/image@sha256:<64 hexadecimal characters>
```

The PR configuration check requires the runner label, PR image, model root,
and dataset root because the PR gate includes a real one-step training case.
Nightly configuration additionally requires the vLLM and SGLang images plus
the model and dataset roots.

For the current BW1000 cases, set `VERL_HCU_DATA_ROOT=/home/github/tly` when
the complete dataset is stored in `/home/github/tly/gsm8k`. The workflow
mounts only `${VERL_HCU_DATA_ROOT}/gsm8k` into the container and keeps it
read-only.

The current PR smoke and nightly model baselines run on an eight-card runner.
All HCU CI support and test launchers live under `tests/hcu/`: shared helpers
are in `tests/hcu/ci/`, PR smoke is in `tests/hcu/pr/`, and nightly launchers
are in `tests/hcu/nightly/`.
Nightly case inventories are grouped by accelerator label under
`tests/hcu/nightly/<accelerator>/ci_cases.yaml`; the current cases are
registered in `tests/hcu/nightly/bw1000/ci_cases.yaml`. The workflow uses the
matching `bw1000` runner label for dispatch. A future accelerator is added as
a separate directory and runner lane instead of mixing hardware baselines.

The runner must provide Docker, `/opt/hyhal`, and the model and dataset roots
configured above. Runtime dependencies belong in the pinned images; the
workflows do not install the floating dependency from the product
`requirements.txt`.

Every container test is followed by a host-side ownership restoration job
before the next test layer starts. It validates that `GITHUB_WORKSPACE` is
inside the runner work root, derives the runner UID/GID from `RUNNER_TEMP`, and
uses the already-pinned HCU image to restore only that workspace. Keeping this
as a separate job makes it run after `actions/checkout` post-job cleanup.

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

- `Quality Gate` reuses the organization-wide incremental checks from
  `HYGON-AI/quality-gate`. It runs independently for pull requests targeting
  `main`.
- `PR Test (HCU)` runs the HCU-specific authorization and routing checks.
  Python changes run the repository HCU unit suite. HCU runtime, HCU test,
  pinned VERL submodule, and PR-workflow changes run four test layers from
  `tests/hcu/pr/`: 219 pinned-upstream sanity/CPU tests, the HCU unit suite,
  fixed-submodule/device/patch/worker/Ray smoke checks, and a real one-step
  Qwen2.5-0.5B GRPO FSDP/vLLM training case on eight cards. Documentation-only
  changes do not occupy the HCU runner.
- `Nightly Test (HCU)` runs at 03:00 Asia/Shanghai. The vLLM and SGLang cases
  under `tests/hcu/nightly/bw1000/` run serially on the same eight-card runner.
  Manual runs can select one case.

Model and dataset downloads are forbidden in both workflows. PR E2E and
nightly tests use only the configured local roots and fail with a clear message
if an input is missing.

After the workflows are stable, configure `Checks / All required checks` and
`PR Test (HCU) / Finish` as required branch-protection checks. The HCU finish
job evaluates every HCU-specific upstream job and prevents skipped runtime
checks from being treated as success.
