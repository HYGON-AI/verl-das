# HCU CI

The HCU workflows follow the test layering used by the pinned upstream VERL
baseline while running only the test cases owned by this repository.

## Repository variables

Configure these variables before enabling the workflows:

| Variable | Purpose |
| --- | --- |
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
`tests/hcu/nightly/<accelerator>/cases.json`; the current cases are registered
in `tests/hcu/nightly/bw1000/cases.json`. That file is the source of the
nightly job matrix, so adding a model requires only a launcher and one case
entry. Manual runs accept `all`, an engine name, an exact case ID, or a
comma-separated selection.

The runner must provide Docker, `/opt/hyhal`, and the model and dataset roots
configured above. Runtime dependencies belong in the pinned images. The only
PR-time installation is the fixed `TransferQueue==0.1.9` package immediately
before the HCU unit suite; the runtime lane still uses the original image
environment.

HCU containers are started and removed explicitly inside each test job. The
cleanup step stops only resources carrying the current CI run ID, restores the
mounted checkout to the runner UID/GID while the container is still running,
and then removes that exact container. This avoids separate permission-repair
jobs and leaves `actions/checkout` post-job cleanup a runner-owned workspace.

The PR gate uses `pull_request_target`, so its authorization and runner
dispatch logic always come from the default branch rather than the PR. HCU
execution accepts both same-repository and fork pull requests. The HCU runner
must remain a dedicated isolated CI machine because pull request contributors
are treated as able to execute code on that runner.

During initial bootstrap, merge the reviewed workflow framework before relying
on PR HCU checks. The first follow-up PR is the earliest change that can execute
the default branch's `pull_request_target` workflow.

## Workflows

PR HCU routing uses three explicit profiles from `tests/hcu/ci/plan_pr.py`:

Pull requests targeting `main` are planned when they are opened, updated,
reopened, or marked ready for review. Path selection deliberately lives in
the planner instead of `on.pull_request_target.paths`: the plan and finish jobs
must still report a stable required check for the `none` profile, while the
HCU unit and runtime lanes are skipped.

| Profile | Trigger | HCU lanes |
| --- | --- | --- |
| `none` | Documentation and other unrelated changes | None; finish reports the skip |
| `unit` | Python changes outside examples, upstream sanity, and CPU-only tests | Unit |
| `runtime` | `hcu_verl/`, `tests/hcu/`, the pinned VERL submodule, `.gitmodules`, or the PR HCU workflow | Unit and runtime |

- `Quality Gate` reuses the organization-wide incremental checks from
  `HYGON-AI/quality-gate`. It runs independently for pull requests targeting
  `main`.
- `PR Test (HCU)` has four jobs: plan, unit, runtime, and finish. Python changes
  run the HCU unit lane. HCU runtime, HCU test, pinned VERL submodule, and
  PR-workflow changes run all four test layers from `tests/hcu/pr/`: 219
  pinned-upstream sanity/CPU tests, the HCU unit suite, fixed-submodule/device/
  patch/worker/Ray smoke checks, and a real one-step Qwen2.5-0.5B GRPO
  FSDP/vLLM training case on eight cards. Documentation-only changes skip both
  HCU test lanes.
- `Nightly Test (HCU)` runs at 03:00 Asia/Shanghai. The vLLM and SGLang cases
  from the generated matrix run serially on the same eight-card runner.

Model and dataset downloads are forbidden in both workflows. PR E2E and
nightly tests use only the configured local roots and fail with a clear message
if an input is missing.

After the workflows are stable, configure `Checks / All required checks` and
`PR Test (HCU) / Finish` as required branch-protection checks. The HCU finish
job evaluates every HCU-specific upstream job and prevents skipped runtime
checks from being treated as success.
