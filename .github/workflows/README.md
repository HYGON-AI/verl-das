# HCU CI

The HCU workflows follow the test layering used by the pinned upstream VERL
baseline while running only the test cases owned by this repository.

## Repository variables

Configure these variables before enabling the workflows:

| Variable | Purpose |
| --- | --- |
| `VERL_HCU_CI_IMAGE` | Fixed image tag shared by PR, nightly vLLM, and nightly SGLang jobs |
| `VERL_HCU_MODEL_ROOT` | Read-only model root mounted into PR E2E and nightly containers |
| `VERL_HCU_DATA_ROOT` | Parent of the read-only `gsm8k/` dataset directory used by PR E2E and nightly jobs |

Set `VERL_HCU_CI_IMAGE` to one fixed tag. Both workflows read the repository
variable, so updating it switches PR, nightly vLLM, and nightly SGLang jobs
together:

```text
registry.example.com/project/image:version-tag
```

The shared configuration check validates the model and dataset roots used by
model tests.

All HCU CI assets use the runner-shared directory:

```text
VERL_HCU_MODEL_ROOT=/ci_public/verl-das/models
VERL_HCU_DATA_ROOT=/ci_public/verl-das/data
```

Both paths are read directly from inside the job container (the PR jobs run on
the ARC K8s pool, whose template injects `/public` and `/ci_public`).

The PR jobs run on the `hygon-bw1100-8card` ARC K8s pool: the job template
injects the eight HCU cards, `/dev/shm`, `/opt/hyhal`, `/public` and
`/ci_public`, and the job itself runs inside `VERL_HCU_CI_IMAGE`, so no
Docker-in-Docker is required. The nightly workflow still runs on the host
runner labelled `bw1100` and keeps the Docker-based launchers.

All HCU CI support and test launchers live under `tests/hcu/`: shared helpers
are in `tests/hcu/ci/`, PR smoke is in `tests/hcu/pr/`, and nightly launchers
are in `tests/hcu/nightly/`.
Nightly case inventories are grouped by accelerator label under
`tests/hcu/nightly/<accelerator>/cases.json`; the current cases are registered
in `tests/hcu/nightly/bw1000/cases.json`. That file is the source of the
nightly job matrix, so adding a model requires only a launcher and one case
entry. Manual runs accept `all`, an engine name, an exact case ID, or a
comma-separated selection.

The nightly runner must provide Docker, `/opt/hyhal`, and the model and dataset
roots configured above; the PR ARC pool provides the cards, `/opt/hyhal`,
`/public` and `/ci_public` directly. Runtime dependencies belong in the pinned
images. The PR unit lane installs its optional Python dependencies before the
suite. The vLLM launcher also installs the fixed `TransferQueue==0.1.9` package
when the pinned runtime image does not already provide it.

The nightlies still start and remove HCU containers explicitly inside each job;
the cleanup step stops only resources carrying the current CI run ID, restores
the mounted checkout to the runner UID/GID while the container is still
running, and then removes that exact container. The PR jobs run inside the image
itself and use the same cleanup helper for Ray, PID and temporary-directory
state instead.

The PR gate uses `pull_request_target`, so its authorization and runner
dispatch logic always come from the default branch rather than the PR. HCU
execution accepts both same-repository and fork pull requests. The HCU runner
must remain a dedicated isolated CI machine because pull request contributors
are treated as able to execute code on that runner.

During initial bootstrap, merge the reviewed workflow framework before relying
on PR HCU checks. The first follow-up PR is the earliest change that can execute
the default branch's `pull_request_target` workflow.

## Workflows

Pull requests targeting `main` start the HCU workflow when a maintainer adds
the `run-ci` label; opening, updating, reopening or marking a pull request
ready for review no longer starts it. The workflow only runs for label events
and only when the pull request changes one of these paths:

- `hcu_verl/**`
- `tests/hcu/**`
- `.github/workflows/pr-test-hcu.yml`
- `.gitmodules`
- `third_party/verl`
- `third_party/Megatron-LM`
- `third_party/VeOmni`
- `requirements.txt`

GitHub applies these filters before dispatching any HCU runner job. Matching
pull requests run the complete PR suite; unrelated changes do not start the
HCU workflow.

`run-ci` stays on the pull request after it dispatches a run, and because the
workflow no longer listens to pushes, HCU CI runs once per label addition. To
start another run, remove the label and add it again (adding an already-present
label produces no new event).

- `Quality Gate` reuses the organization-wide incremental checks from
  `HYGON-AI/quality-gate`. It runs independently for pull requests targeting
  `main`.
- `PR Test (HCU)` has three jobs: unit, runtime, and finish. Matching changes
  run three test layers owned by this repository: the HCU unit suite,
  fixed-submodule/device/patch/worker/Ray smoke checks, and a real one-step
  Qwen2.5-0.5B GRPO FSDP/vLLM training case on eight cards.
- `Nightly Test (HCU)` runs at 03:00 Asia/Shanghai. The vLLM and SGLang cases
  from the generated matrix run serially on the same eight-card runner.

Model and dataset downloads are forbidden in both workflows. PR E2E and
nightly tests use only the configured local roots and fail with a clear message
if an input is missing.

The HCU finish job evaluates both HCU test jobs and fails unless both complete
successfully. If it is later configured as a required branch-protection check,
review how GitHub treats path-filtered workflows before enabling that rule.
