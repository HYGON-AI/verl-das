#!/usr/bin/env bash

set -euo pipefail

normalize_submodule_gitdirs() {
    local repo_root="$1"
    local git_file git_dir relative_git_dir temp_file

    while IFS= read -r -d '' git_file; do
        IFS= read -r git_dir < "${git_file}" || continue
        git_dir="${git_dir#gitdir: }"
        case "${git_dir}" in
            "${repo_root}/.git/"*) ;;
            *) continue ;;
        esac

        if [[ ! -d "${git_dir}" ]]; then
            echo "ERROR: submodule git directory is missing: ${git_dir}" >&2
            return 1
        fi
        relative_git_dir="$(
            realpath --relative-to="$(dirname "${git_file}")" "${git_dir}"
        )"
        temp_file="${git_file}.hcu-ci.$$"
        printf 'gitdir: %s\n' "${relative_git_dir}" > "${temp_file}"
        mv -f -- "${temp_file}" "${git_file}"
    done < <(find "${repo_root}/third_party" -type f -name .git -print0)
}

prepare_hcu_workspace() {
    local script_dir repo_root patch_source patch_target joined_python_path patch_tmp
    local attempt
    local -a python_paths

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/../../.." && pwd)"
    patch_source="${repo_root}/hcu_verl/patch_init.py"
    patch_target="${repo_root}/third_party/verl/verl/__init__.py"

    for safe_path in \
        "${repo_root}" \
        "${repo_root}/third_party/verl" \
        "${repo_root}/third_party/verl/recipe" \
        "${repo_root}/third_party/Megatron-LM" \
        "${repo_root}/third_party/VeOmni"; do
        if ! git config --global --get-all safe.directory 2>/dev/null |
            grep -Fqx -- "${safe_path}"; then
            git config --global --add safe.directory "${safe_path}"
        fi
    done

    if [[ -e "${repo_root}/third_party/verl/.git" ]] &&
        git -C "${repo_root}/third_party/verl" cat-file \
            -e "HEAD:verl/__init__.py" 2>/dev/null; then
        git -C "${repo_root}/third_party/verl" restore \
            --source=HEAD --staged --worktree -- verl/__init__.py
    fi

    git -C "${repo_root}" submodule sync --recursive
    for attempt in 1 2 3; do
        if git -c http.version=HTTP/1.1 -C "${repo_root}" submodule update \
            --init --recursive --depth 1; then
            break
        fi
        if [[ ${attempt} -eq 3 ]]; then
            echo "ERROR: failed to initialize HCU CI submodules after ${attempt} attempts" >&2
            return 1
        fi
        echo "WARNING: retrying HCU CI submodules after attempt ${attempt}" >&2
        sleep "$((attempt * 5))"
    done
    normalize_submodule_gitdirs "${repo_root}"
    python3 "${script_dir}/verify_submodules.py" --repo-root "${repo_root}"

    python_paths=(
        "${repo_root}"
        "${repo_root}/third_party/verl"
        "${repo_root}/third_party/Megatron-LM"
        "${repo_root}/third_party/VeOmni"
    )
    joined_python_path="$(IFS=:; printf '%s' "${python_paths[*]}")"
    export PYTHONPATH="${joined_python_path}${PYTHONPATH:+:${PYTHONPATH}}"
    export VERL_PATH="${repo_root}"

    if [[ ! -f "${patch_source}" ]]; then
        echo "ERROR: HCU patch source is missing: ${patch_source}" >&2
        return 1
    fi
    if [[ ! -f "${patch_target}" ]]; then
        echo "ERROR: verl patch target is missing: ${patch_target}" >&2
        return 1
    fi

    if ! cmp -s "${patch_source}" "${patch_target}"; then
        patch_tmp="${patch_target}.hcu-ci.$$"
        if ! cp -- "${patch_source}" "${patch_tmp}"; then
            rm -f -- "${patch_tmp}"
            echo "ERROR: failed to stage HCU patch at ${patch_tmp}" >&2
            return 1
        fi
        if ! chmod --reference="${patch_target}" "${patch_tmp}"; then
            rm -f -- "${patch_tmp}"
            echo "ERROR: failed to preserve mode for HCU patch target" >&2
            return 1
        fi
        if ! mv -f -- "${patch_tmp}" "${patch_target}"; then
            rm -f -- "${patch_tmp}"
            echo "ERROR: failed to install HCU patch target" >&2
            return 1
        fi
    fi

    if ! cmp -s "${patch_source}" "${patch_target}"; then
        echo "ERROR: HCU patch injection verification failed: ${patch_target}" >&2
        return 1
    fi
    if ! grep -Fq "from hcu_verl import verl_adaptor" "${patch_target}"; then
        echo "ERROR: HCU patch target does not load hcu_verl.verl_adaptor" >&2
        return 1
    fi

    echo "HCU workspace prepared at ${repo_root}"
}

prepare_hcu_workspace
