"""Remote entrypoint for running the committee smoke training on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


JUDGE_ARTIFACT_PATTERNS = [
    "baseline_report.json",
    "training_log_history.json",
    "judging_report.json",
    "judging_report.md",
    "onsite_demo_summary.md",
    "reward_series.json",
    "reward_curve.png",
]


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def clone_repo(repo_url: str, repo_ref: str, destination: Path) -> None:
    run(["git", "clone", "--depth", "1", "--branch", repo_ref, repo_url, str(destination)])
    print(f"cloned_repo={destination}", flush=True)
    run(["git", "rev-parse", "--short", "HEAD"], cwd=destination)


def summarize_outputs(output_dir: Path) -> None:
    print(f"output_dir={output_dir}", flush=True)
    if not output_dir.exists():
        print("warning=output_dir_missing", flush=True)
        return

    files = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file())
    for path in files:
        print(f"artifact={path}", flush=True)

    for name in ("judging_report.md", "onsite_demo_summary.md"):
        artifact = output_dir / name
        if artifact.exists():
            print(f"----- {name} -----", flush=True)
            print(artifact.read_text(encoding="utf-8"), flush=True)


def upload_artifacts(
    output_dir: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    token: str,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        path_in_repo=path_in_repo,
        allow_patterns=JUDGE_ARTIFACT_PATTERNS,
        commit_message=f"Upload HF Jobs artifacts for {path_in_repo}",
    )
    print(f"artifact_repo=dataset://{repo_id}", flush=True)
    print(f"artifact_repo_path={path_in_repo}", flush=True)
    print(f"artifact_repo_url=https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}", flush=True)
    for pattern in JUDGE_ARTIFACT_PATTERNS:
        file_path = output_dir / pattern
        if file_path.exists():
            print(
                f"artifact_url=https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}/{pattern}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run committee smoke training on Hugging Face Jobs.")
    parser.add_argument("--repo-url", required=True, help="Public Git repo URL to clone inside the job.")
    parser.add_argument("--repo-ref", default="main", help="Git ref to clone.")
    parser.add_argument(
        "--job-output-dir",
        default="outputs/committee-grpo-hf-job",
        help="Output dir inside the cloned repo.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--repeats-per-task", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--colab-email", default="anuagar@groww.in")
    parser.add_argument("--notes", default="HF Jobs smoke run for Round 2 committee environment.")
    parser.add_argument(
        "--artifact-repo",
        default=None,
        help="Optional HF dataset repo id used to persist judge-facing artifacts.",
    )
    parser.add_argument(
        "--artifact-subdir",
        default=None,
        help="Optional subdirectory inside the artifact repo.",
    )
    parser.add_argument(
        "--print-baselines",
        action="store_true",
        help="Print baseline summary before training starts.",
    )
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="amc-hf-job-"))
    repo_dir = workspace / "repo"

    try:
        clone_repo(args.repo_url, args.repo_ref, repo_dir)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        train_command = [
            sys.executable,
            "training/committee_grpo_train.py",
            "--model",
            args.model,
            "--output-dir",
            args.job_output_dir,
            "--repeats-per-task",
            str(args.repeats_per_task),
            "--max-steps",
            str(args.max_steps),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--lora-target-modules",
            args.lora_target_modules,
            "--colab-email",
            args.colab_email,
            "--notes",
            args.notes,
        ]
        if args.use_lora:
            train_command.append("--use-lora")
        if args.print_baselines:
            train_command.append("--print-baselines")

        print(
            "job_config="
            f"repo={args.repo_url} ref={args.repo_ref} model={args.model} "
            f"repeats_per_task={args.repeats_per_task} max_steps={args.max_steps} "
            f"use_lora={args.use_lora} output_dir={args.job_output_dir}",
            flush=True,
        )
        run(train_command, cwd=repo_dir)
        output_dir = repo_dir / args.job_output_dir
        summarize_outputs(output_dir)

        artifact_repo = args.artifact_repo
        hf_token = os.getenv("HF_TOKEN")
        artifact_subdir = args.artifact_subdir or datetime.now(timezone.utc).strftime(
            "hf-job-%Y%m%d-%H%M%S"
        )
        if artifact_repo and hf_token:
            upload_artifacts(
                output_dir,
                repo_id=artifact_repo,
                path_in_repo=artifact_subdir,
                token=hf_token,
            )
        elif artifact_repo and not hf_token:
            print("warning=artifact_repo_requested_but_hf_token_missing", flush=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
