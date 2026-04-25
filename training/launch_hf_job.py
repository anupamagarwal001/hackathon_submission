"""Launch and manage Hugging Face Jobs for the committee smoke training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, SpaceHardware, get_token, run_uv_job
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

DEFAULT_REPO_URL = "https://github.com/anupamagarwal001/hackathon_submission.git"
DEFAULT_NAMESPACE = "anupamagarwal001"
DEFAULT_FLAVOR = "t4-small"
DEFAULT_TIMEOUT = "2h"
DEFAULT_OUTPUT_DIR = "outputs/committee-grpo-hf-job"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_DEPENDENCIES = [
    "trl",
    "datasets",
    "accelerate",
    "matplotlib",
    "peft",
    "jmespath",
    "openenv-core[core]>=0.2.2",
    "openai",
    "git+https://github.com/huggingface/transformers.git@main",
]


def _space_hardware(value: str) -> SpaceHardware:
    try:
        return SpaceHardware(value)
    except ValueError as exc:  # pragma: no cover
        choices = ", ".join(h.value for h in SpaceHardware)
        raise argparse.ArgumentTypeError(f"Unknown flavor {value!r}. Expected one of: {choices}") from exc


def _print_job(job) -> None:
    payload = {
        "id": job.id,
        "status": str(job.status),
        "flavor": getattr(job.flavor, "value", job.flavor),
        "url": job.url,
    }
    print(json.dumps(payload, indent=2))


def _token_display(token: Any) -> str:
    if isinstance(token, str):
        return "provided-token"
    if token is True:
        return "local-login"
    if token is False or token is None:
        return "anonymous"
    return str(token)


def resolve_job_token(token: Any) -> str | bool | None:
    """Resolve launcher auth into the concrete token form expected by run_uv_job()."""

    if isinstance(token, str) and token.strip():
        return token
    if token is True:
        resolved = get_token()
        if not resolved:
            raise RuntimeError(
                "Requested local Hugging Face login, but no saved token was found. "
                "Run `hf auth login` first or pass --token <hf_token>."
            )
        return resolved
    return token


def ensure_hf_auth(namespace: str, token: Any) -> None:
    """Fail early with a clear message if the local machine is not logged in."""

    api = HfApi()
    try:
        who = api.whoami(token=token)
    except (HfHubHTTPError, LocalTokenNotFoundError) as exc:  # pragma: no cover
        token_file_candidates = [
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ]
        token_files = [str(path) for path in token_file_candidates if path.exists()]
        raise RuntimeError(
            "Hugging Face authentication is missing or invalid.\n"
            f"- launcher auth mode: {_token_display(token)}\n"
            f"- namespace: {namespace}\n"
            f"- HF_TOKEN env present: {'yes' if os.getenv('HF_TOKEN') else 'no'}\n"
            f"- saved token files: {token_files if token_files else 'none'}\n\n"
            "Why this matters:\n"
            "- run_uv_job() creates a private helper dataset repo in your HF namespace\n"
            "  (for example `anupamagarwal001/hf-cli-jobs-uv-run-scripts`).\n"
            "- that requires a valid logged-in token with write access.\n\n"
            "Fix:\n"
            "1. Run `hf auth login`\n"
            "2. Paste a Hugging Face write token for your personal namespace\n"
            "3. Verify with `hf auth whoami`\n"
            "4. Re-run `python3 training/launch_hf_job.py launch`\n"
        ) from exc

    who_name = who.get("name") or who.get("fullname") or "unknown"
    print(f"hf_auth_ok={who_name}", flush=True)


def launch(args: argparse.Namespace) -> None:
    ensure_hf_auth(args.namespace, args.token)
    job_token = resolve_job_token(args.token)
    script = Path(__file__).resolve().parent / "hf_jobs_smoke.py"
    script_args = [
        "--repo-url",
        args.repo_url,
        "--repo-ref",
        args.repo_ref,
        "--job-output-dir",
        args.output_dir,
        "--model",
        args.model,
        "--repeats-per-task",
        str(args.repeats_per_task),
        "--max-steps",
        str(args.max_steps),
        "--colab-email",
        args.colab_email,
        "--notes",
        args.notes,
    ]
    if args.use_lora:
        script_args.append("--use-lora")
    if args.print_baselines:
        script_args.append("--print-baselines")

    script_args.extend(
        [
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--lora-target-modules",
            args.lora_target_modules,
        ]
    )

    quoted_script_args = [shlex.quote(value) for value in script_args]

    job = run_uv_job(
        script=str(script),
        script_args=quoted_script_args,
        dependencies=DEFAULT_DEPENDENCIES,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        token=job_token,
        env={"PYTHONUNBUFFERED": "1"},
    )
    _print_job(job)
    print("\nNext commands:")
    print(f"python3 training/launch_hf_job.py inspect {job.id}")
    print(f"python3 training/launch_hf_job.py logs {job.id}")


def inspect_job(args: argparse.Namespace) -> None:
    api = HfApi()
    job = api.inspect_job(job_id=args.job_id, namespace=args.namespace, token=args.token)
    payload: dict[str, Any] = {
        "id": job.id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "status": str(job.status),
        "flavor": getattr(job.flavor, "value", job.flavor),
        "url": job.url,
        "command": job.command,
        "arguments": job.arguments,
    }
    print(json.dumps(payload, indent=2))


def logs(args: argparse.Namespace) -> None:
    api = HfApi()
    for line in api.fetch_job_logs(job_id=args.job_id, namespace=args.namespace, token=args.token):
        print(line, end="")


def cancel(args: argparse.Namespace) -> None:
    api = HfApi()
    api.cancel_job(job_id=args.job_id, namespace=args.namespace, token=args.token)
    print(f"cancelled={args.job_id}")


def list_jobs(args: argparse.Namespace) -> None:
    api = HfApi()
    rows = []
    for job in api.list_jobs(namespace=args.namespace, token=args.token):
        rows.append(
            {
                "id": job.id,
                "status": str(job.status),
                "flavor": getattr(job.flavor, "value", job.flavor),
                "url": job.url,
            }
        )
    print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and inspect Hugging Face Jobs for committee training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch", help="Launch the default PM smoke training job.")
    launch_parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    launch_parser.add_argument("--repo-ref", default="main")
    launch_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    launch_parser.add_argument("--token", default=True, help="HF token or True to use local login.")
    launch_parser.add_argument("--flavor", type=_space_hardware, default=SpaceHardware(DEFAULT_FLAVOR))
    launch_parser.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    launch_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    launch_parser.add_argument("--model", default=DEFAULT_MODEL)
    launch_parser.add_argument("--repeats-per-task", type=int, default=2)
    launch_parser.add_argument("--max-steps", type=int, default=4)
    launch_parser.set_defaults(use_lora=True, print_baselines=False)
    launch_parser.add_argument("--use-lora", dest="use_lora", action="store_true")
    launch_parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    launch_parser.add_argument("--lora-r", type=int, default=8)
    launch_parser.add_argument("--lora-alpha", type=int, default=16)
    launch_parser.add_argument("--lora-dropout", type=float, default=0.05)
    launch_parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    launch_parser.add_argument("--colab-email", default="anuagar@groww.in")
    launch_parser.add_argument(
        "--notes",
        default="HF Jobs smoke run for Round 2 committee environment.",
    )
    launch_parser.add_argument("--print-baselines", dest="print_baselines", action="store_true")
    launch_parser.add_argument("--no-print-baselines", dest="print_baselines", action="store_false")
    launch_parser.set_defaults(func=launch)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a single job.")
    inspect_parser.add_argument("job_id")
    inspect_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    inspect_parser.add_argument("--token", default=True)
    inspect_parser.set_defaults(func=inspect_job)

    logs_parser = subparsers.add_parser("logs", help="Stream logs for a single job.")
    logs_parser.add_argument("job_id")
    logs_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    logs_parser.add_argument("--token", default=True)
    logs_parser.set_defaults(func=logs)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running job.")
    cancel_parser.add_argument("job_id")
    cancel_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    cancel_parser.add_argument("--token", default=True)
    cancel_parser.set_defaults(func=cancel)

    list_parser = subparsers.add_parser("list", help="List jobs in the namespace.")
    list_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    list_parser.add_argument("--token", default=True)
    list_parser.set_defaults(func=list_jobs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
