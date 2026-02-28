#!/usr/bin/env python3
"""Run a local Codex specialist workflow with lead synthesis.

This script launches specialist Codex agents in parallel, runs an optional
peer cross-review round, and then optionally runs a lead agent that can apply
code changes in the workspace. Artifacts are saved under:
.codex-team/runs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Role:
    name: str
    objective: str
    sandbox: str


@dataclass(frozen=True)
class EvidenceCommand:
    name: str
    command: str


@dataclass(frozen=True)
class SpecialistResult:
    role_name: str
    report_path: Path
    exit_code: int
    detail: str
    is_dry_run: bool


SPECIALIST_ROLES: dict[str, Role] = {
    "architect": Role(
        name="architect",
        objective=(
            "Define the highest-leverage implementation approach, identify "
            "likely impacted files, and call out tradeoffs."
        ),
        sandbox="read-only",
    ),
    "reviewer": Role(
        name="reviewer",
        objective=(
            "Find likely bugs, regressions, edge cases, and missing tests in "
            "the proposed approach."
        ),
        sandbox="read-only",
    ),
    "tester": Role(
        name="tester",
        objective=(
            "Design a targeted verification plan, including concrete commands "
            "and scenarios to validate behavior."
        ),
        sandbox="read-only",
    ),
}

LEAD_ROLE = Role(
    name="lead",
    objective=(
        "Implement the task using specialist reports as input. Make final "
        "decisions, edit files as needed, run checks, and report outcomes."
    ),
    sandbox="workspace-write",
)

PROMPT_TRIGGER_PATTERN = re.compile(r"agent\s+team", re.IGNORECASE)


def _resolve_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task.strip()
    if not sys.stdin.isatty():
        stdin_payload = sys.stdin.read().strip()
        if stdin_payload:
            return stdin_payload
    raise SystemExit("Task is required. Pass --task or pipe text on stdin.")


def _check_codex_installed() -> None:
    if shutil.which("codex") is None:
        raise SystemExit("`codex` CLI not found in PATH.")


def _supports_terminal_windows() -> bool:
    return platform.system() == "Darwin" and shutil.which("osascript") is not None


def _quote_applescript_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _launch_terminal_window(command: str) -> tuple[bool, str]:
    script = f'tell application "Terminal" to do script {_quote_applescript_string(command)}'
    proc = subprocess.run(
        ["osascript", "-e", 'tell application "Terminal" to activate', "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "failed to launch Terminal").strip()


def _sanitize_task_for_prompt(task: str) -> str:
    """Avoid triggering AGENTS.md keyword routing inside nested specialists."""
    return PROMPT_TRIGGER_PATTERN.sub("agent-team", task)


def _trim_text(text: str, *, max_chars: int) -> str:
    body = (text or "").strip()
    if len(body) <= max_chars:
        return body
    tail = body[-max_chars:]
    return f"...[truncated to last {max_chars} chars]...\n{tail}"


def _render_shared_evidence_prompt_block(shared_evidence: str) -> str:
    if not shared_evidence.strip():
        return "(no shared evidence collected)"
    return _trim_text(shared_evidence, max_chars=14000)


def _build_specialist_prompt(task: str, role: Role, shared_evidence: str) -> str:
    sanitized_task = _sanitize_task_for_prompt(task)
    lines = [
        "You are a Codex specialist in a coordinated debugging workflow.",
        "",
        "Phase: initial specialist analysis",
        f"Role: {role.name}",
        f"Objective: {role.objective}",
        "",
        "Rules:",
        "- Stay in read-only analysis mode.",
        "- Focus only on this role's objective.",
        "- Be concrete about files, risks, and commands when relevant.",
        "",
        "Shared evidence collected once for all specialists:",
        _render_shared_evidence_prompt_block(shared_evidence),
        "",
        "Return this exact structure:",
        "## Findings",
        "- ...",
        "## Recommendations",
        "- ...",
        "## Risks",
        "- ...",
        "",
        "User task:",
        sanitized_task,
    ]
    return "\n".join(lines).strip()


def _build_cross_review_prompt(
    task: str,
    role: Role,
    shared_evidence: str,
    own_report: str,
    peer_reports: dict[str, str],
) -> str:
    sanitized_task = _sanitize_task_for_prompt(task)
    peer_sections: list[str] = []
    for peer_name in sorted(peer_reports):
        if peer_name == role.name:
            continue
        peer_sections.append(
            textwrap.dedent(
                f"""
                ### {peer_name}
                {_trim_text(peer_reports[peer_name], max_chars=4500)}
                """
            ).strip()
        )
    peers_blob = "\n\n".join(peer_sections) if peer_sections else "(no peer reports)"
    lines = [
        "You are a Codex specialist in a coordinated debugging workflow.",
        "",
        "Phase: parallel peer cross-review",
        f"Role: {role.name}",
        f"Objective: {role.objective}",
        "",
        "Rules:",
        "- Stay in read-only analysis mode.",
        "- Cross-check your initial analysis with peer findings.",
        "- Keep conclusions concrete and implementation-oriented.",
        "",
        "Shared evidence collected once for all specialists:",
        _render_shared_evidence_prompt_block(shared_evidence),
        "",
        "Your initial report:",
        _trim_text(own_report, max_chars=5000),
        "",
        "Peer specialist summaries:",
        peers_blob,
        "",
        "Return this exact structure:",
        "## Findings",
        "- ...",
        "## Recommendations",
        "- ...",
        "## Risks",
        "- ...",
        "",
        "User task:",
        sanitized_task,
    ]
    return "\n".join(lines).strip()


def _build_lead_prompt(
    task: str,
    specialist_reports: dict[str, str],
    shared_evidence: str,
) -> str:
    sections: list[str] = []
    for role_name in sorted(specialist_reports):
        sections.append(
            textwrap.dedent(
                f"""
                ### {role_name}
                {specialist_reports[role_name].strip()}
                """
            ).strip()
        )
    reports_blob = "\n\n".join(sections) if sections else "(none)"
    sanitized_task = _sanitize_task_for_prompt(task)
    lines = [
        "You are the lead Codex agent coordinating specialist reports.",
        "",
        "Objective:",
        LEAD_ROLE.objective,
        "",
        "Required behavior:",
        "- Use specialist reports as input, but make final technical decisions.",
        "- Make necessary code changes directly in the workspace.",
        "- Run relevant validation commands.",
        "- End with a concise summary of changes, tests run, and residual risks.",
        "",
        "Shared evidence collected once for all specialists:",
        _render_shared_evidence_prompt_block(shared_evidence),
        "",
        "User task:",
        sanitized_task,
        "",
        "Specialist reports:",
        reports_blob,
    ]
    return "\n".join(lines).strip()


def _read_report(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _default_evidence_commands(repo_root: Path) -> list[EvidenceCommand]:
    venv_python = repo_root / ".venv" / "bin" / "python"
    python_cmd = (
        shlex.quote(str(venv_python))
        if venv_python.exists()
        else shlex.quote(sys.executable or "python3")
    )
    return [
        EvidenceCommand("pytest_quiet", f"{python_cmd} -m pytest -q"),
        EvidenceCommand(
            "paper_diagnose_once",
            f"{python_cmd} main.py run paper once --diagnose",
        ),
        EvidenceCommand(
            "log_errors_tail",
            'rg -n "ERROR|Traceback|Exception" logs/tradingbot.log | tail -n 200',
        ),
    ]


def _run_shell_command(
    *,
    command: str,
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
        return (completed.returncode, completed.stdout or "", completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        suffix = f"\ncommand timed out after {int(timeout_seconds)}s"
        return (124, stdout, f"{stderr}{suffix}".strip())


def _collect_shared_evidence(
    *,
    repo_root: Path,
    run_dir: Path,
    timeout_seconds: int,
    dry_run: bool,
) -> str:
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    commands = _default_evidence_commands(repo_root)
    sections: list[str] = [
        "## Shared Debug Evidence",
        "",
        f"Generated at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
    ]
    for idx, item in enumerate(commands, start=1):
        prefix = f"{idx:02d}_{item.name}"
        stdout_path = evidence_dir / f"{prefix}.stdout.log"
        stderr_path = evidence_dir / f"{prefix}.stderr.log"
        exitcode_path = evidence_dir / f"{prefix}.exitcode"
        cmd_path = evidence_dir / f"{prefix}.cmd.txt"
        cmd_path.write_text(item.command + "\n", encoding="utf-8")
        if dry_run:
            exit_code = 0
            stdout = ""
            stderr = f"[dry-run] {item.command}"
        else:
            exit_code, stdout, stderr = _run_shell_command(
                command=item.command,
                cwd=repo_root,
                timeout_seconds=timeout_seconds,
            )
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        exitcode_path.write_text(str(exit_code), encoding="utf-8")
        status = "ok" if exit_code == 0 else f"failed({exit_code})"
        stdout_snippet = _trim_text(stdout, max_chars=2500) if stdout.strip() else "(empty)"
        stderr_snippet = _trim_text(stderr, max_chars=1000) if stderr.strip() else "(empty)"
        sections.extend(
            [
                f"### {item.name} [{status}]",
                f"Command: `{item.command}`",
                "stdout:",
                "```text",
                stdout_snippet,
                "```",
                "stderr:",
                "```text",
                stderr_snippet,
                "```",
                "",
            ]
        )
    summary = "\n".join(sections).strip()
    (evidence_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    return summary


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_pid(pid: int, *, grace_seconds: float = 5.0) -> bool:
    if pid <= 0:
        return False
    if not _process_exists(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + max(0.1, float(grace_seconds))
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return False
    return not _process_exists(pid)


def _run_codex(
    *,
    prompt: str,
    cwd: Path,
    output_file: Path,
    sandbox: str,
    model: str | None,
    dry_run: bool,
    timeout_seconds: int,
    run_in_terminal_window: bool,
) -> tuple[int, str, bool]:
    cmd: list[str] = [
        "codex",
        "-a",
        "never",
        "exec",
        "--cd",
        str(cwd),
        "--sandbox",
        sandbox,
        "--skip-git-repo-check",
        "-o",
        str(output_file),
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    if dry_run:
        return (0, " ".join(cmd), True)

    if run_in_terminal_window:
        status_file = output_file.with_suffix(output_file.suffix + ".exitcode")
        stderr_file = output_file.with_suffix(output_file.suffix + ".stderr.log")
        pid_file = output_file.with_suffix(output_file.suffix + ".pid")
        for path in (status_file, stderr_file, pid_file):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
        shell_cmd = (
            f"cd {shlex.quote(str(cwd))}; "
            "export CODEX_TEAM_RUNNER=1; "
            "{ "
            + quoted_cmd
            + f" 2> {shlex.quote(str(stderr_file))}; "
            + f"printf '%s' $? > {shlex.quote(str(status_file))}; "
            + "} & "
            + "runner_pid=$!; "
            + f"printf '%s' \"$runner_pid\" > {shlex.quote(str(pid_file))}; "
            + "wait \"$runner_pid\""
        )
        ok, launch_error = _launch_terminal_window(shell_cmd)
        if not ok:
            return (2, launch_error, False)

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while time.monotonic() < deadline:
            if status_file.exists():
                try:
                    exit_code = int((status_file.read_text(encoding="utf-8") or "1").strip())
                except ValueError:
                    exit_code = 1
                stderr = ""
                if stderr_file.exists():
                    stderr = stderr_file.read_text(encoding="utf-8").strip()
                return (exit_code, stderr, False)
            time.sleep(0.5)

        timeout_detail = (
            f"codex exec timed out after {int(timeout_seconds)}s (terminal window mode)"
        )
        terminated = False
        pid_value: int | None = None
        if pid_file.exists():
            raw_pid = pid_file.read_text(encoding="utf-8").strip()
            try:
                pid_value = int(raw_pid)
            except ValueError:
                pid_value = None
            if pid_value is not None:
                terminated = _terminate_pid(pid_value)
        if pid_value is not None:
            suffix = "terminated" if terminated else "failed to terminate"
            timeout_detail = f"{timeout_detail}; {suffix} pid={pid_value}"
        else:
            timeout_detail = f"{timeout_detail}; pid unavailable"
        return (124, timeout_detail, False)

    env = os.environ.copy()
    env["CODEX_TEAM_RUNNER"] = "1"
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
        stderr = completed.stderr.strip()
        return (completed.returncode, stderr, False)
    except subprocess.TimeoutExpired:
        return (
            124,
            f"codex exec timed out after {int(timeout_seconds)}s",
            False,
        )


def _run_specialist_prompt(
    *,
    role: Role,
    prompt: str,
    repo_root: Path,
    prompt_path: Path,
    report_path: Path,
    model: str | None,
    dry_run: bool,
    timeout_seconds: int,
    run_in_terminal_window: bool,
) -> SpecialistResult:
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    exit_code, detail, is_dry_run = _run_codex(
        prompt=prompt,
        cwd=repo_root,
        output_file=report_path,
        sandbox=role.sandbox,
        model=model,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
        run_in_terminal_window=run_in_terminal_window,
    )
    if is_dry_run:
        report_path.write_text(
            textwrap.dedent(
                f"""
                ## Findings
                - Dry run only; command not executed.
                ## Recommendations
                - Execute without `--dry-run` to generate specialist analysis.
                ## Risks
                - No specialist report content was generated in dry-run mode.

                Command:
                `{detail}`
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
    return SpecialistResult(role.name, report_path, exit_code, detail, is_dry_run)


def _run_specialist_phase(
    *,
    roles: list[Role],
    prompt_by_role: dict[str, str],
    prompt_name_by_role: dict[str, str],
    report_path_by_role: dict[str, Path],
    repo_root: Path,
    model: str | None,
    dry_run: bool,
    timeout_seconds: int,
    run_in_terminal_window: bool,
) -> list[SpecialistResult]:
    results: list[SpecialistResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(roles)) as executor:
        futures = [
            executor.submit(
                _run_specialist_prompt,
                role=role,
                prompt=prompt_by_role[role.name],
                repo_root=repo_root,
                prompt_path=Path(prompt_name_by_role[role.name]),
                report_path=report_path_by_role[role.name],
                model=model,
                dry_run=dry_run,
                timeout_seconds=timeout_seconds,
                run_in_terminal_window=run_in_terminal_window,
            )
            for role in roles
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.role_name)


def _parse_roles(raw_roles: str) -> list[Role]:
    requested = [part.strip().lower() for part in raw_roles.split(",") if part.strip()]
    if not requested:
        raise SystemExit("No specialist roles provided.")
    roles: list[Role] = []
    seen: set[str] = set()
    for role_name in requested:
        if role_name in seen:
            raise SystemExit(f"Duplicate role '{role_name}' is not allowed.")
        seen.add(role_name)
        role = SPECIALIST_ROLES.get(role_name)
        if role is None:
            known = ", ".join(sorted(SPECIALIST_ROLES))
            raise SystemExit(f"Unknown role '{role_name}'. Known roles: {known}")
        roles.append(role)
    return roles


def _collect_specialist_reports(
    specialist_results: Iterable[SpecialistResult],
) -> tuple[dict[str, str], list[str]]:
    specialist_reports: dict[str, str] = {}
    failures: list[str] = []
    for result in sorted(specialist_results, key=lambda item: item.role_name):
        report = _read_report(result.report_path)
        specialist_reports[result.role_name] = report
        if result.exit_code != 0:
            failures.append(f"{result.role_name}: exit code {result.exit_code}")
        if not report.strip():
            failures.append(f"{result.role_name}: empty report")
    return specialist_reports, failures


def _print_specialist_results(specialist_results: Iterable[SpecialistResult]) -> None:
    for result in sorted(specialist_results, key=lambda item: item.role_name):
        status = "ok" if result.exit_code == 0 else f"failed({result.exit_code})"
        print(f"[specialist:{result.role_name}] {status} -> {result.report_path}")
        if result.detail:
            label = "dry-run-cmd" if result.is_dry_run else "stderr"
            print(f"[specialist:{result.role_name}:{label}] {result.detail}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Codex specialist-team workflow locally.",
    )
    parser.add_argument(
        "--task",
        help="Task description for the team. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--roles",
        default="architect,reviewer,tester",
        help="Comma-separated specialist roles.",
    )
    parser.add_argument(
        "--model",
        help="Model for specialist roles.",
    )
    parser.add_argument(
        "--lead-model",
        help="Model for lead role (defaults to --model).",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root for codex execution.",
    )
    parser.add_argument(
        "--skip-lead",
        action="store_true",
        help="Run specialists only.",
    )
    parser.add_argument(
        "--skip-cross-review",
        action="store_true",
        help="Skip the specialist peer cross-review phase.",
    )
    parser.add_argument(
        "--skip-shared-evidence",
        action="store_true",
        help="Skip pre-collection of shared debug evidence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print underlying codex commands without executing them.",
    )
    parser.add_argument(
        "--specialist-timeout-seconds",
        type=int,
        default=300,
        help="Timeout in seconds for each specialist run.",
    )
    parser.add_argument(
        "--lead-timeout-seconds",
        type=int,
        default=900,
        help="Timeout in seconds for the lead run.",
    )
    parser.add_argument(
        "--evidence-timeout-seconds",
        type=int,
        default=900,
        help="Timeout in seconds for each shared-evidence command.",
    )
    parser.add_argument(
        "--spawn-terminal-windows",
        action="store_true",
        help=(
            "Run each codex sub-agent in its own macOS Terminal window while still "
            "executing specialists in parallel."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _check_codex_installed()
    task = _resolve_task(args)
    repo_root = Path(args.repo_root).resolve()
    roles = _parse_roles(args.roles)
    use_terminal_windows = bool(args.spawn_terminal_windows)
    if use_terminal_windows and not _supports_terminal_windows():
        print(
            "[warning] --spawn-terminal-windows requested, but this host does not "
            "support macOS Terminal automation. Falling back to inline mode."
        )
        use_terminal_windows = False

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repo_root / ".codex-team" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.txt").write_text(task + "\n", encoding="utf-8")

    if args.skip_shared_evidence:
        shared_evidence = "Shared evidence collection skipped by --skip-shared-evidence."
    else:
        print("[phase] collecting shared debug evidence")
        shared_evidence = _collect_shared_evidence(
            repo_root=repo_root,
            run_dir=run_dir,
            timeout_seconds=args.evidence_timeout_seconds,
            dry_run=args.dry_run,
        )

    print("[phase] specialist initial pass")
    round1_prompts = {
        role.name: _build_specialist_prompt(task, role, shared_evidence) for role in roles
    }
    round1_prompt_paths = {
        role.name: str(run_dir / "prompts" / f"{role.name}.round1.txt") for role in roles
    }
    round1_report_paths = {
        role.name: run_dir / "specialists" / "round1" / f"{role.name}.md" for role in roles
    }
    round1_results = _run_specialist_phase(
        roles=roles,
        prompt_by_role=round1_prompts,
        prompt_name_by_role=round1_prompt_paths,
        report_path_by_role=round1_report_paths,
        repo_root=repo_root,
        model=args.model,
        dry_run=args.dry_run,
        timeout_seconds=args.specialist_timeout_seconds,
        run_in_terminal_window=use_terminal_windows,
    )
    _print_specialist_results(round1_results)
    round1_reports, round1_failures = _collect_specialist_reports(round1_results)
    if round1_failures:
        print("[error] initial specialist pass failed:")
        for failure in round1_failures:
            print(f"  - {failure}")
        print(f"[done] Artifacts: {run_dir}")
        return 1

    final_specialist_reports = round1_reports
    final_specialist_results = round1_results
    if not args.skip_cross_review:
        print("[phase] specialist peer cross-review")
        round2_prompts: dict[str, str] = {}
        round2_prompt_paths: dict[str, str] = {}
        round2_report_paths: dict[str, Path] = {}
        for role in roles:
            own_report = round1_reports.get(role.name, "")
            round2_prompts[role.name] = _build_cross_review_prompt(
                task=task,
                role=role,
                shared_evidence=shared_evidence,
                own_report=own_report,
                peer_reports=round1_reports,
            )
            round2_prompt_paths[role.name] = str(
                run_dir / "prompts" / f"{role.name}.cross_review.txt"
            )
            round2_report_paths[role.name] = run_dir / "specialists" / f"{role.name}.md"
        final_specialist_results = _run_specialist_phase(
            roles=roles,
            prompt_by_role=round2_prompts,
            prompt_name_by_role=round2_prompt_paths,
            report_path_by_role=round2_report_paths,
            repo_root=repo_root,
            model=args.model,
            dry_run=args.dry_run,
            timeout_seconds=args.specialist_timeout_seconds,
            run_in_terminal_window=use_terminal_windows,
        )
        _print_specialist_results(final_specialist_results)
        final_specialist_reports, final_failures = _collect_specialist_reports(
            final_specialist_results
        )
        if final_failures:
            print("[error] cross-review specialist pass failed:")
            for failure in final_failures:
                print(f"  - {failure}")
            print(f"[done] Artifacts: {run_dir}")
            return 1
    else:
        for role in roles:
            final_path = run_dir / "specialists" / f"{role.name}.md"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(
                round1_reports.get(role.name, "").strip() + "\n",
                encoding="utf-8",
            )

    if args.skip_lead:
        print(f"[done] Specialist phase complete. Artifacts: {run_dir}")
        return 0

    lead_prompt = _build_lead_prompt(task, final_specialist_reports, shared_evidence)
    lead_prompt_path = run_dir / "prompts" / "lead.txt"
    lead_output_path = run_dir / "lead.md"
    lead_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    lead_prompt_path.write_text(lead_prompt + "\n", encoding="utf-8")

    lead_model = args.lead_model or args.model
    lead_exit, lead_detail, lead_is_dry_run = _run_codex(
        prompt=lead_prompt,
        cwd=repo_root,
        output_file=lead_output_path,
        sandbox=LEAD_ROLE.sandbox,
        model=lead_model,
        dry_run=args.dry_run,
        timeout_seconds=args.lead_timeout_seconds,
        run_in_terminal_window=use_terminal_windows,
    )
    lead_status = "ok" if lead_exit == 0 else f"failed({lead_exit})"
    print(f"[lead] {lead_status} -> {lead_output_path}")
    if lead_detail:
        label = "dry-run-cmd" if lead_is_dry_run else "stderr"
        print(f"[lead:{label}] {lead_detail}")
    print(f"[done] Artifacts: {run_dir}")
    return 0 if lead_exit == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
