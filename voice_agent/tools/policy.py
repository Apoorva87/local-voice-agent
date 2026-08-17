"""Classifies shell commands as auto-runnable or confirmation-required.

The PRD's rule: read-only tools run automatically; writes, sends, purchases
and destructive or system-level actions require confirmation.

This matters more for voice than for text. During benchmarking both candidate
models produced ``lsof -ti:8080 | xargs kill -9`` confidently and correctly
from a casual spoken request -- and speech recognition can mishear. A voice
agent with unguarded shell access is one mis-transcription from a bad day.

The classifier **fails closed**: anything not positively recognised as
read-only requires confirmation.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    READ_ONLY = "read_only"  # runs automatically
    NEEDS_CONFIRMATION = "needs_confirmation"  # must be approved out loud
    BLOCKED = "blocked"  # never runs, confirmation or not


@dataclass(frozen=True)
class Verdict:
    risk: Risk
    reason: str

    @property
    def auto(self) -> bool:
        return self.risk is Risk.READ_ONLY


# Commands that only observe. Anything outside this list is not read-only,
# no matter how harmless it looks.
READ_ONLY_COMMANDS = frozenset(
    {
        "ls", "pwd", "cat", "head", "tail", "wc", "file", "stat", "find",
        "grep", "rg", "which", "whoami", "hostname", "uptime", "date", "cal",
        "df", "du", "free", "ps", "top", "uname", "env", "printenv", "echo",
        "sw_vers", "system_profiler", "vm_stat", "sysctl", "id", "groups",
        "lsof", "netstat", "ifconfig", "dig", "host", "ping", "tree", "diff",
        "sort", "uniq", "cut", "awk", "sed", "jq", "basename", "dirname",
        "realpath", "readlink", "man", "help", "type", "history",
    }
)

# Refused outright: too destructive to be one mis-transcription away, even
# with a spoken "yes".
BLOCKED_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)*/(\s|$)"), "it deletes the filesystem root"),
    (re.compile(r"\bmkfs\b"), "it formats a filesystem"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "it writes directly to a device"),
    (re.compile(r":\(\)\s*\{.*\}\s*;\s*:"), "it is a fork bomb"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/(\s|$)"), "it opens up permissions on the root filesystem"),
    (re.compile(r">\s*/dev/(sd|disk|nvme)"), "it writes directly to a disk device"),
    (re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b"), "it powers off the machine"),
]

# Shell metacharacters that let a benign-looking prefix hide something else.
_CHAINING = re.compile(r"[;&|]|\$\(|`|\n|>>|>")

# Flags that turn an otherwise read-only command into a writer.
_WRITE_FLAGS = {
    "find": re.compile(r"\B-(delete|exec|execdir|ok)\b"),
    "sed": re.compile(r"\B-i\b|--in-place"),
    "awk": re.compile(r"\bprint\s*>"),
}


def classify(command: str) -> Verdict:
    """Decide whether a command may run without asking."""
    text = (command or "").strip()
    if not text:
        return Verdict(Risk.BLOCKED, "the command is empty")

    lowered = text.lower()
    for pattern, reason in BLOCKED_PATTERNS:
        if pattern.search(lowered):
            return Verdict(Risk.BLOCKED, reason)

    # Any chaining, piping, redirection or substitution means the whole
    # command cannot be judged by its first word.
    if _CHAINING.search(text):
        return Verdict(
            Risk.NEEDS_CONFIRMATION,
            "it combines multiple commands or redirects output",
        )

    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return Verdict(Risk.NEEDS_CONFIRMATION, f"it could not be parsed ({exc})")
    if not parts:
        return Verdict(Risk.BLOCKED, "the command is empty")

    # Strip a leading env assignment such as FOO=bar ls
    while parts and "=" in parts[0] and not parts[0].startswith("-"):
        parts = parts[1:]
    if not parts:
        return Verdict(Risk.NEEDS_CONFIRMATION, "it only sets environment variables")

    binary = parts[0].rsplit("/", 1)[-1]
    if binary == "sudo":
        return Verdict(Risk.NEEDS_CONFIRMATION, "it runs as root")
    if binary not in READ_ONLY_COMMANDS:
        return Verdict(Risk.NEEDS_CONFIRMATION, f"{binary} is not a known read-only command")

    write_flag = _WRITE_FLAGS.get(binary)
    if write_flag and write_flag.search(text):
        return Verdict(Risk.NEEDS_CONFIRMATION, f"{binary} is being used to modify files")

    return Verdict(Risk.READ_ONLY, f"{binary} only reads")
