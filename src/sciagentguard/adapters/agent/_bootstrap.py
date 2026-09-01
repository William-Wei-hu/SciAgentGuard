"""Child-process confinement for agent-written analysis code.

This module is executed as a script inside an isolated subprocess. It installs an audit hook and
resource limits before executing the proposed code, so that an agent's ordinary mistakes -- a
stray network call, a shell command, a write outside the working directory, a runaway loop --
fail loudly instead of touching the host.

It is a guardrail against accidents, not a security boundary. Code that is deliberately trying to
escape can do so, for instance through `ctypes`, which is left available because the scientific
stack needs it. Never run untrusted or adversarial code through this module.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

DENIED_EVENT_PREFIXES = (
    "subprocess.",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.spawn",
    "shutil.copyfile",
    "shutil.move",
    "webbrowser.open",
    "urllib.Request",
)
# Creating a socket object is not egress: the scientific stack opens an asyncio event loop on
# import, which allocates one. Only name resolution and non-loopback endpoints are denied.
_DENIED_RESOLUTION_EVENTS = frozenset(
    {
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.sethostname",
    }
)
_ADDRESSED_SOCKET_EVENTS = frozenset({"socket.connect", "socket.bind", "socket.sendto"})
_LOOPBACK_HOSTS = frozenset({"", "localhost", "::1", "0.0.0.0", "::"})
_WRITE_EVENTS = ("open",)
_WRITE_MODES = frozenset("wxa+")


def _is_local_address(address: object) -> bool:
    """Allow loopback and filesystem sockets; treat everything else as egress."""

    if isinstance(address, (str, bytes)):
        # An AF_UNIX path: local by construction.
        return True
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            host = host.decode("utf-8", "replace")
        if not isinstance(host, str):
            return False
        return host in _LOOPBACK_HOSTS or host.startswith("127.")
    return False


class ConfinementError(Exception):
    """A proposed operation was denied by the sandbox policy."""


def _install_audit_hook(workdir: Path) -> None:
    resolved_workdir = workdir.resolve()

    def hook(event: str, args: tuple[Any, ...]) -> None:
        for prefix in DENIED_EVENT_PREFIXES:
            if event.startswith(prefix):
                raise ConfinementError(f"denied operation: {event}")
        if event in _DENIED_RESOLUTION_EVENTS:
            raise ConfinementError(f"denied network name resolution: {event}")
        if event in _ADDRESSED_SOCKET_EVENTS and len(args) >= 2:
            if not _is_local_address(args[1]):
                raise ConfinementError(f"denied non-local network endpoint: {event}")
            return
        if event in _WRITE_EVENTS and len(args) >= 2:
            path, mode = args[0], args[1]
            if not isinstance(mode, str) or not _WRITE_MODES.intersection(mode):
                return
            if path is None or isinstance(path, int):
                return
            try:
                target = Path(os.fsdecode(path)).resolve()
            except (TypeError, ValueError, OSError):
                raise ConfinementError("denied write to an unresolvable path") from None
            if resolved_workdir != target and resolved_workdir not in target.parents:
                raise ConfinementError("denied write outside the sandbox working directory")

    sys.addaudithook(hook)


def _apply_resource_limits(memory_bytes: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - resource is absent only on unsupported platforms
        return
    # RLIMIT_AS is not enforced consistently on every platform, so the parent's wall-clock
    # timeout remains the backstop rather than the primary limit.
    for limit, value in (
        (resource.RLIMIT_AS, memory_bytes),
        (resource.RLIMIT_CPU, cpu_seconds),
    ):
        try:
            hard = resource.getrlimit(limit)[1]
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(limit, (ceiling, hard))
        except (OSError, ValueError):  # pragma: no cover - platform dependent
            continue


def main() -> int:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    workdir = Path(config["workdir"])
    code = Path(config["code_path"]).read_text(encoding="utf-8")

    _apply_resource_limits(int(config["memory_bytes"]), int(config["cpu_seconds"]))
    os.chdir(workdir)
    _install_audit_hook(workdir)

    namespace: dict[str, Any] = {
        # The proposal is executed as a script, so it must see __name__ == "__main__". Anything
        # else silently skips the `if __name__ == "__main__":` guard that most Python authors
        # write, and the run is then misrecorded as the author producing no output.
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "INPUT_PATH": config["input_path"],
        "OUTPUT_PATH": config["output_path"],
    }
    try:
        exec(compile(code, "<agent-proposal>", "exec"), namespace)
    except ConfinementError as error:
        print(f"CONFINEMENT: {error}", file=sys.stderr)
        return 3
    except MemoryError:
        print("MEMORY: allocation refused", file=sys.stderr)
        return 4
    except BaseException as error:  # the child only reports; the parent classifies
        print(f"RUNTIME: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
