#!/usr/bin/env python3
"""Start a vLLM embedding server with configurable defaults and CLI overrides."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Defaults  (CLI args > env vars > these values)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOG_DIR          = REPO_ROOT / "logs"
DEFAULT_LOG_FILE         = "vllm_embed.log"
DEFAULT_HOST             = "0.0.0.0"
DEFAULT_PORT             = 8081
DEFAULT_MODEL            = "NeuML/pubmedbert-base-embeddings"
DEFAULT_GPU_MEM_UTIL     = 0.15
DEFAULT_DEVICE           = "gpu"
DEFAULT_CPU_DTYPE        = "half"
DEFAULT_TRUST_REMOTE     = True


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def resolve(arg_value: Any, env_name: str, default: Any,
            caster: Callable[[str], Any] | None = None) -> Any:
    """Return arg_value > env var > default, with optional casting."""
    if arg_value is not None:
        return arg_value
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw if caster is None else caster(raw)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Start vLLM OpenAI-compatible embedding server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log-dir",  default=None, help="Directory for log files")
    p.add_argument("--log-file", default=None, help="Log file name")
    p.add_argument("--host",     default=None, help="Server host")
    p.add_argument("--port",     type=int, default=None, help="Server port")
    p.add_argument("--model",    default=None, help="Embedding model name")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help="GPU memory fraction (device=gpu only)")
    p.add_argument("--device", choices=["gpu", "cpu"], default=None,
                   help="Device for embedding server")
    p.add_argument("--cpu-dtype", default=None,
                   help="Dtype passed to vLLM when device=cpu")

    trust = p.add_mutually_exclusive_group()
    trust.add_argument("--trust-remote-code",    dest="trust_remote_code",
                       action="store_true",  help="Enable --trust-remote-code")
    trust.add_argument("--no-trust-remote-code", dest="trust_remote_code",
                       action="store_false", help="Disable --trust-remote-code")
    p.set_defaults(trust_remote_code=None)
    return p


def stream_to_console_and_file(proc: subprocess.Popen[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as fp:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fp.write(line)
    return proc.wait()


def main() -> int:
    args = build_parser().parse_args()

    log_dir = Path(resolve(args.log_dir,  "LOG_DIR",          str(DEFAULT_LOG_DIR),  str)).resolve()
    log_file =     resolve(args.log_file, "EMBED_LOG_FILE",   DEFAULT_LOG_FILE,      str)
    host     =     resolve(args.host,     "VLLM_HOST",        DEFAULT_HOST,          str)
    port     =     resolve(args.port,     "VLLM_EMBED_PORT",  DEFAULT_PORT,          int)
    model    =     resolve(args.model,    "EMBEDDING_MODEL",  DEFAULT_MODEL,         str)
    gpu_mem  =     resolve(args.gpu_memory_utilization,
                           "VLLM_EMBED_GPU_MEM_UTIL", DEFAULT_GPU_MEM_UTIL, float)
    device   =     resolve(args.device,   "EMBED_DEVICE",     DEFAULT_DEVICE,        str).lower()
    cpu_dtype =    resolve(args.cpu_dtype, "CPU_DTYPE",        DEFAULT_CPU_DTYPE,     str)
    trust     =    resolve(args.trust_remote_code,
                           "TRUST_REMOTE_CODE", DEFAULT_TRUST_REMOTE, parse_bool)

    if device not in {"gpu", "cpu"}:
        raise ValueError(f"Invalid device: {device!r}")

    if platform.system() == "Darwin" and device == "gpu":
        print("WARNING: macOS detected — falling back to CPU mode.")
        device = "cpu"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--runner", "pooling",
        "--host",  host,
        "--port",  str(port),
    ]

    if device == "cpu":
        cmd.extend(["--dtype", cpu_dtype])
    else:
        cmd.extend(["--gpu-memory-utilization", str(gpu_mem)])

    if trust:
        cmd.append("--trust-remote-code")

    child_env = os.environ.copy()
    if device == "cpu":
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        if platform.system() == "Darwin":
            child_env["VLLM_PLUGINS"] = ""

    print(f"Starting vLLM embedding server on {host}:{port} (device={device})")
    print(f"Model: {model}")
    print(f"Log:   {log_path}")

    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    try:
        return stream_to_console_and_file(proc, log_path)
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
