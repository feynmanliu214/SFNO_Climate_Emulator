"""Command-line launcher for SFNO training, inference, and ensemble runs.

This is a thin wrapper over the Makani entry points (``makani.train``,
``makani.inference``, ``makani.ensemble``). It exists to give the common
invocations short, memorable names, to resolve configuration paths up front so
mistakes surface before a job reaches the scheduler, and to handle the
single-node versus MPI launch split in one place.

Any argument the wrapper does not recognise is forwarded verbatim to the
underlying Makani entry point, so the full upstream option surface stays
available:

    sfno-emulator train --config configs/example_sfno.yaml \
        --config-name example_sfno --amp_mode bf16 --multistep_count 1

Use ``--dry-run`` to print the resulting command without executing it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

#: Wrapper sub-command -> Makani module executed as ``python -m <module>``.
_ENTRY_POINTS = {
    "train": "makani.train",
    "infer": "makani.inference",
    "ensemble": "makani.ensemble",
}


def _resolve_config(path: Path) -> Path:
    """Resolve a config path, failing early with a clear message.

    Makani reads the YAML well after distributed wireup, so a typo would
    otherwise only surface once ranks are already allocated.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"config not found: {resolved}")
    return resolved


def _build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """Assemble the full argv for the underlying Makani entry point."""
    module = _ENTRY_POINTS[args.stage]

    cmd: list[str] = []
    if args.nproc > 1:
        launcher = shutil.which("mpirun")
        if launcher is None:
            raise SystemExit(
                "--nproc > 1 requires mpirun on PATH; Makani wires up its "
                "communicators over MPI. Run with --nproc 1 for a single process."
            )
        cmd += [launcher, "-np", str(args.nproc)]

    cmd += [sys.executable, "-u", "-m", module]
    cmd += [f"--yaml_config={args.config}", f"--config={args.config_name}"]

    if args.run_name is not None:
        cmd += [f"--run_num={args.run_name}"]

    cmd += passthrough
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfno-emulator",
        description=(
            "Launch SFNO training, inference, and ensemble runs on top of "
            "NVIDIA Makani. Unrecognised arguments are forwarded to the "
            "underlying Makani entry point."
        ),
        epilog="Full option reference: makani/README.md",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="stage", metavar="STAGE", required=True)

    specs = [
        ("train", "train an SFNO emulator (makani.train)"),
        ("infer", "roll a checkpoint forward and score it (makani.inference)"),
        ("ensemble", "run an ensemble forecast (makani.ensemble)"),
    ]
    for name, help_text in specs:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True, help="path to the YAML config file")
        p.add_argument(
            "--config-name",
            required=True,
            help="name of the configuration block to select inside the YAML file",
        )
        p.add_argument("--run-name", default=None, help="run identifier (Makani --run_num)")
        p.add_argument("--nproc", type=int, default=1, help="processes to launch under mpirun")
        p.add_argument("--dry-run", action="store_true", help="print the command instead of running it")

    return parser


def main(argv: list[str] | None = None) -> int:
    args, passthrough = build_parser().parse_known_args(argv)

    args.config = _resolve_config(args.config)
    if args.nproc < 1:
        raise SystemExit(f"--nproc must be >= 1, got {args.nproc}")

    cmd = _build_command(args, passthrough)

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
