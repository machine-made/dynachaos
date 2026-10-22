"""Command-line interface for dynachaos pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynachaos import __version__
from dynachaos.pipelines.registry import get_section, list_sections
from dynachaos.pipelines.runner import (
    inspect_section_artifacts,
    run_all,
    run_section,
    validate_section_cache,
    validate_section_outputs,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynachaos", description="Run dynachaos paper pipelines")
    parser.add_argument("--version", action="version", version=f"dynachaos {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available section IDs")

    run = sub.add_parser("run", help="Run one section or all sections")
    run.add_argument(
        "target",
        help="Section ID (for example sec02_circle_map) or all",
    )
    run.add_argument(
        "--profile",
        choices=("paper", "smoke"),
        default="paper",
        help="paper: allow compute; smoke: require precomputed caches",
    )
    run.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base output directory (defaults to ./figures)",
    )
    run.add_argument(
        "--recompute",
        action="store_true",
        help="Delete existing outputs for target sections before running",
    )
    run.add_argument(
        "--timing-ledger",
        type=Path,
        default=None,
        help="Append per-module timing events to this JSONL file",
    )

    style = sub.add_parser("style", help="Inspect and render plotting style themes")
    style_sub = style.add_subparsers(dest="style_command", required=True)
    style_sub.add_parser("list", help="List available style themes")

    preview = style_sub.add_parser("preview", help="Render style preview image(s)")
    preview.add_argument(
        "--theme",
        default=None,
        help="Render only one theme. Omit to render all themes.",
    )
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/style/themes"),
        help="Directory for generated preview PNG files.",
    )

    verify = sub.add_parser("verify", help="Validate cached artifacts without running modules")
    verify_sub = verify.add_subparsers(dest="verify_command", required=True)
    for name in ("caches", "outputs"):
        verify_cmd = verify_sub.add_parser(name, help=f"Validate section {name}")
        verify_cmd.add_argument(
            "target",
            nargs="?",
            default="all",
            help="Section ID or all",
        )
        verify_cmd.add_argument(
            "--output-root",
            type=Path,
            default=None,
            help="Base output directory (defaults to ./figures)",
        )

    inspect = sub.add_parser("inspect", help="Inspect pipeline contracts without running modules")
    inspect_sub = inspect.add_subparsers(dest="inspect_command", required=True)
    inspect_section = inspect_sub.add_parser("section", help="Inspect one section")
    inspect_section.add_argument("section_id", help="Section ID")
    inspect_section.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base output directory (defaults to ./figures)",
    )

    analyze = sub.add_parser("analyze", help="Run a JSONC-configured signal analysis workflow")
    analyze.add_argument("config", type=Path, help="Path to workflow JSONC config")

    return parser


def _selected_sections(parser: argparse.ArgumentParser, target: str) -> tuple[str, ...]:
    if target == "all":
        return list_sections()
    if target not in set(list_sections()):
        parser.error(f"Unknown target '{target}'. Use 'dynachaos list' for valid sections.")
    return (target,)


def _require_section(parser: argparse.ArgumentParser, section_id: str) -> str:
    if section_id not in set(list_sections()):
        parser.error(f"Unknown target '{section_id}'. Use 'dynachaos list' for valid sections.")
    return section_id


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for section_id in list_sections():
            spec = get_section(section_id)
            modules = ",".join(spec.modules)
            print(f"{section_id}\t{modules}")
        return 0

    if args.command == "style":
        from dynachaos.utils.style import available_themes, save_theme_previews, theme_description

        if args.style_command == "list":
            for theme in available_themes():
                print(f"{theme}\t{theme_description(theme)}")
            return 0

        if args.theme is not None:
            valid = available_themes()
            if args.theme not in valid:
                parser.error(f"invalid choice: '{args.theme}' (choose from {', '.join(valid)})")
        selected = None if args.theme is None else [args.theme]
        for path in save_theme_previews(args.output_dir, themes=selected):
            print(path)
        return 0

    if args.command == "verify":
        sections = _selected_sections(parser, args.target)
        validator = (
            validate_section_cache if args.verify_command == "caches" else validate_section_outputs
        )
        label = "caches" if args.verify_command == "caches" else "outputs"
        try:
            for section_id in sections:
                paths = validator(section_id, output_root=args.output_root)
                print(f"[{section_id}] {len(paths)} {label} ok")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    if args.command == "inspect":
        section_id = _require_section(parser, args.section_id)
        spec = get_section(section_id)
        print(f"section\t{section_id}")
        print(f"modules\t{','.join(spec.modules)}")
        for item in inspect_section_artifacts(section_id, output_root=args.output_root):
            required = ",".join(item.required_keys) if item.required_keys else "-"
            print(f"{item.role}\t{item.status}\t{item.path}\t{item.detail}\t{required}")
        return 0

    if args.command == "analyze":
        from dynachaos.workflow import WorkflowError, run_workflow

        try:
            paths = run_workflow(args.config)
        except (WorkflowError, ValueError) as exc:
            print(f"dynachaos analyze: {exc}", file=sys.stderr)
            return 2
        print(f"output_dir\t{paths['output_dir']}")
        print(f"results\t{paths['results']}")
        print(f"metadata\t{paths['metadata']}")
        print(f"summary\t{paths['summary']}")
        return 0

    target = args.target

    if target == "all":
        results = run_all(
            output_root=args.output_root,
            profile=args.profile,
            recompute=args.recompute,
            timing_ledger=args.timing_ledger,
        )
        for section_id, paths in results.items():
            print(f"[{section_id}] {len(paths)} outputs")
        return 0

    if target not in set(list_sections()):
        parser.error(f"Unknown target '{target}'. Use 'dynachaos list' for valid sections.")

    paths = run_section(
        target,
        output_root=args.output_root,
        profile=args.profile,
        recompute=args.recompute,
        timing_ledger=args.timing_ledger,
    )
    print(f"[{target}] {len(paths)} outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
