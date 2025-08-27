import argparse
from typing import Sequence
from .runner import run_experiment, RunConfig

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the trading experiment.")
    p.add_argument("url", type=str, help="URL to fetch experiment links from.")
    p.add_argument("model_name", type=str, help="Model name to use for the experiment.")
    p.add_argument("-p", "--production", action="store_true",
                   help="Use real-experiment-runs data folder.")
    p.add_argument("-m", "--message", type=str,
                   help="Notes for the real experiment (required if --production).")
    p.add_argument("--timeout-minutes", type=int, default=105,
                   help="Timeout before exiting (default: 105).")
    p.add_argument("--num-bots", type=int, default=None,
                   help="How many subjects should be bot traders. "
                        "Default: all subjects in the config file.")
    return p

def validate_args(args: argparse.Namespace) -> None:
    if args.production and not args.message:
        raise SystemExit("Error: A message (-m/--message) is required for real experiment runs.")
    if args.num_bots is not None and args.num_bots < 0:
        raise SystemExit("--num-bots must be >= 0")

def args_to_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        url=args.url,
        model_name=args.model_name,
        production=bool(args.production),
        message=args.message,
        timeout_minutes=args.timeout_minutes,
        num_bots=args.num_bots,
    )

def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    cfg = args_to_config(args)
    run_experiment(cfg)