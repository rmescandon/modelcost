import argparse
import sys
from .calculator import calculate_cost, list_models, VALID_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmcost")
    parser.add_argument("model", nargs="?", help="Model ID (optional with --list-models)")
    parser.add_argument("input_tokens", nargs="?", type=int)
    parser.add_argument("output_tokens", nargs="?", type=int)
    parser.add_argument(
        "--source",
        default="litellm",
        choices=VALID_SOURCES,
        help="Pricing source (default: litellm)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models for the given source",
    )
    parser.add_argument(
        "--filter",
        default=None,
        metavar="TERM",
        help="Filter model list by substring (e.g. --filter gpt)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # ── Modo listado ──────────────────────────────────────────────
    if args.list_models:
        src = args.source if args.source != "all" else "litellm"
        try:
            models = list_models(src)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.filter:
            models = [m for m in models if args.filter.lower() in m.lower()]

        if args.json:
            import json
            print(json.dumps({"source": src, "count": len(models), "models": models}, indent=2))
        else:
            print(f"Models available in [{src}] ({len(models)} total):\n")
            for m in models:
                print(f"  {m}")
        return

    # ── Modo cálculo ──────────────────────────────────────────────
    if not all([args.model, args.input_tokens is not None, args.output_tokens is not None]):
        parser.error("model, input_tokens and output_tokens are required for cost calculation")

    try:
        result = calculate_cost(args.model, args.input_tokens, args.output_tokens, source=args.source)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
        return

    if result.single_source:
        s = result.sources[0]
        if s.available:
            print(f"${s.total_cost_usd:.6f}")
        else:
            print(f"unavailable — {s.error}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Model: {result.model}  ({result.input_tokens} in / {result.output_tokens} out)\n")
        for s in result.sources:
            if s.available:
                print(f"  [{s.source:<12}] ${s.total_cost_usd:.6f} USD")
            else:
                print(f"  [{s.source:<12}] unavailable — {s.error}")
