import sys
import json as json_mod
import click
from .calculator import calculate_cost, list_models, VALID_SOURCES


class DefaultCostGroup(click.Group):
    """A Group that falls back to the 'cost' command when no subcommand is matched."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # If the first non-option token is not a known subcommand, prepend 'cost'
        for i, arg in enumerate(args):
            if not arg.startswith("-"):
                if arg not in self.commands:
                    args = ["cost"] + args
                break
        return super().parse_args(ctx, args)


@click.group(cls=DefaultCostGroup)
def main() -> None:
    """Calculate LLM API call costs from token usage."""


@main.command("cost")
@click.argument("model")
@click.argument("input_tokens", type=int)
@click.argument("output_tokens", type=int)
@click.option(
    "--source",
    default="litellm",
    show_default=True,
    type=click.Choice(VALID_SOURCES),
    help="Pricing source.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def cost_cmd(
    model: str, input_tokens: int, output_tokens: int, source: str, as_json: bool
) -> None:
    """Calculate the cost for MODEL with INPUT_TOKENS and OUTPUT_TOKENS."""
    # ── Calculation mode ──────────────────────────────────────────────
    try:
        result = calculate_cost(model, input_tokens, output_tokens, source=source)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json_mod.dumps(result.to_dict(), indent=2))
        return

    if result.single_source:
        s = result.sources[0]
        if s.available:
            click.echo(f"${s.total_cost_usd:.6f}")
        else:
            click.echo(f"unavailable — {s.error}", err=True)
            sys.exit(1)
    else:
        click.echo(
            f"Model: {result.model}  ({result.input_tokens} in / {result.output_tokens} out)\n"
        )
        for s in result.sources:
            if s.available:
                click.echo(f"  [{s.source:<12}] ${s.total_cost_usd:.6f} USD")
            else:
                click.echo(f"  [{s.source:<12}] unavailable — {s.error}")


@main.command("models")
@click.option(
    "--source",
    default="litellm",
    show_default=True,
    type=click.Choice(("litellm", "openrouter", "tokencost")),
    help="Pricing source.",
)
@click.option(
    "--filter",
    "filter_term",
    default=None,
    metavar="TERM",
    help="Filter by substring (e.g. gpt).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def models_cmd(source: str, filter_term: str | None, as_json: bool) -> None:
    """List available models for the given source."""
    # ── List mode ──────────────────────────────────────────────
    try:
        models = list_models(source)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if filter_term:
        models = [m for m in models if filter_term.lower() in m.lower()]

    if as_json:
        click.echo(
            json_mod.dumps(
                {"source": source, "count": len(models), "models": models}, indent=2
            )
        )
    else:
        click.echo(f"Models available in [{source}] ({len(models)} total):\n")
        for m in models:
            click.echo(f"  {m}")
