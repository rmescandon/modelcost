import json
from unittest.mock import patch

from click.testing import CliRunner

from modelcost.cli import main
from modelcost.models import CostResult, SourceCost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_result(model="gpt-4o", cost=0.00105, source="litellm"):
    return CostResult(
        model=model,
        input_tokens=100,
        output_tokens=50,
        sources=[
            SourceCost(
                source=source,
                total_cost_usd=cost,
                price_per_million_input=3.0,
                price_per_million_output=15.0,
            )
        ],
        single_source=True,
    )


def _unavailable_result(model="gpt-4o", source="litellm"):
    return CostResult(
        model=model,
        input_tokens=100,
        output_tokens=50,
        sources=[
            SourceCost(
                source=source,
                total_cost_usd=None,
                price_per_million_input=None,
                price_per_million_output=None,
                error="Model not found",
            )
        ],
        single_source=True,
    )


def _multi_result():
    return CostResult(
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        sources=[
            SourceCost("litellm", 0.00105, 3.0, 15.0),
            SourceCost("openrouter", 0.00110, 3.2, 16.0),
            SourceCost("tokencost", None, None, None, error="not found"),
        ],
        single_source=False,
    )


def _multi_result_with_extras():
    return CostResult(
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        sources=[
            SourceCost("litellm", 0.005, 3.0, 15.0),
            SourceCost("openrouter", 0.006, 3.2, 16.0),
        ],
        single_source=False,
        cached_input_tokens=200,
        cache_creation_input_tokens=100,
        reasoning_tokens=150,
    )


# ---------------------------------------------------------------------------
# cost command
# ---------------------------------------------------------------------------


class TestCostCommand:
    def test_displays_cost_with_six_decimal_places(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result(cost=0.00105)
        ):
            result = runner.invoke(main, ["cost", "gpt-4o", "100", "50"])

        assert result.exit_code == 0
        assert "$0.001050" in result.output

    def test_default_subcommand_redirects_to_cost(self):
        """Invoking without 'cost' keyword must still work (DefaultCostGroup)."""
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result(cost=0.00105)
        ):
            result = runner.invoke(main, ["gpt-4o", "100", "50"])

        assert result.exit_code == 0
        assert "$0.001050" in result.output

    def test_json_flag_outputs_valid_json(self):
        runner = CliRunner()
        with patch("modelcost.cli.calculate_cost", return_value=_single_result()):
            result = runner.invoke(main, ["cost", "gpt-4o", "100", "50", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model"] == "gpt-4o"
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert "costs" in data

    def test_unavailable_source_prints_error_and_exits_1(self):
        runner = CliRunner()
        with patch("modelcost.cli.calculate_cost", return_value=_unavailable_result()):
            result = runner.invoke(main, ["cost", "gpt-4o", "100", "50"])

        assert result.exit_code == 1
        assert "unavailable" in result.output

    def test_calculate_cost_exception_exits_1(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", side_effect=ValueError("Invalid source")
        ):
            result = runner.invoke(main, ["cost", "gpt-4o", "100", "50"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_source_all_shows_multi_source_table(self):
        runner = CliRunner()
        with patch("modelcost.cli.calculate_cost", return_value=_multi_result()):
            result = runner.invoke(
                main, ["cost", "gpt-4o", "100", "50", "--source", "all"]
            )

        assert result.exit_code == 0
        assert "litellm" in result.output
        assert "openrouter" in result.output
        assert "tokencost" in result.output

    def test_unavailable_source_shown_in_multi_mode(self):
        runner = CliRunner()
        with patch("modelcost.cli.calculate_cost", return_value=_multi_result()):
            result = runner.invoke(
                main, ["cost", "gpt-4o", "100", "50", "--source", "all"]
            )

        assert "unavailable" in result.output

    def test_multi_source_header_shows_extended_tokens(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost",
            return_value=_multi_result_with_extras(),
        ):
            result = runner.invoke(
                main, ["cost", "gpt-4o", "1000", "500", "--source", "all"]
            )

        assert result.exit_code == 0
        assert "200 cached" in result.output
        assert "100 cache-create" in result.output
        assert "150 reasoning" in result.output

    def test_multi_source_header_omits_zero_extended_tokens(self):
        runner = CliRunner()
        with patch("modelcost.cli.calculate_cost", return_value=_multi_result()):
            result = runner.invoke(
                main, ["cost", "gpt-4o", "100", "50", "--source", "all"]
            )

        assert result.exit_code == 0
        assert "100 in / 50 out)" in result.output
        assert "cached" not in result.output
        assert "reasoning" not in result.output

    def test_source_option_is_forwarded(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost",
            return_value=_single_result(source="openrouter"),
        ) as mock_calc:
            runner.invoke(
                main, ["cost", "gpt-4o", "100", "50", "--source", "openrouter"]
            )

        mock_calc.assert_called_once_with(
            "gpt-4o",
            100,
            50,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            reasoning_tokens=0,
            source="openrouter",
        )

    def test_model_and_tokens_forwarded_correctly(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result()
        ) as mock_calc:
            runner.invoke(main, ["cost", "claude-3-5-sonnet", "200", "75"])

        mock_calc.assert_called_once_with(
            "claude-3-5-sonnet",
            200,
            75,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            reasoning_tokens=0,
            source="litellm",
        )

    def test_cached_input_tokens_flag(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result()
        ) as mock_calc:
            runner.invoke(
                main,
                ["cost", "gpt-4o", "1000", "500", "--cached-input-tokens", "200"],
            )

        mock_calc.assert_called_once_with(
            "gpt-4o",
            1000,
            500,
            cached_input_tokens=200,
            cache_creation_input_tokens=0,
            reasoning_tokens=0,
            source="litellm",
        )

    def test_cache_creation_input_tokens_flag(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result()
        ) as mock_calc:
            runner.invoke(
                main,
                [
                    "cost",
                    "gpt-4o",
                    "1000",
                    "500",
                    "--cache-creation-input-tokens",
                    "100",
                ],
            )

        mock_calc.assert_called_once_with(
            "gpt-4o",
            1000,
            500,
            cached_input_tokens=0,
            cache_creation_input_tokens=100,
            reasoning_tokens=0,
            source="litellm",
        )

    def test_reasoning_tokens_flag(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result()
        ) as mock_calc:
            runner.invoke(
                main,
                ["cost", "gpt-4o", "1000", "500", "--reasoning-tokens", "150"],
            )

        mock_calc.assert_called_once_with(
            "gpt-4o",
            1000,
            500,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            reasoning_tokens=150,
            source="litellm",
        )

    def test_all_new_flags_together(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.calculate_cost", return_value=_single_result()
        ) as mock_calc:
            runner.invoke(
                main,
                [
                    "cost",
                    "gpt-4o",
                    "1000",
                    "500",
                    "--cached-input-tokens",
                    "200",
                    "--cache-creation-input-tokens",
                    "100",
                    "--reasoning-tokens",
                    "150",
                ],
            )

        mock_calc.assert_called_once_with(
            "gpt-4o",
            1000,
            500,
            cached_input_tokens=200,
            cache_creation_input_tokens=100,
            reasoning_tokens=150,
            source="litellm",
        )


# ---------------------------------------------------------------------------
# models command
# ---------------------------------------------------------------------------


class TestModelsCommand:
    def test_lists_models_in_plain_text(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.list_models", return_value=["gpt-4o", "gpt-3.5-turbo"]
        ):
            result = runner.invoke(main, ["models"])

        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "gpt-3.5-turbo" in result.output

    def test_default_source_is_litellm(self):
        runner = CliRunner()
        with patch("modelcost.cli.list_models", return_value=[]) as mock_list:
            runner.invoke(main, ["models"])

        mock_list.assert_called_once_with("litellm")

    def test_source_option_forwarded(self):
        runner = CliRunner()
        with patch("modelcost.cli.list_models", return_value=[]) as mock_list:
            runner.invoke(main, ["models", "--source", "openrouter"])

        mock_list.assert_called_once_with("openrouter")

    def test_filter_option_filters_models(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.list_models",
            return_value=["gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet"],
        ):
            result = runner.invoke(main, ["models", "--filter", "gpt"])

        assert "gpt-4o" in result.output
        assert "gpt-3.5-turbo" in result.output
        assert "claude-3-5-sonnet" not in result.output

    def test_filter_is_case_insensitive(self):
        runner = CliRunner()
        with patch("modelcost.cli.list_models", return_value=["GPT-4o", "claude-3"]):
            result = runner.invoke(main, ["models", "--filter", "gpt"])

        assert "GPT-4o" in result.output
        assert "claude-3" not in result.output

    def test_json_flag_outputs_valid_json(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.list_models", return_value=["gpt-4o", "gpt-3.5-turbo"]
        ):
            result = runner.invoke(main, ["models", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["source"] == "litellm"
        assert data["count"] == 2
        assert data["models"] == ["gpt-4o", "gpt-3.5-turbo"]

    def test_json_with_filter_reflects_filtered_count(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.list_models",
            return_value=["gpt-4o", "gpt-3.5-turbo", "claude-3"],
        ):
            result = runner.invoke(main, ["models", "--filter", "gpt", "--json"])

        data = json.loads(result.output)
        assert data["count"] == 2
        assert "claude-3" not in data["models"]

    def test_list_models_exception_exits_1(self):
        runner = CliRunner()
        with patch("modelcost.cli.list_models", side_effect=ValueError("bad source")):
            result = runner.invoke(main, ["models"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_shows_total_count_in_output(self):
        runner = CliRunner()
        with patch(
            "modelcost.cli.list_models", return_value=["gpt-4o", "gpt-3.5-turbo"]
        ):
            result = runner.invoke(main, ["models"])

        assert "2" in result.output
