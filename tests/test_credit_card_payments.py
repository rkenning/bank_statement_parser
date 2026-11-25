"""Tests for transaction-derived payment totals."""

import re
from decimal import Decimal

import polars as pl
import pytest

from bank_statement_parser.modules.errors import ConfigError
from bank_statement_parser.modules.import_config import ImportConfigManager
from bank_statement_parser.modules.statements import Statement


class TestTransactionDerivedPayments:
    def test_derives_header_totals_from_transaction_lines(self) -> None:
        """Transaction totals update header and balance-check values."""
        statement = object.__new__(Statement)
        statement.header_results = pl.DataFrame(
            {
                "STD_PAYMENTS_IN": [Decimal("0.0000")],
                "STD_PAYMENTS_OUT": [Decimal("0.0000")],
            }
        ).lazy()
        statement.lines_results = pl.DataFrame(
            {
                "STD_TRANSACTION_PAYMENTS_IN": [Decimal("12.3400"), Decimal("0.0000")],
                "STD_TRANSACTION_PAYMENTS_OUT": [Decimal("0.0000"), Decimal("4.5000")],
            }
        ).lazy()
        statement.checks_and_balances = pl.DataFrame(
            {
                "STD_PAYMENTS_IN": [Decimal("0.0000")],
                "STD_PAYMENTS_OUT": [Decimal("0.0000")],
                "STD_BALANCE_OF_PAYMENTS": [Decimal("0.0000")],
            }
        )

        statement._apply_transaction_derived_payments()

        header = statement.header_results.collect()
        assert header["STD_PAYMENTS_IN"][0] == Decimal("12.3400")
        assert header["STD_PAYMENTS_OUT"][0] == Decimal("4.5000")
        assert statement.checks_and_balances["STD_BALANCE_OF_PAYMENTS"][0] == Decimal("7.8400")

    def test_rejects_empty_transaction_lines(self) -> None:
        """Deriving payment totals without transactions raises a config error."""
        statement = object.__new__(Statement)
        statement.header_results = pl.DataFrame({"STD_PAYMENTS_IN": [0.0], "STD_PAYMENTS_OUT": [0.0]}).lazy()
        statement.lines_results = pl.DataFrame(
            schema={
                "STD_TRANSACTION_PAYMENTS_IN": pl.Decimal(precision=12, scale=4),
                "STD_TRANSACTION_PAYMENTS_OUT": pl.Decimal(precision=12, scale=4),
            }
        ).lazy()
        statement.checks_and_balances = pl.DataFrame(
            {
                "STD_PAYMENTS_IN": [0.0],
                "STD_PAYMENTS_OUT": [0.0],
                "STD_BALANCE_OF_PAYMENTS": [0.0],
            }
        )

        with pytest.raises(ConfigError, match="without extracted transaction rows"):
            statement._apply_transaction_derived_payments()


class TestMandsConfig:
    def test_default_patterns_match_mands_statement_values(self) -> None:
        """Default M&S field patterns accept account numbers and transaction dates."""
        config = ImportConfigManager()
        statement_type = config.statement_types["MANDS_UK_CRD"]
        card_number_field = statement_type.header.configs[-1].field
        assert card_number_field is not None
        assert card_number_field.string_pattern is not None
        assert re.fullmatch(card_number_field.string_pattern, "2323 2323 2323 2323")

        statement_table = config.config_dict["statement_tables"]["config"]["MANDS_UK_CRD_TRANSACTIONS"]
        transaction_date_field = next(field for field in statement_table.fields if field.field == "transaction_date")
        assert transaction_date_field.string_pattern is not None
        assert re.fullmatch(transaction_date_field.string_pattern, "13 Mar 26")
