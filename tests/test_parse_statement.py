"""Synthetic tests for the public normalized statement API."""

from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl
import pytest

import bank_statement_parser.parse as parser
from bank_statement_parser.modules.import_config import ImportConfigManager


def _write_synthetic_tesco_pdf(path: Path) -> None:
    """Write a deterministic single-page Tesco-style statement PDF."""
    page_height = 842
    text_lines = (
        (40, 32, "Tesco Bank"),
        (40, 78, "Tesco Bank Credit Card"),
        (395, 48, "31 March 2026"),
        (395, 78, "1234 5678 9012 3456"),
        (350, 142, "Opening balance"),
        (480, 142, "100.00"),
        (350, 162, "Closing balance"),
        (480, 162, "107.50"),
        (40, 260, "Date"),
        (110, 260, "Description"),
        (340, 260, "Amount"),
        (430, 260, "Type"),
        (505, 260, "Balance"),
        (40, 282, "10 Mar 26"),
        (110, 282, "SYNTHETIC PAYMENT"),
        (340, 282, "12.50 CR"),
        (505, 282, "112.50 CR"),
        (40, 304, "20 Mar 26"),
        (110, 304, "SYNTHETIC PURCHASE"),
        (340, 304, "20.00"),
        (505, 304, "107.50"),
    )
    commands = []
    for x, top, text in text_lines:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"BT /F1 9 Tf 1 0 0 1 {x} {page_height - top} Tm ({escaped}) Tj ET")
    content = ("\n".join(commands) + "\n").encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
    )
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(pdf)


class _SyntheticStatement:
    """Deterministic statement fixture for public API normalization tests."""

    __slots__ = (
        "checks_and_balances",
        "config",
        "error_message",
        "header_results",
        "lines_results",
        "success",
    )

    provider_accounts: ClassVar[dict[str, str]] = {
        "HSBC_UK": "HSBC_UK_CUR_STD",
        "NATWEST_UK": "NATWEST_UK_CUR_STD_OLD",
        "MANDS_UK": "MANDS_UK_CRD_STD",
        "TESCO_UK": "TESCO_UK_CRD_STD",
    }

    def __init__(
        self,
        file: Path,
        company_key: str | None,
        account_key: str | None,
        project_path: Path | None,
        ID_BATCH: str | None = None,
        skip_project_validation: bool = False,
    ) -> None:
        manager = ImportConfigManager(project_path)
        selected_account = account_key or self.provider_accounts.get(company_key or "", "HSBC_UK_CUR_STD")
        self.config = manager.accounts[selected_account]
        self.error_message = ""
        self.success = True
        self.header_results = pl.DataFrame(
            {
                "STD_ACCOUNT_NUMBER": ["12345678"],
                "STD_STATEMENT_DATE": [date(2026, 3, 31)],
                "STD_OPENING_BALANCE": [Decimal("100.0000")],
                "STD_CLOSING_BALANCE": [Decimal("92.5000")],
                "STD_PAYMENTS_IN": [Decimal("12.5000")],
                "STD_PAYMENTS_OUT": [Decimal("20.0000")],
            }
        ).lazy()
        self.lines_results = pl.DataFrame(
            {
                "STD_TRANSACTION_DATE": [date(2026, 3, 10), date(2026, 3, 20)],
                "STD_TRANSACTION_PAYMENTS_IN": [Decimal("12.5000"), Decimal("0.0000")],
                "STD_TRANSACTION_PAYMENTS_OUT": [Decimal("0.0000"), Decimal("20.0000")],
                "STD_TRANSACTION_DESC": ["SYNTHETIC CREDIT", "SYNTHETIC DEBIT"],
                "STD_PAGE_NUMBER": [1, 2],
                "STD_TRANSACTION_NUMBER": [1, 2],
            }
        ).lazy()
        self.checks_and_balances = pl.DataFrame(
            {
                "STD_PAYMENTS_IN": [Decimal("12.5000")],
                "STD_PAYMENTS_OUT": [Decimal("20.0000")],
                "STD_TRANSACTION_PAYMENTS_IN": [Decimal("12.5000")],
                "STD_TRANSACTION_PAYMENTS_OUT": [Decimal("20.0000")],
                "STD_MOVEMENT": [Decimal("-7.5000")],
                "STD_TRANSACTION_MOVEMENT": [Decimal("-7.5000")],
                "STD_CLOSING_BALANCE": [Decimal("92.5000")],
                "STD_RUNNING_BALANCE": [Decimal("92.5000")],
                "BAL_PAYMENTS_IN": [True],
                "BAL_PAYMENTS_OUT": [True],
                "BAL_MOVEMENT": [True],
                "BAL_CLOSING": [True],
                "TRANSACTION_LINE_COUNT": [2],
                "TRANSACTION_LINES_WITH_NULL_DATE": [0],
                "TRANSACTION_LINES_WITH_NULL_DESC": [0],
                "ZERO_TRANSACTION_STATEMENT": [False],
            }
        )

    def cleanup(self) -> None:
        """Release the synthetic parser resource."""


class TestParseStatementProviders:
    @pytest.fixture
    def synthetic_parser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace PDF extraction with deterministic synthetic normalized frames."""
        monkeypatch.setattr(parser, "Statement", _SyntheticStatement)

    def _assert_statement(self, tmp_path: Path, provider: str, expected_provider: str) -> parser.ParsedStatement:
        """Parse and verify the shared normalized contract."""
        source = tmp_path / "synthetic.pdf"
        source.write_bytes(b"%PDF-1.4\n% synthetic")
        result = parser.parse_statement(source, provider=provider)

        assert result.schema_version == "1.0"
        assert result.provider == expected_provider
        assert result.account is not None
        assert result.statement_date == date(2026, 3, 31)
        assert result.opening_balance == Decimal("100.0000")
        assert result.closing_balance == Decimal("92.5000")
        assert result.payments_in == Decimal("12.5000")
        assert result.payments_out == Decimal("20.0000")
        assert result.movement == Decimal("-7.5000")
        assert len(result.transactions) == 2
        assert [transaction.amount for transaction in result.transactions] == [Decimal("12.5000"), Decimal("-20.0000")]
        assert [transaction.page for transaction in result.transactions] == [1, 2]
        assert [transaction.source_ordinal for transaction in result.transactions] == [1, 2]
        assert result.validation.status == "valid"
        assert all(check.passed for check in result.validation.checks)
        assert result.validation.errors == ()
        assert all("id_" not in key for key in asdict(result))
        return result

    def test_hsbc_current_account(self, tmp_path: Path, synthetic_parser: None) -> None:
        """HSBC current accounts normalize dates, balances, movement, and signed transactions."""
        result = self._assert_statement(tmp_path, "HSBC", "HSBC UK")
        assert result.account is not None
        assert result.account.account_type == "current_account"

    def test_natwest_old_current_account(self, tmp_path: Path, synthetic_parser: None) -> None:
        """NatWest old-format current accounts use the normalized API contract."""
        result = self._assert_statement(tmp_path, "NatWest", "NatWest UK")
        assert result.account is not None
        assert result.account.account_type == "current_account"

    def test_mands_credit_card(self, tmp_path: Path, synthetic_parser: None) -> None:
        """M&S credit cards normalize purchases and payments as signed amounts."""
        result = self._assert_statement(tmp_path, "M&S", "M&S UK")
        assert result.account is not None
        assert result.account.account_type == "credit_card"

    def test_tesco_credit_card(self, tmp_path: Path, synthetic_parser: None) -> None:
        """Tesco Bank credit cards normalize through the same versioned schema."""
        result = self._assert_statement(tmp_path, "Tesco Bank", "Tesco Bank UK")
        assert result.account is not None
        assert result.account.account_type == "credit_card"

    def test_synthetic_tesco_pdf_uses_deterministic_table_parsing(self, tmp_path: Path) -> None:
        """A synthetic Tesco PDF produces validated normalized transactions."""
        source = tmp_path / "synthetic-tesco.pdf"
        _write_synthetic_tesco_pdf(source)

        result = parser.parse_statement(source, provider="Tesco Bank")

        assert result.validation.status == "valid"
        assert result.statement_date == date(2026, 3, 31)
        assert result.opening_balance == Decimal("-100.0000")
        assert result.closing_balance == Decimal("-107.5000")
        assert result.payments_in == Decimal("12.5000")
        assert result.payments_out == Decimal("20.0000")
        assert result.movement == Decimal("-7.5000")
        assert [transaction.amount for transaction in result.transactions] == [Decimal("12.5000"), Decimal("-20.0000")]
        assert [transaction.source_ordinal for transaction in result.transactions] == [1, 2]


class _InvalidSyntheticStatement(_SyntheticStatement):
    """Synthetic statement with deterministic validation failures."""

    validation_mode: Literal["invalid", "parse"] = "invalid"

    def __init__(
        self,
        file: Path,
        company_key: str | None,
        account_key: str | None,
        project_path: Path | None,
        ID_BATCH: str | None = None,
        skip_project_validation: bool = False,
    ) -> None:
        super().__init__(file, company_key, account_key, project_path, ID_BATCH, skip_project_validation)
        if self.validation_mode == "parse":
            self.error_message = "Synthetic parse failure"
            self.success = False
            return
        self.success = False
        self.checks_and_balances = self.checks_and_balances.with_columns(
            BAL_CLOSING=pl.lit(False),
            STD_RUNNING_BALANCE=pl.lit(Decimal("91.5000")),
            TRANSACTION_LINES_WITH_NULL_DATE=pl.lit(1),
        )
        self.lines_results = self.lines_results.with_columns(
            STD_TRANSACTION_DATE=pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("STD_TRANSACTION_DATE"))
        )


class TestParseStatementFailures:
    def test_failed_parser_returns_structured_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed extraction returns a failed validation result instead of raising."""
        _InvalidSyntheticStatement.validation_mode = "parse"
        monkeypatch.setattr(parser, "Statement", _InvalidSyntheticStatement)
        source = tmp_path / "malformed.pdf"
        source.write_bytes(b"not a pdf")

        result = parser.parse_statement(source, provider="HSBC")

        assert result.validation.status == "failed"
        assert [error.code for error in result.validation.errors] == ["PARSE_ERROR"]
        assert result.transactions == ()

    def test_invalid_checks_keep_partial_transactions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failed balance and transaction validation produces structured review data."""
        _InvalidSyntheticStatement.validation_mode = "invalid"
        monkeypatch.setattr(parser, "Statement", _InvalidSyntheticStatement)
        source = tmp_path / "invalid.pdf"
        source.write_bytes(b"%PDF-1.4\n% synthetic")

        result = parser.parse_statement(source, provider="NatWest")

        assert result.validation.status == "invalid"
        assert len(result.transactions) == 1
        assert any(check.code == "closing_balance_match" and not check.passed for check in result.validation.checks)
        assert any(error.code == "INVALID_TRANSACTION_DATE" for error in result.validation.errors)

    def test_actual_malformed_pdf_returns_failed_result(self, tmp_path: Path) -> None:
        """A malformed PDF is represented as a structured failure."""
        source = tmp_path / "actually-malformed.pdf"
        source.write_bytes(b"this is not a PDF")

        result = parser.parse_statement(source, provider="HSBC")

        assert result.validation.status == "failed"
        assert result.validation.errors[0].code == "PARSE_ERROR"
