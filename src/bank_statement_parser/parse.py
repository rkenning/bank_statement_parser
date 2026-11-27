# This file is part of bank_statement_parser.
#
# Copyright (c) 2026 Jason Farrar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Public, side-effect-free API for normalized statement parsing."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Literal

import polars as pl
from dacite.exceptions import DaciteError
from pdfminer.pdfdocument import PDFEncryptionError
from pdfminer.pdfparser import PSEOF, PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException

from bank_statement_parser.modules.errors import StatementError
from bank_statement_parser.modules.import_config import ImportConfigManager
from bank_statement_parser.modules.statements import Statement

PARSE_RESULT_SCHEMA_VERSION = "1.0"

_PROVIDER_KEYS: dict[str, str] = {
    "hsbc": "HSBC_UK",
    "hsbc uk": "HSBC_UK",
    "hsbc_uk": "HSBC_UK",
    "hsbc bank uk": "HSBC_UK",
    "natwest": "NATWEST_UK",
    "natwest uk": "NATWEST_UK",
    "natwest_uk": "NATWEST_UK",
    "natwest bank uk": "NATWEST_UK",
    "m&s": "MANDS_UK",
    "m&s uk": "MANDS_UK",
    "mands": "MANDS_UK",
    "mands uk": "MANDS_UK",
    "mands_uk": "MANDS_UK",
    "marks & spencer": "MANDS_UK",
    "marks and spencer": "MANDS_UK",
    "tesco": "TESCO_UK",
    "tesco bank": "TESCO_UK",
    "tesco bank uk": "TESCO_UK",
    "tesco_uk": "TESCO_UK",
}

_PROVIDER_NAMES: dict[str, str] = {
    "HSBC_UK": "HSBC UK",
    "NATWEST_UK": "NatWest UK",
    "MANDS_UK": "M&S UK",
    "TESCO_UK": "Tesco Bank UK",
}

_ACCOUNT_TYPES: dict[str, Literal["credit_card", "current_account", "savings_account", "unknown"]] = {
    "CRD": "credit_card",
    "CUR": "current_account",
    "SAV": "savings_account",
}


@dataclass(frozen=True, slots=True)
class ParsedAccount:
    """Normalized identity for the account that owns a statement."""

    provider: str
    account_type: Literal["credit_card", "current_account", "savings_account", "unknown"]
    name: str
    account_number: str | None
    currency: str


@dataclass(frozen=True, slots=True)
class ParsedTransaction:
    """A validated normalized transaction."""

    date: date
    amount: Decimal
    description: str
    page: int
    source_ordinal: int


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One deterministic statement validation check."""

    code: str
    passed: bool
    expected: Decimal | int | None
    actual: Decimal | int | None
    difference: Decimal | int | None
    message: str


@dataclass(frozen=True, slots=True)
class ParseError:
    """A structured parsing or validation error."""

    code: str
    message: str
    source_ordinal: int | None = None


@dataclass(frozen=True, slots=True)
class ParseValidation:
    """Validation status and deterministic evidence for a parse result."""

    status: Literal["valid", "invalid", "failed"]
    checks: tuple[ValidationCheck, ...]
    errors: tuple[ParseError, ...]


@dataclass(frozen=True, slots=True)
class ParsedStatement:
    """Versioned, normalized result returned by :func:`parse_statement`."""

    schema_version: str
    provider: str | None
    account: ParsedAccount | None
    statement_date: date | None
    opening_balance: Decimal | None
    closing_balance: Decimal | None
    payments_in: Decimal | None
    payments_out: Decimal | None
    movement: Decimal | None
    transactions: tuple[ParsedTransaction, ...]
    validation: ParseValidation


def _decimal(value: object) -> Decimal | None:
    """Convert a numeric parser value to Decimal without binary float artifacts."""
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _date(value: object) -> date | None:
    """Normalize a parser date value."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _provider_key(provider: str | None) -> tuple[str | None, ParseError | None]:
    """Resolve a public provider alias to an internal configuration key."""
    if provider is None:
        return None, None
    key = provider.strip().lower().replace("’", "'")
    resolved = _PROVIDER_KEYS.get(key)
    if resolved is None:
        return None, ParseError("UNKNOWN_PROVIDER", f"Unknown statement provider: {provider!r}.")
    return resolved, None


def _account_key(manager: ImportConfigManager, provider_key: str | None, account: str | None) -> tuple[str | None, ParseError | None]:
    """Resolve an optional public account selector to an extraction account key."""
    if account is None:
        return None, None
    requested = account.strip()
    normalized = requested.lower().replace("-", "_").replace(" ", "_")
    exact_key = next((key for key in manager.accounts if key.lower() == requested.lower()), None)
    if exact_key is not None:
        configured = manager.accounts[exact_key]
        if provider_key is None or configured.company_key == provider_key:
            return exact_key, None
    candidates = [
        (key, configured)
        for key, configured in manager.accounts.items()
        if (provider_key is None or configured.company_key == provider_key)
        and (
            configured.account.lower() == requested.lower()
            or configured.account_type_key.lower() == normalized
            or _ACCOUNT_TYPES.get(configured.account_type_key) == normalized
        )
    ]
    if len(candidates) == 1:
        return candidates[0][0], None
    return None, ParseError("UNKNOWN_ACCOUNT", f"Unknown or ambiguous statement account: {account!r}.")


def _failed_result(error: ParseError, provider: str | None = None) -> ParsedStatement:
    """Build a stable failed result for setup and parse errors."""
    return ParsedStatement(
        schema_version=PARSE_RESULT_SCHEMA_VERSION,
        provider=provider,
        account=None,
        statement_date=None,
        opening_balance=None,
        closing_balance=None,
        payments_in=None,
        payments_out=None,
        movement=None,
        transactions=(),
        validation=ParseValidation(status="failed", checks=(), errors=(error,)),
    )


def _check(
    code: str,
    passed: bool,
    expected: Decimal | int | None,
    actual: Decimal | int | None,
    failure_message: str,
) -> ValidationCheck:
    """Build a normalized validation check."""
    difference = None
    if expected is not None and actual is not None:
        difference = actual - expected
    return ValidationCheck(
        code=code,
        passed=passed,
        expected=expected,
        actual=actual,
        difference=difference,
        message="" if passed else failure_message,
    )


def _validation_checks(cab: pl.DataFrame) -> tuple[tuple[ValidationCheck, ...], tuple[ParseError, ...]]:
    """Convert internal checks and balances into public validation evidence."""
    if cab.is_empty():
        error = ParseError("VALIDATION_UNAVAILABLE", "Statement validation checks could not be produced.")
        return (), (error,)

    row = cab.row(0, named=True)
    checks: list[ValidationCheck] = []
    errors: list[ParseError] = []
    specifications = (
        ("payments_in_match", "BAL_PAYMENTS_IN", "STD_PAYMENTS_IN", "STD_TRANSACTION_PAYMENTS_IN", "Payments in do not match."),
        ("payments_out_match", "BAL_PAYMENTS_OUT", "STD_PAYMENTS_OUT", "STD_TRANSACTION_PAYMENTS_OUT", "Payments out do not match."),
        ("movement_match", "BAL_MOVEMENT", "STD_MOVEMENT", "STD_TRANSACTION_MOVEMENT", "Statement movement does not match."),
        ("closing_balance_match", "BAL_CLOSING", "STD_CLOSING_BALANCE", "STD_RUNNING_BALANCE", "Closing balance does not match."),
    )
    for code, passed_column, expected_column, actual_column, message in specifications:
        passed = bool(row.get(passed_column, False))
        expected = _decimal(row.get(expected_column))
        actual = _decimal(row.get(actual_column))
        validation_check = _check(code, passed, expected, actual, message)
        checks.append(validation_check)
        if not passed:
            errors.append(ParseError(code.upper(), message))

    null_dates = int(row.get("TRANSACTION_LINES_WITH_NULL_DATE", 0))
    null_descriptions = int(row.get("TRANSACTION_LINES_WITH_NULL_DESC", 0))
    transaction_count = int(row.get("TRANSACTION_LINE_COUNT", 0))
    transaction_checks = (
        _check("transaction_dates_valid", null_dates == 0, 0, null_dates, f"{null_dates} transaction date(s) are invalid."),
        _check(
            "transaction_descriptions_valid",
            null_descriptions == 0,
            0,
            null_descriptions,
            f"{null_descriptions} transaction description(s) are invalid.",
        ),
        _check(
            "transaction_count",
            transaction_count > 0 or bool(row.get("ZERO_TRANSACTION_STATEMENT")),
            None,
            transaction_count,
            "No transactions were extracted.",
        ),
    )
    for validation_check in transaction_checks:
        checks.append(validation_check)
        if not validation_check.passed:
            errors.append(ParseError(validation_check.code.upper(), validation_check.message))

    return tuple(checks), tuple(errors)


def _transactions(lines: pl.DataFrame) -> tuple[tuple[ParsedTransaction, ...], tuple[ParseError, ...]]:
    """Convert normalized line columns to validated public transactions."""
    transactions: list[ParsedTransaction] = []
    errors: list[ParseError] = []
    if lines.is_empty():
        return (), ()

    for row in lines.iter_rows(named=True):
        ordinal_value = row.get("STD_TRANSACTION_NUMBER")
        try:
            ordinal = int(ordinal_value)
        except (TypeError, ValueError):
            ordinal = len(transactions) + 1
        transaction_date = _date(row.get("STD_TRANSACTION_DATE"))
        if transaction_date is None:
            errors.append(ParseError("INVALID_TRANSACTION_DATE", "Transaction date could not be validated.", ordinal))
            continue
        payments_in = _decimal(row.get("STD_TRANSACTION_PAYMENTS_IN"))
        payments_out = _decimal(row.get("STD_TRANSACTION_PAYMENTS_OUT"))
        if payments_in is None or payments_out is None:
            errors.append(ParseError("INVALID_TRANSACTION_AMOUNT", "Transaction amount could not be validated.", ordinal))
            continue
        description_value = row.get("STD_TRANSACTION_DESC")
        description = "" if description_value is None else str(description_value).strip()
        if not description:
            errors.append(ParseError("INVALID_TRANSACTION_DESCRIPTION", "Transaction description is empty.", ordinal))
            continue
        try:
            page = int(row.get("STD_PAGE_NUMBER") or 1)
        except (TypeError, ValueError):
            page = 1
        transactions.append(
            ParsedTransaction(
                date=transaction_date,
                amount=payments_in - payments_out,
                description=description,
                page=page,
                source_ordinal=ordinal,
            )
        )
    return tuple(transactions), tuple(errors)


def parse_statement(
    pdf: str | Path,
    provider: str | None = None,
    account: str | None = None,
    project_path: str | Path | None = None,
) -> ParsedStatement:
    """Parse one PDF statement into a versioned, normalized result.

    Parsing is deterministic and reads only the PDF and local TOML configuration.
    No database, parquet, cloud, GitHub, or network operations are performed.

    Args:
        pdf: Path to the statement PDF.
        provider: Optional provider alias, such as ``"HSBC"``, ``"NatWest"``,
            ``"M&S"``, or ``"Tesco Bank"``. When omitted, the provider is
            detected from the PDF.
        account: Optional configured account name or key, such as
            ``"current_account"``. When omitted, it is detected from the PDF.
        project_path: Optional project root containing custom ``config/import``
            TOML overrides. The bundled configuration is used when omitted.

    Returns:
        A :class:`ParsedStatement`. Parser and validation failures are represented
        in ``validation.errors`` with ``status`` set to ``"failed"`` or
        ``"invalid"`` rather than being raised.
    """
    source = Path(pdf)
    provider_key, provider_error = _provider_key(provider)
    if provider_error is not None:
        return _failed_result(provider_error)

    config_root = Path(project_path) if project_path is not None else None
    try:
        manager = ImportConfigManager(config_root)
        account_key, account_error = _account_key(manager, provider_key, account)
    except (DaciteError, KeyError, OSError, StatementError, TOMLDecodeError, TypeError, ValueError) as exc:
        return _failed_result(ParseError("CONFIGURATION_ERROR", f"Unable to load parser configuration: {exc}"))
    if account_error is not None:
        return _failed_result(account_error, _PROVIDER_NAMES.get(provider_key or "", provider))

    statement: Statement | None = None
    try:
        statement = Statement(
            file=source,
            company_key=provider_key,
            account_key=account_key,
            project_path=config_root,
            skip_project_validation=True,
        )
        if statement.error_message:
            return _failed_result(ParseError("PARSE_ERROR", statement.error_message), _PROVIDER_NAMES.get(provider_key or ""))

        config = statement.config
        if config is None or config.company is None:
            return _failed_result(ParseError("PROVIDER_NOT_DETECTED", "The statement provider and account could not be detected."))

        header = statement.header_results.collect()
        cab = statement.checks_and_balances
        checks, validation_errors = _validation_checks(cab)
        transactions, transaction_errors = _transactions(statement.lines_results.collect())

        header_row = header.row(0, named=True) if not header.is_empty() else {}
        statement_date = _date(header_row.get("STD_STATEMENT_DATE"))
        opening_balance = _decimal(header_row.get("STD_OPENING_BALANCE"))
        closing_balance = _decimal(header_row.get("STD_CLOSING_BALANCE"))
        cab_row = cab.row(0, named=True) if not cab.is_empty() else {}
        payments_in = _decimal(cab_row.get("STD_PAYMENTS_IN", header_row.get("STD_PAYMENTS_IN")))
        payments_out = _decimal(cab_row.get("STD_PAYMENTS_OUT", header_row.get("STD_PAYMENTS_OUT")))
        movement = _decimal(cab_row.get("STD_MOVEMENT"))
        if movement is None and opening_balance is not None and closing_balance is not None:
            movement = closing_balance - opening_balance

        provider_name = _PROVIDER_NAMES.get(config.company_key, config.company.company)
        account_number_value = header_row.get("STD_ACCOUNT_NUMBER")
        normalized_account = ParsedAccount(
            provider=provider_name,
            account_type=_ACCOUNT_TYPES.get(config.account_type_key, "unknown"),
            name=config.account,
            account_number=str(account_number_value).strip() if account_number_value is not None else None,
            currency=config.currency,
        )
        metadata_errors: list[ParseError] = []
        required_values = (
            ("STATEMENT_DATE_MISSING", statement_date),
            ("OPENING_BALANCE_MISSING", opening_balance),
            ("CLOSING_BALANCE_MISSING", closing_balance),
            ("PAYMENTS_IN_MISSING", payments_in),
            ("PAYMENTS_OUT_MISSING", payments_out),
        )
        for code, value in required_values:
            if value is None:
                metadata_errors.append(ParseError(code, f"Required normalized field {code.removesuffix('_MISSING').lower()} is missing."))

        errors = validation_errors + tuple(transaction_errors) + tuple(metadata_errors)
        if statement.success and not errors:
            status: Literal["valid", "invalid", "failed"] = "valid"
        else:
            status = "invalid"

        return ParsedStatement(
            schema_version=PARSE_RESULT_SCHEMA_VERSION,
            provider=provider_name,
            account=normalized_account,
            statement_date=statement_date,
            opening_balance=opening_balance,
            closing_balance=closing_balance,
            payments_in=payments_in,
            payments_out=payments_out,
            movement=movement,
            transactions=transactions,
            validation=ParseValidation(status=status, checks=checks, errors=errors),
        )
    except (
        EOFError,
        IndexError,
        KeyError,
        OSError,
        PDFEncryptionError,
        PDFSyntaxError,
        PdfminerException,
        PSEOF,
        StatementError,
        TypeError,
        ValueError,
        pl.exceptions.PolarsError,
    ) as exc:
        return _failed_result(ParseError("PARSE_ERROR", f"Unable to parse statement: {exc}"), _PROVIDER_NAMES.get(provider_key or ""))
    finally:
        if statement is not None:
            statement.cleanup()
