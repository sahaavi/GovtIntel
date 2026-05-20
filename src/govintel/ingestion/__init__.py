"""Ingestion package exports."""

from typing import Any

__all__ = ["BootstrapResult", "bootstrap_contract_data"]


def __getattr__(name: str) -> Any:
    """Load bootstrap exports lazily so ``python -m`` stays warning-free."""

    if name in __all__:
        from govintel.ingestion.bootstrap import BootstrapResult, bootstrap_contract_data

        return {
            "BootstrapResult": BootstrapResult,
            "bootstrap_contract_data": bootstrap_contract_data,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
