"""Core components of the MLMom AutoML agent."""

from typing import TYPE_CHECKING, Any

from .contracts import DatasetProfile, ResultFormat, TaskSpec

if TYPE_CHECKING:
    from .data_inspector import DataInspector

__all__ = ["DataInspector", "DatasetProfile", "ResultFormat", "TaskSpec"]


def __getattr__(name: str) -> Any:
    """Import the CLI module lazily so ``python -m`` runs without warnings."""

    if name == "DataInspector":
        from .data_inspector import DataInspector

        return DataInspector
    raise AttributeError(name)

