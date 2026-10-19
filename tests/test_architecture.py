"""Layering rules from docs/ARCHITEKTURA_ROZSZERZENIA.md, checked on the source code.

A module may import only from its own layer or the layers below it, and the user interfaces
(``app``, ``cli``) only from ``services`` plus the shared ``domain`` / ``exceptions``. Imports
inside functions count too.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "src" / "race_predictor"

LAYERS = {
    "domain": 0,
    "exceptions": 0,
    "utils": 0,
    "sources": 1,
    "processing": 2,
    "features": 3,
    "models": 4,
    "evaluation": 5,
    "services": 6,
    "predictor": 7,  # backward-compatible alias of services
    "app": 7,
    "cli": 7,
}
INTERFACES = {"app", "cli"}
INTERFACE_ALLOWED = {"services", "domain", "exceptions"}


def top_level(module: str) -> str | None:
    """'race_predictor.models.router' -> 'models'; None for non-package imports."""
    parts = module.split(".")
    return parts[1] if parts[0] == "race_predictor" and len(parts) > 1 else None


def package_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {m for m in found if top_level(m)}


MODULES = sorted(p for p in PACKAGE.rglob("*.py") if p.name != "__init__.py" or p.parent != PACKAGE)


def owner(path: Path) -> str:
    return path.relative_to(PACKAGE).parts[0].removesuffix(".py")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_module_has_a_layer(path):
    assert owner(path) in LAYERS, "new top-level module: add it to LAYERS deliberately"


def upward_imports(path: Path, me: str) -> list[str]:
    return sorted(m for m in package_imports(path) if LAYERS[str(top_level(m))] > LAYERS[me])


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_imports_respect_layers(path):
    me = owner(path)
    upward = upward_imports(path, me)
    assert not upward, f"{me} (layer {LAYERS[me]}) imports a higher layer: {upward}"


@pytest.mark.parametrize(
    "path", [p for p in MODULES if owner(p) in INTERFACES], ids=lambda p: p.name
)
def test_interfaces_use_only_services(path):
    bad = [m for m in package_imports(path) if top_level(m) not in INTERFACE_ALLOWED]
    assert not bad, f"user interface must go through race_predictor.services, not {bad}"


def test_checker_reports_an_upward_import_inside_a_function(tmp_path):
    module = tmp_path / "bad.py"
    module.write_text(
        "from race_predictor.features import RaceDataset\n"
        "def f():\n    from race_predictor.services import RacePredictor\n"
    )
    assert upward_imports(module, "models") == ["race_predictor.services"]
    assert upward_imports(module, "services") == []
