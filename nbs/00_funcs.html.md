---
title: funcs
---



> Pure functional helpers (side-effect free) used by mixins.



::: {#2 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from nbdev.showdoc import *
```
:::


::: {#3 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Iterable, Mapping, Sequence, Union, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from projio.core import ProjectIO
```
:::


::: {#4 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
StrPath = Union[str, Path]
```
:::


## Path normalization

::: {#6 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def normalize_path(path: StrPath | None, base: Path | None = None) -> Path:
    """Expand user, resolve against base when relative.
    
    Parameters:
        path: Path to normalize (string or Path). If None, returns base or cwd.
        base: Base path to resolve relative paths against.
        
    Returns:
        Resolved absolute Path.
    """
    if path is None:
        if base is None:
            return Path.cwd()
        return base
    p = Path(path).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    return p.resolve(strict=False)
```
:::


::: {#7 .cell}
``` {.python .cell-code}
# Test normalize_path
assert normalize_path(None) == Path.cwd()
assert normalize_path("foo", Path("/base")) == Path("/base/foo")
assert normalize_path("/abs/path", Path("/base")) == Path("/abs/path")
print("normalize_path tests passed")
```
:::


## Extension handling

::: {#9 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def ensure_extension(name: str, ext: str | None) -> str:
    """Ensure filename has the given extension.
    
    Extension may be provided with or without leading dot.
    Returns unchanged name if ext is None/empty.
    
    Parameters:
        name: Base filename.
        ext: Extension to ensure (with or without leading dot).
        
    Returns:
        Filename with proper extension.
    """
    if not ext:
        return name
    ext = ext if ext.startswith(".") else f".{ext}"
    if name.endswith(ext):
        return name
    # strip any existing extension
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return f"{stem}{ext}"
```
:::


::: {#10 .cell}
``` {.python .cell-code}
# Test ensure_extension
assert ensure_extension("model", ".ckpt") == "model.ckpt"
assert ensure_extension("model", "ckpt") == "model.ckpt"  # without dot
assert ensure_extension("model.ckpt", ".ckpt") == "model.ckpt"  # already has it
assert ensure_extension("model.old", ".ckpt") == "model.ckpt"  # replace ext
assert ensure_extension("model", None) == "model"  # no ext
assert ensure_extension("model", "") == "model"  # empty ext
print("ensure_extension tests passed")
```
:::


## Datestamp formatting and parsing

::: {#12 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def format_datestamp(dt: datetime | None, fmt: str) -> str:
    """Format datetime to datestamp string.
    
    Parameters:
        dt: Datetime to format. Uses now() if None.
        fmt: strftime format string.
        
    Returns:
        Formatted datestamp string.
    """
    return (dt or datetime.now()).strftime(fmt)
```
:::


::: {#13 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def parse_datestamp(text: str, fmt: str) -> datetime:
    """Parse datestamp string to datetime.
    
    Parameters:
        text: Datestamp string to parse.
        fmt: strftime format string used to create the datestamp.
        
    Returns:
        Parsed datetime object.
        
    Raises:
        ValueError: If text doesn't match the format.
    """
    try:
        return datetime.strptime(text, fmt)
    except ValueError as e:
        raise ValueError(f"Cannot parse '{text}' with format '{fmt}': {e}") from e
```
:::


::: {#14 .cell}
``` {.python .cell-code}
# Test datestamp functions
from datetime import datetime
dt = datetime(2024, 3, 15)
assert format_datestamp(dt, "%Y_%m_%d") == "2024_03_15"
assert parse_datestamp("2024_03_15", "%Y_%m_%d") == datetime(2024, 3, 15)

# Test error message
try:
    parse_datestamp("bad", "%Y_%m_%d")
    assert False, "Should have raised"
except ValueError as e:
    assert "Cannot parse 'bad'" in str(e)
print("datestamp tests passed")
```
:::


## Tree rendering

::: {#16 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def build_tree(root: Path, max_depth: int = 4, files: bool = False) -> str:
    """Build ASCII directory tree representation.
    
    Parameters:
        root: Root directory to start from.
        max_depth: Maximum depth to descend (default 4).
        files: If True, include files; otherwise only directories.
        
    Returns:
        ASCII tree string.
    """
    root = Path(root)
    lines: list[str] = [root.name or str(root)]
    
    def walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted([p for p in path.iterdir() if files or p.is_dir()])
        except PermissionError:
            return
        for i, child in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            lines.append(f"{prefix}{connector}{child.name}")
            if child.is_dir():
                extension = "    " if is_last else "\u2502   "
                walk(child, prefix + extension, depth + 1)
    
    if root.exists():
        walk(root, "", 1)
    return "\n".join(lines)
```
:::


::: {#17 .cell}
``` {.python .cell-code}
# Test build_tree with a simple example
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "test_root"
    (root / "subdir1").mkdir(parents=True)
    (root / "subdir2").mkdir(parents=True)
    (root / "subdir1" / "nested").mkdir()
    (root / "file.txt").touch()
    
    # Dirs only
    tree = build_tree(root, max_depth=2, files=False)
    assert "subdir1" in tree
    assert "nested" in tree
    assert "file.txt" not in tree
    
    # With files
    tree_with_files = build_tree(root, max_depth=2, files=True)
    assert "file.txt" in tree_with_files
print("build_tree tests passed")
```
:::


## Gitignore handling

::: {#19 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def render_gitignore(existing_text: str, entries: Iterable[str]) -> str:
    """Add entries to gitignore text, avoiding duplicates.
    
    Parameters:
        existing_text: Current gitignore content.
        entries: New entries to add.
        
    Returns:
        Updated gitignore text with new entries appended.
    """
    existing = {line.strip() for line in existing_text.splitlines() if line.strip() and not line.strip().startswith("#")}
    additions = [e for e in entries if e not in existing]
    if not additions:
        return existing_text
    # Preserve original text and append new entries
    text = existing_text.rstrip()
    if text:
        text += "\n"
    return text + "\n".join(additions) + "\n"
```
:::


::: {#20 .cell}
``` {.python .cell-code}
# Test render_gitignore
existing = "*.pyc\n__pycache__/\n"
result = render_gitignore(existing, ["checkpoints/", "logs/"])
assert "checkpoints/" in result
assert "logs/" in result
assert result.count("*.pyc") == 1  # not duplicated

# Idempotent
result2 = render_gitignore(result, ["checkpoints/", "logs/"])
assert result2 == result
print("render_gitignore tests passed")
```
:::


## Template specification

::: {#22 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
@dataclass
class TemplateSpec:
    """Specification for a path template.
    
    Attributes:
        name: Template name for registration/lookup.
        base: Base path or callable returning base path from ProjectIO.
        pattern: Sequence of path parts or mapping of key->filename.
        root: Which root to resolve relative base against ('inputs', 'outputs', 'cache', 'custom').
        datestamp: Override datestamp behavior (None uses instance default).
        create: Override auto_create behavior (None uses instance default).
    """
    name: str
    base: Union[StrPath, Callable[[Any], Path]]
    pattern: Sequence[str] | Mapping[str, str]
    root: str = "outputs"
    datestamp: bool | None = None
    create: bool | None = None
```
:::


## Template resolution

::: {#24 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def _resolve_base(spec: TemplateSpec, io: Any) -> Path:
    """Resolve template base path."""
    if callable(spec.base):
        base = spec.base(io)
    else:
        base = Path(spec.base)
    if not base.is_absolute():
        if spec.root == "inputs":
            base = io.inputs / base
        elif spec.root == "cache":
            base = io.cache / base
        elif spec.root == "outputs":
            base = io.outputs / base
        else:  # custom
            base = io.outputs / base
    return base
```
:::


::: {#25 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
def resolve_template(
    spec: TemplateSpec,
    io: Any,
    variant: str | None,
    fmt: Mapping[str, str],
    datestamp: bool | None,
    timestamp: datetime | None
) -> Path | dict[str, Path]:
    """Resolve a template spec to concrete path(s).
    
    Parameters:
        spec: Template specification.
        io: ProjectIO instance for context.
        variant: Optional variant name (e.g., run/model name).
        fmt: Format placeholders for pattern.
        datestamp: Override datestamp behavior.
        timestamp: Specific timestamp for datestamp.
        
    Returns:
        Single Path if pattern is a sequence, or dict of Paths if pattern is a mapping.
    """
    base = _resolve_base(spec, io)
    ds = datestamp if datestamp is not None else spec.datestamp
    if ds is None:
        ds = getattr(io, "use_datestamp", False)
    
    # Build format context
    format_ctx = dict(fmt)
    if variant is not None:
        format_ctx.setdefault("variant", variant)
        format_ctx.setdefault("run", variant)
    
    def maybe_datestamp_dir(path: Path) -> Path:
        if ds and getattr(io, "datestamp_in", "dirs") in ("dirs", "both"):
            return path / format_datestamp(timestamp, io.datestamp_format)
        return path
    
    # Handle mapping pattern (multiple files)
    if isinstance(spec.pattern, Mapping):
        out: dict[str, Path] = {}
        for key, pattern in spec.pattern.items():
            filename = pattern.format_map(format_ctx) if "{" in pattern else pattern
            target = maybe_datestamp_dir(base) / filename
            out[key] = target
            should_create = spec.create if spec.create is not None else io.auto_create
            if should_create and not io.dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
        return out
    
    # Handle sequence pattern (path parts)
    parts = [p.format_map(format_ctx) if "{" in p else p for p in spec.pattern]
    target = maybe_datestamp_dir(base)
    if variant:
        target = target / variant
    target = target.joinpath(*parts)
    
    # Add datestamp to filename if needed
    if ds and getattr(io, "datestamp_in", "dirs") in ("files", "both"):
        parent = target.parent
        pref = format_datestamp(timestamp, io.datestamp_format)
        target = parent / f"{pref}__{target.name}"
    
    should_create = spec.create if spec.create is not None else io.auto_create
    if should_create and not io.dry_run:
        # Create parent for files, or the dir itself for directories
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
    
    return target
```
:::


::: {#26 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
import nbdev; nbdev.nbdev_export()
```
:::


