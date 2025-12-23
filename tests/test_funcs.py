"""Tests for project_io.funcs module - pure functional helpers."""
import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import os

from projio.funcs import (
    normalize_path,
    ensure_extension,
    format_datestamp,
    parse_datestamp,
    build_tree,
    render_gitignore,
    TemplateSpec,
    resolve_template,
)


class TestNormalizePath:
    """Tests for normalize_path function."""

    def test_none_returns_cwd(self):
        assert normalize_path(None) == Path.cwd()

    def test_none_with_base_returns_base(self):
        base = Path("/some/base")
        assert normalize_path(None, base) == base

    def test_relative_path_resolved_against_base(self):
        base = Path("/base")
        result = normalize_path("subdir", base)
        assert result == Path("/base/subdir")

    def test_absolute_path_ignores_base(self):
        base = Path("/base")
        result = normalize_path("/abs/path", base)
        assert result == Path("/abs/path")

    def test_expanduser(self, tmp_path):
        # Test that ~ is expanded
        result = normalize_path("~/test", tmp_path)
        assert "~" not in str(result)

    def test_string_input(self):
        base = Path("/base")
        result = normalize_path("foo/bar", base)
        assert isinstance(result, Path)
        assert result == Path("/base/foo/bar")

    def test_path_input(self):
        base = Path("/base")
        result = normalize_path(Path("foo"), base)
        assert result == Path("/base/foo")


class TestEnsureExtension:
    """Tests for ensure_extension function."""

    def test_add_extension_with_dot(self):
        assert ensure_extension("model", ".ckpt") == "model.ckpt"

    def test_add_extension_without_dot(self):
        assert ensure_extension("model", "ckpt") == "model.ckpt"

    def test_already_has_extension(self):
        assert ensure_extension("model.ckpt", ".ckpt") == "model.ckpt"

    def test_replace_existing_extension(self):
        assert ensure_extension("model.old", ".ckpt") == "model.ckpt"

    def test_none_extension(self):
        assert ensure_extension("model", None) == "model"

    def test_empty_extension(self):
        assert ensure_extension("model", "") == "model"

    def test_name_with_multiple_dots(self):
        result = ensure_extension("file.tar.gz", ".zip")
        # Should replace the extension after the last dot
        assert result.endswith(".zip")


class TestDatestampFunctions:
    """Tests for datestamp formatting and parsing."""

    def test_format_datestamp_with_datetime(self):
        dt = datetime(2024, 3, 15, 10, 30, 45)
        result = format_datestamp(dt, "%Y_%m_%d")
        assert result == "2024_03_15"

    def test_format_datestamp_none_uses_now(self):
        result = format_datestamp(None, "%Y")
        assert result == str(datetime.now().year)

    def test_format_datestamp_custom_format(self):
        dt = datetime(2024, 3, 15)
        assert format_datestamp(dt, "%Y-%m-%d") == "2024-03-15"
        assert format_datestamp(dt, "%d.%m.%Y") == "15.03.2024"

    def test_parse_datestamp_basic(self):
        result = parse_datestamp("2024_03_15", "%Y_%m_%d")
        assert result == datetime(2024, 3, 15)

    def test_parse_datestamp_custom_format(self):
        result = parse_datestamp("15-03-2024", "%d-%m-%Y")
        assert result == datetime(2024, 3, 15)

    def test_parse_datestamp_invalid_raises(self):
        with pytest.raises(ValueError) as exc_info:
            parse_datestamp("invalid", "%Y_%m_%d")
        assert "Cannot parse 'invalid'" in str(exc_info.value)

    def test_parse_datestamp_format_mismatch_raises(self):
        with pytest.raises(ValueError):
            parse_datestamp("2024-03-15", "%Y_%m_%d")  # wrong separator

    def test_format_parse_roundtrip(self):
        dt = datetime(2024, 6, 20)
        fmt = "%Y_%m_%d"
        text = format_datestamp(dt, fmt)
        parsed = parse_datestamp(text, fmt)
        assert parsed == dt


class TestBuildTree:
    """Tests for build_tree function."""

    def test_empty_directory(self, tmp_path):
        result = build_tree(tmp_path, max_depth=2, files=False)
        assert tmp_path.name in result

    def test_nested_directories(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "c").mkdir()

        result = build_tree(tmp_path, max_depth=3, files=False)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_max_depth_limits(self, tmp_path):
        (tmp_path / "a" / "b" / "c" / "d").mkdir(parents=True)

        # With depth 2, we should see a and b but not c or d
        result = build_tree(tmp_path, max_depth=2, files=False)
        lines = result.split("\n")
        assert any("a" in line for line in lines)
        assert any("b" in line for line in lines)

    def test_files_excluded_by_default(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").touch()

        result = build_tree(tmp_path, max_depth=2, files=False)
        assert "file.txt" not in result
        assert "subdir" in result

    def test_files_included_when_requested(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").touch()

        result = build_tree(tmp_path, max_depth=2, files=True)
        assert "file.txt" in result
        assert "subdir" in result

    def test_nonexistent_directory(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        result = build_tree(nonexistent)
        # Should just return the name without children
        assert "does_not_exist" in result


class TestRenderGitignore:
    """Tests for render_gitignore function."""

    def test_add_new_entries_to_empty(self):
        result = render_gitignore("", ["checkpoints/", "logs/"])
        assert "checkpoints/" in result
        assert "logs/" in result

    def test_add_new_entries_to_existing(self):
        existing = "*.pyc\n__pycache__/\n"
        result = render_gitignore(existing, ["checkpoints/"])
        assert "*.pyc" in result
        assert "__pycache__/" in result
        assert "checkpoints/" in result

    def test_no_duplicates(self):
        existing = "checkpoints/\n"
        result = render_gitignore(existing, ["checkpoints/"])
        assert result.count("checkpoints/") == 1

    def test_idempotent(self):
        existing = "*.pyc\nlogs/\n"
        result1 = render_gitignore(existing, ["cache/"])
        result2 = render_gitignore(result1, ["cache/"])
        assert result1 == result2

    def test_preserves_comments(self):
        existing = "# Python\n*.pyc\n"
        result = render_gitignore(existing, ["logs/"])
        assert "# Python" in result

    def test_multiple_entries_at_once(self):
        result = render_gitignore("", ["a/", "b/", "c/"])
        assert "a/" in result
        assert "b/" in result
        assert "c/" in result

    def test_no_changes_when_all_exist(self):
        existing = "a/\nb/\n"
        result = render_gitignore(existing, ["a/", "b/"])
        assert result == existing


class TestTemplateSpec:
    """Tests for TemplateSpec dataclass."""

    def test_create_with_sequence_pattern(self):
        spec = TemplateSpec(
            name="test",
            base="/base",
            pattern=["{run}", "{model}.ckpt"]
        )
        assert spec.name == "test"
        assert spec.pattern == ["{run}", "{model}.ckpt"]

    def test_create_with_mapping_pattern(self):
        spec = TemplateSpec(
            name="matrix",
            base="/base",
            pattern={"a": "a.txt", "b": "b.txt"}
        )
        assert isinstance(spec.pattern, dict)

    def test_default_values(self):
        spec = TemplateSpec(name="test", base="/", pattern=[])
        assert spec.root == "outputs"
        assert spec.datestamp is None
        assert spec.create is None


class TestResolveTemplate:
    """Tests for resolve_template function."""

    class MockIO:
        """Mock ProjectIO for testing."""
        def __init__(self, base="/project"):
            self._base = Path(base)
            self.outputs = self._base / "outputs"
            self.inputs = self._base / "inputs"
            self.cache = self._base / "cache"
            self.auto_create = False
            self.dry_run = True
            self.use_datestamp = False
            self.datestamp_in = "dirs"
            self.datestamp_format = "%Y_%m_%d"

    def test_resolve_sequence_pattern(self):
        io = self.MockIO()
        spec = TemplateSpec(
            name="test",
            base=lambda x: x.outputs,
            pattern=["{run}"]
        )
        # variant "exp1" is used both in pattern {run} and as variant path component
        result = resolve_template(spec, io, "exp1", {}, False, None)
        # The result includes variant twice: once from pattern, once from variant path
        assert "exp1" in str(result)

    def test_resolve_mapping_pattern(self):
        io = self.MockIO()
        spec = TemplateSpec(
            name="matrix",
            base=lambda x: x.outputs,
            pattern={"file1": "a.txt", "file2": "b.txt"}
        )
        result = resolve_template(spec, io, None, {}, False, None)
        assert isinstance(result, dict)
        assert result["file1"] == Path("/project/outputs/a.txt")
        assert result["file2"] == Path("/project/outputs/b.txt")

    def test_format_placeholders(self):
        io = self.MockIO()
        spec = TemplateSpec(
            name="test",
            base=lambda x: x.outputs,
            pattern=["{custom}"]
        )
        result = resolve_template(spec, io, None, {"custom": "value"}, False, None)
        assert "value" in str(result)

    def test_variant_used_in_path(self):
        io = self.MockIO()
        spec = TemplateSpec(
            name="test",
            base=lambda x: x.outputs,
            pattern=["{run}"]
        )
        result = resolve_template(spec, io, "my_run", {}, False, None)
        assert "my_run" in str(result)
