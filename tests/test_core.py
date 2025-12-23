"""Tests for project_io.core module."""
import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import os

from projio.core import (
    ProjectIO,
    PIO,
    ProducerRecord,
    TemplateSpec,
    LIGHTNING_TEMPLATES,
    CORE_TEMPLATES,
)


class TestRootCascade:
    """Tests for root path cascade behavior."""

    def test_root_defaults_to_cwp(self, tmp_path):
        os.chdir(tmp_path)
        io = ProjectIO(auto_create=False)
        assert io.root == tmp_path

    def test_iroot_oroot_follow_root(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False)
        assert io.iroot == tmp_path
        assert io.oroot == tmp_path

    def test_explicit_iroot_not_overwritten(self, tmp_path):
        iroot = tmp_path / "inputs"
        io = ProjectIO(root=tmp_path, iroot=iroot, auto_create=False)
        assert io.iroot == iroot

        # Changing root should not affect explicitly set iroot
        new_root = tmp_path / "new_root"
        io.root = new_root
        assert io.iroot == iroot
        assert io.oroot == new_root  # oroot was not explicitly set

    def test_explicit_oroot_not_overwritten(self, tmp_path):
        oroot = tmp_path / "outputs"
        io = ProjectIO(root=tmp_path, oroot=oroot, auto_create=False)
        assert io.oroot == oroot

        # Changing root should not affect explicitly set oroot
        new_root = tmp_path / "new_root"
        io.root = new_root
        assert io.oroot == oroot
        assert io.iroot == new_root  # iroot was not explicitly set

    def test_setting_iroot_marks_user_set(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False)

        # Initially, iroot follows root
        io.root = tmp_path / "a"
        assert io.iroot == tmp_path / "a"

        # After setting iroot, it's marked as user-set
        io.iroot = tmp_path / "custom_inputs"
        io.root = tmp_path / "b"
        assert io.iroot == tmp_path / "custom_inputs"

    def test_root_type_error(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False)
        with pytest.raises(TypeError):
            io.root = 123


class TestDatestampPlacements:
    """Tests for datestamp_in dirs/files/both/none."""

    def test_datestamp_in_dirs(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            datestamp_in="dirs",
            use_datestamp=True,
            auto_create=False
        )
        io.datestamp_value = lambda ts=None: "DATE"
        path = io.checkpoint_path("model", run="run1")
        parts = path.parts
        assert "DATE" in parts
        assert not path.name.startswith("DATE__")

    def test_datestamp_in_files(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            datestamp_in="files",
            use_datestamp=True,
            auto_create=False
        )
        io.datestamp_value = lambda ts=None: "DATE"
        path = io.checkpoint_path("model")
        assert path.name == "DATE__model.ckpt"

    def test_datestamp_in_both(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            datestamp_in="both",
            use_datestamp=True,
            auto_create=False
        )
        io.datestamp_value = lambda ts=None: "DATE"
        path = io.checkpoint_path("model")
        assert "DATE" in path.parts  # in directory
        assert path.name.startswith("DATE__")  # in filename

    def test_datestamp_in_none(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            datestamp_in="none",
            use_datestamp=True,
            auto_create=False
        )
        io.datestamp_value = lambda ts=None: "DATE"
        path = io.checkpoint_path("model")
        assert "DATE" not in str(path)

    def test_use_datestamp_false_overrides(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            datestamp_in="dirs",
            use_datestamp=False,
            auto_create=False
        )
        path = io.checkpoint_path("model")
        today = datetime.now().strftime("%Y_%m_%d")
        assert today not in str(path)

    def test_path_for_files_datestamp(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            datestamp_in="files",
            use_datestamp=True,
            auto_create=False
        )
        io.datestamp_value = lambda ts=None: "DATE"
        path = io.path_for("logs", name="train.log")
        assert path.name == "DATE__train.log"


class TestDryRun:
    """Tests for dry_run mode - no filesystem side effects."""

    def test_dry_run_no_directory_creation(self, tmp_path):
        io = ProjectIO(root=tmp_path, dry_run=True, auto_create=True)

        # Access properties that would normally create dirs
        _ = io.checkpoints
        _ = io.logs
        _ = io.cache

        # Verify no directories were created
        assert not (tmp_path / "lightning").exists()
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "cache").exists()

    def test_dry_run_checkpoint_path(self, tmp_path):
        io = ProjectIO(root=tmp_path, dry_run=True, auto_create=True)
        path = io.checkpoint_path("model", run="exp1")

        # Path should be returned but parent should not exist
        assert path is not None
        assert not path.parent.exists()

    def test_dry_run_gitignore_no_write(self, tmp_path):
        gi_path = tmp_path / ".gitignore"
        io = ProjectIO(root=tmp_path, dry_run=True, gitignore=gi_path)

        io.append_gitignore(["test/"])

        # .gitignore should not exist
        assert not gi_path.exists()

    def test_using_context_for_dry_run(self, tmp_path):
        io = ProjectIO(root=tmp_path, dry_run=False, auto_create=True)

        with io.using(dry_run=True):
            _ = io.cache  # Would create dir if not dry_run

        # After context, should still not create (was in dry_run)
        # But now outside, it should work
        assert io.dry_run is False


class TestGitignoreIntegration:
    """Tests for gitignore management."""

    def test_append_gitignore_creates_file(self, tmp_path):
        gi_path = tmp_path / ".gitignore"
        io = ProjectIO(root=tmp_path, gitignore=gi_path)

        io.append_gitignore(["checkpoints/"])

        assert gi_path.exists()
        assert "checkpoints/" in gi_path.read_text()

    def test_append_gitignore_idempotent(self, tmp_path):
        gi_path = tmp_path / ".gitignore"
        gi_path.write_text("existing/\n")
        io = ProjectIO(root=tmp_path, gitignore=gi_path)

        io.append_gitignore(["existing/", "new/"])

        content = gi_path.read_text()
        assert content.count("existing/") == 1
        assert "new/" in content

    def test_ensure_gitignored_adds_kinds(self, tmp_path):
        gi_path = tmp_path / ".gitignore"
        io = ProjectIO(root=tmp_path, gitignore=gi_path, use_datestamp=False)

        io.ensure_gitignored("cache", "logs")

        content = gi_path.read_text()
        assert "cache/" in content
        assert "logs/" in content

    def test_gitignore_false_skips(self, tmp_path):
        io = ProjectIO(root=tmp_path, gitignore=False)
        io.append_gitignore(["test/"])
        # Should not raise and should not create any file
        assert not (tmp_path / ".gitignore").exists()


class TestProducerTracking:
    """Tests for producer tracking functionality."""

    def test_track_producer(self, tmp_path):
        io = ProjectIO(root=tmp_path)

        io.track_producer(
            target=tmp_path / "output.csv",
            producer=tmp_path / "script.py",
            kind="data"
        )

        records = io.producers_of(tmp_path / "output.csv")
        assert len(records) == 1
        assert records[0].kind == "data"

    def test_producers_of(self, tmp_path):
        io = ProjectIO(root=tmp_path)

        io.track_producer(tmp_path / "a.csv", tmp_path / "script.py")
        io.track_producer(tmp_path / "b.csv", tmp_path / "script.py")

        records = io.producers_of(tmp_path / "a.csv")
        assert len(records) == 1
        assert records[0].target == tmp_path / "a.csv"

    def test_outputs_of(self, tmp_path):
        io = ProjectIO(root=tmp_path)

        io.track_producer(tmp_path / "a.csv", tmp_path / "script.py")
        io.track_producer(tmp_path / "b.csv", tmp_path / "script.py")
        io.track_producer(tmp_path / "c.csv", tmp_path / "other.py")

        records = io.outputs_of(tmp_path / "script.py")
        assert len(records) == 2

        targets = {r.target for r in records}
        assert tmp_path / "a.csv" in targets
        assert tmp_path / "b.csv" in targets

    def test_producer_record_dataclass(self, tmp_path):
        record = ProducerRecord(
            target=tmp_path / "out.csv",
            producer=tmp_path / "script.py",
            kind="checkpoint"
        )
        assert record.target == tmp_path / "out.csv"
        assert record.producer == tmp_path / "script.py"
        assert record.kind == "checkpoint"


class TestTreeRendering:
    """Tests for ASCII tree display."""

    def test_tree_basic(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        io = ProjectIO(root=tmp_path, auto_create=False)

        tree = io.tree(tmp_path)
        assert "subdir" in tree

    def test_tree_max_depth(self, tmp_path):
        (tmp_path / "a" / "b" / "c" / "d").mkdir(parents=True)
        io = ProjectIO(root=tmp_path, auto_create=False)

        tree = io.tree(tmp_path, max_depth=2)
        assert "a" in tree
        assert "b" in tree

    def test_tree_with_files(self, tmp_path):
        (tmp_path / "dir").mkdir()
        (tmp_path / "file.txt").touch()
        io = ProjectIO(root=tmp_path, auto_create=False)

        tree_no_files = io.tree(tmp_path, files=False)
        tree_with_files = io.tree(tmp_path, files=True)

        assert "file.txt" not in tree_no_files
        assert "file.txt" in tree_with_files


class TestUsingContext:
    """Tests for context manager temporary overrides."""

    def test_using_restores_values(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=True)
        assert io.use_datestamp is True

        with io.using(use_datestamp=False):
            assert io.use_datestamp is False

        assert io.use_datestamp is True

    def test_using_multiple_overrides(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=True, auto_create=True)

        with io.using(use_datestamp=False, auto_create=False):
            assert io.use_datestamp is False
            assert io.auto_create is False

        assert io.use_datestamp is True
        assert io.auto_create is True

    def test_using_unknown_attribute_raises(self, tmp_path):
        io = ProjectIO(root=tmp_path)

        with pytest.raises(AttributeError):
            with io.using(nonexistent_attr=True):
                pass

    def test_using_yields_self(self, tmp_path):
        io = ProjectIO(root=tmp_path)

        with io.using(dry_run=True) as ctx:
            assert ctx is io


class TestDescribe:
    """Tests for describe() method."""

    def test_describe_returns_dict(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False)
        desc = io.describe()

        assert isinstance(desc, dict)
        assert "root" in desc
        assert "inputs" in desc
        assert "outputs" in desc

    def test_describe_paths_as_strings(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False)
        desc = io.describe()

        assert isinstance(desc["root"], str)
        assert str(tmp_path) in desc["root"]

    def test_describe_includes_settings(self, tmp_path):
        io = ProjectIO(
            root=tmp_path,
            auto_create=False,
            use_datestamp=True,
            dry_run=False
        )
        desc = io.describe()

        assert desc["auto_create"] is False
        assert desc["use_datestamp"] is True
        assert desc["dry_run"] is False


class TestPathFor:
    """Tests for path_for method."""

    def test_path_for_outputs(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.path_for("outputs", "result.csv")
        assert path == tmp_path / "result.csv"

    def test_path_for_with_subdir(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.path_for("logs", "train.log", subdir="exp1")
        assert path == tmp_path / "logs" / "exp1" / "train.log"

    def test_path_for_with_subdir_list(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.path_for("outputs", "result.csv", subdir=["year", "month"])
        assert path == tmp_path / "year" / "month" / "result.csv"

    def test_path_for_unknown_kind_raises(self, tmp_path):
        io = ProjectIO(root=tmp_path)
        with pytest.raises(ValueError):
            io.path_for("unknown_kind", "file.txt")

    def test_path_for_with_extension(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.path_for("cache", "data", ext=".pkl")
        assert path.suffix == ".pkl"


class TestLightningPaths:
    """Tests for Lightning-specific path helpers."""

    def test_checkpoint_path_with_run(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.checkpoint_path("model", run="exp1")
        assert "exp1" in path.parts
        assert path.name == "model.ckpt"

    def test_checkpoint_path_run_with_separator_raises(self, tmp_path):
        io = ProjectIO(root=tmp_path)
        with pytest.raises(ValueError):
            io.checkpoint_path("model", run="exp/1")

    def test_log_path(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.log_path("training", run="exp1")
        assert path.suffix == ".log"
        assert "exp1" in path.parts

    def test_tensorboard_run_returns_directory(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.tensorboard_run(run="exp1")
        assert path.suffix == ""  # Directory, no extension
        assert "exp1" in path.parts


class TestPIOSingleton:
    """Tests for PIO class-level proxy."""

    def test_pio_lazy_instantiation(self, tmp_path):
        # Reset to ensure clean state (stored on PIO class, not metaclass)
        try:
            type.__delattr__(PIO, 'stored_default')
        except AttributeError:
            pass  # Not set yet

        # First access should create default
        default = PIO.default
        assert default is not None
        assert PIO.stored_default is not None

    def test_pio_attribute_forwarding(self, tmp_path):
        # Set up a fresh default
        new_io = ProjectIO(root=tmp_path, auto_create=False)
        PIO.stored_default = new_io

        # Verify forwarding works
        assert PIO.root == tmp_path
        assert PIO.auto_create is False

    def test_pio_setter_forwarding(self, tmp_path):
        new_io = ProjectIO(root=tmp_path)
        PIO.stored_default = new_io

        PIO.use_datestamp = False
        assert PIO.stored_default.use_datestamp is False

    def test_pio_method_forwarding(self, tmp_path):
        new_io = ProjectIO(root=tmp_path, use_datestamp=False, auto_create=False)
        PIO.stored_default = new_io

        path = PIO.checkpoint_path("model")
        assert "checkpoints" in str(path)


class TestResourcePath:
    """Tests for resource path discovery."""

    def test_resource_path_fallback_to_cwp(self, tmp_path):
        os.chdir(tmp_path)
        io = ProjectIO(root=tmp_path, package=None, auto_create=False)
        # resources falls back to cwp/resources (cwp is set at instantiation)
        assert io.resources == io.cwp / "resources"

    def test_resource_path_must_exist_raises(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False)

        with pytest.raises(FileNotFoundError):
            io.resource_path("nonexistent.txt", must_exist=True)

    def test_resource_path_create(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=True)
        (tmp_path / "resources").mkdir(parents=True)

        path = io.resource_path("new_dir", must_exist=False, create=True)
        assert path.exists()


class TestExtensionHandling:
    """Tests for file extension normalization."""

    def test_extension_with_dot(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.checkpoint_path("model", ext=".ckpt")
        assert path.suffix == ".ckpt"

    def test_extension_without_dot(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.checkpoint_path("model", ext="ckpt")
        assert path.suffix == ".ckpt"

    def test_no_double_extension(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
        path = io.checkpoint_path("model.ckpt", ext=".ckpt")
        assert str(path).count(".ckpt") == 1
