"""Tests for project_io.callbacks module."""
import pytest
from pathlib import Path
import tempfile

from projio.core import ProjectIO
from projio.callbacks import IOCheckpointCallback, IOLogCallback


class TestIOCheckpointCallback:
    """Tests for IOCheckpointCallback."""

    def test_init_with_defaults(self, tmp_path):
        cb = IOCheckpointCallback()
        assert cb.io is not None
        assert cb.run is None
        assert cb.datestamp is None

    def test_init_with_io(self, tmp_path):
        io = ProjectIO(root=tmp_path)
        cb = IOCheckpointCallback(io=io, run="exp1")
        assert cb.io is io
        assert cb.run == "exp1"

    def test_checkpoint_dir(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=False)
        cb = IOCheckpointCallback(io=io, run="exp1")

        ckpt_dir = cb.checkpoint_dir
        assert "exp1" in str(ckpt_dir)

    def test_get_checkpoint_path(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=False, auto_create=False)
        cb = IOCheckpointCallback(io=io, run="exp1")

        path = cb.get_checkpoint_path(epoch=5, step=1000)
        assert path.suffix == ".ckpt"
        assert "05-001000" in path.name

    def test_custom_filename_format(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=False, auto_create=False)
        cb = IOCheckpointCallback(
            io=io,
            filename="epoch{epoch:03d}_step{step:08d}"
        )

        path = cb.get_checkpoint_path(epoch=10, step=5000)
        assert "epoch010_step00005000" in path.name

    def test_track_producer_disabled_by_default(self, tmp_path):
        io = ProjectIO(root=tmp_path)
        cb = IOCheckpointCallback(io=io, track_producer=False)

        # Should have no producer tracking
        assert len(io.producers) == 0


class TestIOLogCallback:
    """Tests for IOLogCallback."""

    def test_init_with_defaults(self, tmp_path):
        cb = IOLogCallback()
        assert cb.io is not None
        assert cb.run is None
        assert cb.datestamp is None

    def test_init_with_io(self, tmp_path):
        io = ProjectIO(root=tmp_path)
        cb = IOLogCallback(io=io, run="exp1")
        assert cb.io is io
        assert cb.run == "exp1"

    def test_log_dir(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=False)
        cb = IOLogCallback(io=io, run="exp1")

        log_dir = cb.log_dir
        assert "exp1" in str(log_dir)
        assert "tensorboard" in str(log_dir)

    def test_log_dir_with_datestamp(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=True, datestamp_in="dirs")
        io.datestamp_value = lambda ts=None: "2024_03_15"
        cb = IOLogCallback(io=io, run="exp1", datestamp=True)

        log_dir = cb.log_dir
        assert "2024_03_15" in str(log_dir)


class TestCallbackIntegration:
    """Integration tests for callbacks."""

    def test_shared_io_instance(self, tmp_path):
        io = ProjectIO(root=tmp_path, use_datestamp=False)

        ckpt_cb = IOCheckpointCallback(io=io, run="exp1")
        log_cb = IOLogCallback(io=io, run="exp1")

        # Both callbacks should share the same IO
        assert ckpt_cb.io is log_cb.io

        # Paths should be consistent
        assert "exp1" in str(ckpt_cb.checkpoint_dir)
        assert "exp1" in str(log_cb.log_dir)

    def test_dry_run_mode(self, tmp_path):
        io = ProjectIO(root=tmp_path, dry_run=True)
        cb = IOCheckpointCallback(io=io)

        path = cb.get_checkpoint_path(epoch=1, step=100)
        assert not path.parent.exists()

    def test_auto_create_directories(self, tmp_path):
        io = ProjectIO(root=tmp_path, auto_create=True, dry_run=False)
        cb = IOLogCallback(io=io, run="exp1")

        log_dir = cb.log_dir
        assert log_dir.exists()
