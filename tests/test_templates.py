from pathlib import Path
from projio.core import ProjectIO, LIGHTNING_TEMPLATES, CORE_TEMPLATES


def test_lightning_templates_registered_separately(tmp_path):
    io = ProjectIO(root=tmp_path)
    # Both lightning and core templates should be present
    assert set(LIGHTNING_TEMPLATES).issubset(io.templates)
    assert set(CORE_TEMPLATES).issubset(io.templates)


def test_tensorboard_template_uses_lightning_root(tmp_path):
    io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
    path = io.template_path("tensorboard", run="abc")
    assert path == tmp_path / "lightning" / "tensorboard" / "abc"


def test_filtered_matrix_template_uses_outputs(tmp_path):
    io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=False)
    paths = io.template_path("filtered_matrix", run="exp1")
    assert paths["matrix"] == tmp_path / "matrix.mtx"
    assert paths["barcodes"] == tmp_path / "barcodes.tsv.gz"
    assert paths["features"] == tmp_path / "features.tsv.gz"


def test_template_datestamp_override_false(tmp_path):
    io = ProjectIO(root=tmp_path, auto_create=False, use_datestamp=True)
    io.datestamp_value = lambda timestamp=None: "DATE"
    path = io.template_path("tensorboard", run="abc", datestamp=False)
    # datestamp override should prevent date directory even though use_datestamp True
    assert path == tmp_path / "lightning" / "tensorboard" / "abc"
