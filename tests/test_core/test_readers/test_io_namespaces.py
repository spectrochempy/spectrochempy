"""
Tests for the namespace-based I/O API.

These tests verify that ``scp.<namespace>.read(...)`` and
``scp.<namespace>.write(...)`` delegate correctly to the existing
public API without changing behavior.
"""

import pytest

import spectrochempy as scp


class TestIONamespacesExist:
    """Verify that I/O namespaces are exposed on the spectrochempy package."""

    @pytest.mark.parametrize(
        "ns",
        [
            "jcamp",
            "csv",
            "matlab",
            "omnic",
            "opus",
            "quadera",
            "soc",
            "spc",
            "wire",
            "labspec",
        ],
    )
    def test_namespace_exists(self, ns):
        assert hasattr(scp, ns), f"scp.{ns} should exist"

    def test_namespace_names_in_dir(self):
        names = dir(scp)
        assert "jcamp" in names
        assert "csv" in names
        assert "omnic" in names


class TestIONamespaceRead:
    """Verify that namespace ``read()`` delegates to the top-level reader."""

    def test_jcamp_read_delegates(self):
        assert scp.jcamp.read is scp.read_jcamp

    def test_csv_read_delegates(self):
        assert scp.csv.read is scp.read_csv

    def test_matlab_read_delegates(self):
        assert scp.matlab.read is scp.read_matlab

    def test_omnic_read_delegates(self):
        assert scp.omnic.read is scp.read_omnic

    def test_opus_read_delegates(self):
        assert scp.opus.read is scp.read_opus

    def test_quadera_read_delegates(self):
        assert scp.quadera.read is scp.read_quadera

    def test_soc_read_delegates(self):
        assert scp.soc.read is scp.read_soc

    def test_spc_read_delegates(self):
        assert scp.spc.read is scp.read_spc

    def test_wire_read_delegates(self):
        assert scp.wire.read is scp.read_wire

    def test_labspec_read_delegates(self):
        assert scp.labspec.read is scp.read_labspec


class TestIONamespaceWrite:
    """Verify that namespace ``write()`` delegates to the top-level writer."""

    def test_jcamp_write_delegates(self):
        assert scp.jcamp.write is scp.write_jcamp

    def test_csv_write_delegates(self):
        assert scp.csv.write is scp.write_csv

    def test_matlab_write_delegates(self):
        assert scp.matlab.write is scp.write_matlab


class TestIONamespaceReadOnly:
    """Verify that read-only namespaces do not expose ``write()``."""

    @pytest.mark.parametrize(
        "ns",
        ["omnic", "opus", "quadera", "soc", "spc", "wire", "labspec"],
    )
    def test_read_only_namespace_no_write(self, ns):
        namespace = getattr(scp, ns)
        with pytest.raises(AttributeError):
            _ = namespace.write


class TestExistingAPIUnchanged:
    """Verify that all existing public aliases remain intact."""

    def test_top_level_read_aliases_still_exist(self):
        assert callable(scp.read_jcamp)
        assert callable(scp.read_csv)
        assert callable(scp.read_omnic)
        assert callable(scp.read_opus)

    def test_top_level_write_aliases_still_exist(self):
        assert callable(scp.write_jcamp)
        assert callable(scp.write_csv)
        assert callable(scp.write_matlab)

    def test_generic_read_untouched(self):
        assert callable(scp.read)

    def test_generic_write_untouched(self):
        assert callable(scp.write)

    def test_reader_aliases_untouched(self):
        assert scp.read_mat is scp.read_matlab
