# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for PluginRegistry and specialised registries."""

from spectrochempy.plugins.capabilities import PluginCapability
from spectrochempy.plugins.registries import IORegistry
from spectrochempy.plugins.registries import MetadataRegistry
from spectrochempy.plugins.registries import ProcessingRegistry
from spectrochempy.plugins.registries import VisualizationRegistry
from spectrochempy.plugins.registry import PluginRegistry
from spectrochempy.plugins.registry import registry as global_registry

# ------------------------------------------------------------------
# PluginRegistry — composite structure
# ------------------------------------------------------------------


def test_composite_structure():
    """PluginRegistry composes all specialised registries."""
    registry = PluginRegistry()
    assert isinstance(registry.io, IORegistry)
    assert isinstance(registry.processing, ProcessingRegistry)
    assert isinstance(registry.visualization, VisualizationRegistry)
    assert isinstance(registry.metadata, MetadataRegistry)


def test_independent_instances():
    """PluginRegistry() now creates independent instances."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()
    assert r1 is not r2
    assert r1 is not global_registry


def test_global_registry_is_instance():
    """The module-level registry is a PluginRegistry instance."""
    assert isinstance(global_registry, PluginRegistry)


# ------------------------------------------------------------------
# PluginRegistry — backward-compatible forwarding
# ------------------------------------------------------------------


def test_register_reader():
    registry = PluginRegistry()

    def dummy_reader(path):
        ...

    registry.register_reader("test", dummy_reader, description="test reader")
    info = registry.get_reader("test")
    assert info is not None
    assert info["func"] is dummy_reader
    assert info["description"] == "test reader"


def test_available_readers():
    registry = PluginRegistry()

    def dummy_reader(path):
        ...

    registry.register_reader("test", dummy_reader)
    assert "test" in registry.available_readers


def test_register_processor():
    registry = PluginRegistry()

    def dummy_processor(data):
        ...

    registry.register_processor("test_proc", dummy_processor)
    info = registry.get_processor("test_proc")
    assert info is not None
    assert info["func"] is dummy_processor


def test_register_accessor():
    registry = PluginRegistry()

    def dummy_accessor(dataset):
        ...

    registry.register_accessor(
        "test_accessor", dummy_accessor, description="test accessor"
    )
    info = registry.get_accessor("test_accessor")
    assert info is not None
    assert info["obj"] is dummy_accessor
    assert info["description"] == "test accessor"
    assert "test_accessor" in registry.available_accessors


def test_register_writer():
    registry = PluginRegistry()

    def dummy_writer(data, path):
        ...

    registry.register_writer("test_writer", dummy_writer)
    info = registry.get_writer("test_writer")
    assert info is not None
    assert info["func"] is dummy_writer


def test_dtype_handler():
    registry = PluginRegistry()
    handler = object()
    registry.register_dtype_handler("quaternion", handler)
    assert registry.has_dtype_handler("quaternion")
    assert registry.get_dtype_handler("quaternion") is handler


def test_unit_context():
    registry = PluginRegistry()

    def setup():
        ...

    registry.register_unit_context("nmr", setup)
    assert registry.get_unit_context("nmr") is setup


def test_register_plugin():
    registry = PluginRegistry()
    plugin = object()
    registry.register_plugin("test_plugin", plugin)
    assert registry.get_plugin("test_plugin") is plugin
    assert "test_plugin" in registry.available_plugins


def test_filetype():
    registry = PluginRegistry()
    info = {"ext": ".xyz", "reader": "test"}
    registry.register_filetype("xyz", info)
    assert registry.get_filetype("xyz") == info
    assert "xyz" in registry.available_filetypes


def test_clear():
    """clear() removes all entries from the registry."""
    registry = PluginRegistry()

    def dummy():
        ...

    registry.register_reader("r1", dummy)
    registry.register_writer("w1", dummy)
    registry.register_processor("p1", dummy)
    registry.register_accessor("a1", dummy)
    registry.register_plugin("p1", dummy)
    registry.register_filetype("ext", {})
    registry.register_dtype_handler("d1", dummy)
    registry.register_unit_context("u1", dummy)

    registry.clear()

    assert registry.available_readers == {}
    assert registry.available_writers == {}
    assert registry.available_processors == {}
    assert registry.available_accessors == {}
    assert registry.available_plugins == {}
    assert registry.available_filetypes == {}
    assert registry.get_reader("r1") is None
    assert registry.get_dtype_handler("d1") is None


def test_registries_isolated():
    """Two independent registries do not share state."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()

    def dummy():
        ...

    r1.register_reader("shared", dummy)
    assert r2.get_reader("shared") is None


# ------------------------------------------------------------------
# PluginRegistry — forwarding targets sub-registries
# ------------------------------------------------------------------


def test_register_reader_forwards_to_io():
    registry = PluginRegistry()

    def dummy(path):
        ...

    registry.register_reader("fwd", dummy)
    assert registry.io.get_reader("fwd") is not None


def test_register_writer_forwards_to_io():
    registry = PluginRegistry()

    def dummy(data, path):
        ...

    registry.register_writer("fwd", dummy)
    assert registry.io.get_writer("fwd") is not None


def test_register_processor_forwards_to_processing():
    registry = PluginRegistry()

    def dummy(data):
        ...

    registry.register_processor("fwd", dummy)
    assert registry.processing.get_processor("fwd") is not None


def test_register_plugin_forwards_to_metadata():
    registry = PluginRegistry()
    registry.register_plugin("fwd", "plugin")
    assert registry.metadata.get_plugin("fwd") == "plugin"


def test_register_accessor_forwards_to_extensions():
    registry = PluginRegistry()

    def dummy(data):
        ...

    registry.register_accessor("fwd", dummy)
    assert registry.extensions.get("accessor", "fwd") is not None


def test_dtype_handler_forwards_to_processing():
    registry = PluginRegistry()
    registry.register_dtype_handler("fwd", object())
    assert registry.processing.has_dtype_handler("fwd")


# ------------------------------------------------------------------
# IORegistry — isolated tests
# ------------------------------------------------------------------


def test_io_registry_independent():
    io = IORegistry()

    def dummy(path):
        ...

    io.register_reader("r", dummy)
    io.register_writer("w", dummy)
    io.register_filetype("ext", {})

    assert "r" in io.available_readers
    assert "w" in io.available_writers
    assert "ext" in io.available_filetypes

    io.clear()
    assert io.available_readers == {}
    assert io.available_writers == {}
    assert io.available_filetypes == {}


# ------------------------------------------------------------------
# ProcessingRegistry — isolated tests
# ------------------------------------------------------------------


def test_processing_registry_independent():
    pr = ProcessingRegistry()

    def dummy(data):
        ...

    def setup():
        ...

    pr.register_processor("p", dummy)
    pr.register_unit_context("ctx", setup)
    pr.register_dtype_handler("dtype", object())

    assert "p" in pr.available_processors
    assert pr.get_unit_context("ctx") is setup
    assert pr.has_dtype_handler("dtype")

    pr.clear()
    assert pr.available_processors == {}
    assert pr.get_unit_context("ctx") is None
    assert not pr.has_dtype_handler("dtype")


# ------------------------------------------------------------------
# VisualizationRegistry — isolated tests
# ------------------------------------------------------------------


def test_visualization_registry():
    vr = VisualizationRegistry()

    def plot():
        ...

    vr.register_visualizer("myplot", plot)
    assert vr.get_visualizer("myplot") is not None
    assert "myplot" in vr.available_visualizers

    vr.clear()
    assert vr.available_visualizers == {}


# ------------------------------------------------------------------
# MetadataRegistry — isolated tests
# ------------------------------------------------------------------


def test_metadata_registry():
    mr = MetadataRegistry()
    mr.register_plugin("p1", "descriptor")
    assert mr.get_plugin("p1") == "descriptor"
    assert "p1" in mr.available_plugins

    mr.clear()
    assert mr.available_plugins == {}


# ------------------------------------------------------------------
# Specialised registries don't leak across PluginRegistry instances
# ------------------------------------------------------------------


def test_sub_registries_are_independent_instances():
    r1 = PluginRegistry()
    r2 = PluginRegistry()

    assert r1.io is not r2.io
    assert r1.processing is not r2.processing
    assert r1.visualization is not r2.visualization
    assert r1.metadata is not r2.metadata


# ------------------------------------------------------------------
# Contribution dataclasses
# ------------------------------------------------------------------


def test_reader_contribution():
    from spectrochempy.plugins.contributions import ReaderContribution

    def dummy(path):
        ...

    c = ReaderContribution(
        name="test", func=dummy, description="desc", extensions=[".ext"]
    )
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"
    assert c.extensions == [".ext"]


def test_reader_contribution_defaults():
    from spectrochempy.plugins.contributions import ReaderContribution

    def dummy(path):
        ...

    c = ReaderContribution(name="test", func=dummy)
    assert c.description == ""
    assert c.extensions is None


def test_writer_contribution():
    from spectrochempy.plugins.contributions import WriterContribution

    def dummy(data, path):
        ...

    c = WriterContribution(name="test", func=dummy, description="desc")
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"


def test_visualizer_contribution():
    from spectrochempy.plugins.contributions import VisualizerContribution

    def dummy(data):
        ...

    c = VisualizerContribution(name="test", func=dummy, description="desc")
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"


def test_processor_contribution():
    from spectrochempy.plugins.contributions import ProcessorContribution

    def dummy(data):
        ...

    c = ProcessorContribution(name="test", func=dummy, description="desc")
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"


def test_reader_from_dict():
    from spectrochempy.plugins.contributions import reader_from_dict

    def dummy(path):
        ...

    c = reader_from_dict(
        {"name": "r", "func": dummy, "description": "d", "extensions": [".x"]}
    )
    assert c.name == "r"
    assert c.func is dummy
    assert c.extensions == [".x"]


def test_writer_from_dict():
    from spectrochempy.plugins.contributions import writer_from_dict

    def dummy(data, path):
        ...

    c = writer_from_dict({"name": "w", "func": dummy})
    assert c.name == "w"
    assert c.func is dummy
    assert c.description == ""


def test_visualizer_from_dict():
    from spectrochempy.plugins.contributions import visualizer_from_dict

    def dummy(data):
        ...

    c = visualizer_from_dict({"name": "v", "func": dummy, "description": "viz"})
    assert c.name == "v"
    assert c.func is dummy
    assert c.description == "viz"


def test_processor_from_dict():
    from spectrochempy.plugins.contributions import processor_from_dict

    def dummy(data):
        ...

    c = processor_from_dict({"name": "p", "func": dummy, "description": "proc"})
    assert c.name == "p"
    assert c.func is dummy
    assert c.description == "proc"


# ------------------------------------------------------------------
# ExtensionRegistry
# ------------------------------------------------------------------


def test_extension_registry():
    """ExtensionRegistry stores arbitrary categorised extensions."""
    from spectrochempy.plugins.registries import ExtensionRegistry

    reg = ExtensionRegistry()
    reg.register("analysis", "pca", lambda: 42, description="PCA")
    reg.register("analysis", "mcr", lambda: 43)
    reg.register(
        "simulation", "equilibrium", lambda: 44, metadata={"engine": "cantera"}
    )

    assert reg.get("analysis", "pca")["description"] == "PCA"
    assert reg.get("analysis", "pca")["obj"]() == 42
    assert reg.get("analysis", "mcr")["obj"]() == 43
    assert reg.get("simulation", "equilibrium")["metadata"]["engine"] == "cantera"
    assert reg.get("missing", "x") is None
    assert reg.get("analysis", "missing") is None


def test_extension_list_category():
    """ExtensionRegistry.list_category returns all items in a category."""
    from spectrochempy.plugins.registries import ExtensionRegistry

    reg = ExtensionRegistry()
    reg.register("analysis", "pca", 1)
    reg.register("analysis", "mcr", 2)
    reg.register("simulation", "eq", 3)

    analyses = reg.list_category("analysis")
    assert "pca" in analyses
    assert "mcr" in analyses
    assert "eq" not in analyses
    assert reg.list_category("nonexistent") == {}


def test_extension_categories():
    """ExtensionRegistry.categories returns all category names."""
    from spectrochempy.plugins.registries import ExtensionRegistry

    reg = ExtensionRegistry()
    reg.register("a", "x", 1)
    reg.register("b", "y", 2)
    assert sorted(reg.categories) == ["a", "b"]


def test_extension_clear():
    """ExtensionRegistry.clear() removes all entries."""
    from spectrochempy.plugins.registries import ExtensionRegistry

    reg = ExtensionRegistry()
    reg.register("analysis", "pca", 1)
    reg.clear()
    assert reg.list_category("analysis") == {}
    assert reg.categories == []


def test_extension_registry_in_composite():
    """PluginRegistry includes an ExtensionRegistry as ``.extensions``."""
    registry = PluginRegistry()
    from spectrochempy.plugins.registries import ExtensionRegistry

    assert isinstance(registry.extensions, ExtensionRegistry)

    registry.extensions.register("analysis", "test", 42)
    assert registry.extensions.get("analysis", "test")["obj"] == 42


# ------------------------------------------------------------------
# PluginRegistry.get_by_capability
# ------------------------------------------------------------------


def test_get_by_capability_reader():
    """get_by_capability(READER) returns registered readers."""
    registry = PluginRegistry()

    def dummy(path):
        ...

    registry.register_reader("myfmt", dummy, description="test")
    results = registry.get_by_capability(PluginCapability.READER)
    assert any(r["name"] == "myfmt" for r in results)


def test_get_by_capability_writer():
    """get_by_capability(WRITER) returns registered writers."""
    registry = PluginRegistry()

    def dummy(data, path):
        ...

    registry.register_writer("myfmt", dummy)
    results = registry.get_by_capability(PluginCapability.WRITER)
    assert any(r["name"] == "myfmt" for r in results)


def test_get_by_capability_processor():
    """get_by_capability(PROCESSOR) returns registered processors."""
    registry = PluginRegistry()

    def dummy(data):
        ...

    registry.register_processor("smooth", dummy)
    results = registry.get_by_capability(PluginCapability.PROCESSOR)
    assert any(r["name"] == "smooth" for r in results)


def test_get_by_capability_analysis():
    """get_by_capability(ANALYSIS) returns items from extensions registry."""
    registry = PluginRegistry()
    registry.extensions.register("analysis", "pca", lambda: 42)
    results = registry.get_by_capability(PluginCapability.ANALYSIS)
    assert any(r["name"] == "pca" for r in results)


def test_get_by_capability_simulation():
    """get_by_capability(SIMULATION) returns items from extensions registry."""
    registry = PluginRegistry()
    registry.extensions.register("simulation", "cantera_eq", lambda: 42)
    results = registry.get_by_capability(PluginCapability.SIMULATION)
    assert any(r["name"] == "cantera_eq" for r in results)


def test_get_by_capability_accessor():
    """get_by_capability(ACCESSOR) returns items from extensions registry."""
    registry = PluginRegistry()
    registry.extensions.register("accessor", "iris_kernel_matrix", lambda: 42)
    results = registry.get_by_capability(PluginCapability.ACCESSOR)
    assert any(r["name"] == "iris_kernel_matrix" for r in results)


def test_get_by_capability_empty():
    """get_by_capability returns empty list when nothing is registered."""
    registry = PluginRegistry()
    assert registry.get_by_capability(PluginCapability.ANALYSIS) == []
    assert registry.get_by_capability(PluginCapability.SIMULATION) == []
    assert registry.get_by_capability(PluginCapability.ACCESSOR) == []


# ------------------------------------------------------------------
# Contribution dataclasses (analysis, simulation)
# ------------------------------------------------------------------


def test_analysis_contribution():
    """AnalysisContribution stores name, func, description."""
    from spectrochempy.plugins.contributions import AnalysisContribution

    def dummy(data):
        ...

    c = AnalysisContribution(name="pca", func=dummy, description="PCA")
    assert c.name == "pca"
    assert c.func is dummy
    assert c.description == "PCA"


def test_simulation_contribution():
    """SimulationContribution stores name, func, description."""
    from spectrochempy.plugins.contributions import SimulationContribution

    def dummy(data):
        ...

    c = SimulationContribution(name="equilibrium", func=dummy)
    assert c.name == "equilibrium"
    assert c.func is dummy
    assert c.description == ""


def test_accessor_contribution():
    """AccessorContribution stores name, func, description."""
    from spectrochempy.plugins.contributions import AccessorContribution

    def dummy(data):
        ...

    c = AccessorContribution(
        name="iris_kernel_matrix", func=dummy, description="IRIS kernel"
    )
    assert c.name == "iris_kernel_matrix"
    assert c.func is dummy
    assert c.description == "IRIS kernel"


def test_analysis_from_dict():
    """analysis_from_dict converts a dict to AnalysisContribution."""
    from spectrochempy.plugins.contributions import analysis_from_dict

    def dummy(data):
        ...

    c = analysis_from_dict({"name": "pca", "func": dummy, "description": "PCA"})
    assert c.name == "pca"
    assert c.func is dummy
    assert c.description == "PCA"


def test_simulation_from_dict():
    """simulation_from_dict converts a dict to SimulationContribution."""
    from spectrochempy.plugins.contributions import simulation_from_dict

    def dummy(data):
        ...

    c = simulation_from_dict({"name": "eq", "func": dummy})
    assert c.name == "eq"
    assert c.func is dummy


def test_accessor_from_dict():
    """accessor_from_dict converts a dict to AccessorContribution."""
    from spectrochempy.plugins.contributions import accessor_from_dict

    def dummy(data):
        ...

    c = accessor_from_dict(
        {"name": "iris_kernel_matrix", "func": dummy, "description": "IRIS kernel"}
    )
    assert c.name == "iris_kernel_matrix"
    assert c.func is dummy
    assert c.description == "IRIS kernel"
