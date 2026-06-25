.. _plugin-testing:

================
Testing a Plugin
================

SpectroChemPy provides :class:`~spectrochempy.testing.plugins.PluginTestHarness`
to help you write isolated tests for your plugin.

The test harness creates a fresh
:class:`~spectrochempy.plugins.registry.PluginRegistry` and
:class:`~spectrochempy.plugins.manager.PluginManager` for every test,
preventing state leakage between tests.

Quick start
===========

.. code-block:: python

    from spectrochempy.testing.plugins import PluginTestHarness
    from spectrochempy.api.plugins import PluginState

    from myplugin import MyPlugin


    def test_registration():
        harness = PluginTestHarness()
        harness.register(MyPlugin())

        # Contributions are registered
        reader = harness.get_reader("myformat")
        assert reader is not None

        # Lifecycle state is correct
        assert harness.get_plugin_state("myplugin") == PluginState.ACTIVE

        # No failed plugins
        assert harness.get_failed_plugins() == {}


    def test_context_manager():
        with PluginTestHarness() as harness:
            harness.register(MyPlugin())
            assert harness.has_plugin("myplugin")
        # Registry is automatically cleared on exit


Available helpers
=================

+----------------------------------------------------+----------------------------------------------------------+
| Method                                             | Purpose                                                  |
+====================================================+==========================================================+
| ``register(plugin)``                               | Register a plugin in the isolated manager                |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_reader(name)``                               | Look up a registered reader                              |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_writer(name)``                               | Look up a registered writer                              |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_processor(name)``                            | Look up a registered processor                           |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_visualizer(name)``                           | Look up a registered visualizer                          |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_plugin_state(name)``                         | Inspect plugin lifecycle state                           |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_plugin_descriptor(name)``                    | Get full plugin descriptor snapshot                      |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_active_plugins()``                           | List all ACTIVE plugins                                  |
+----------------------------------------------------+----------------------------------------------------------+
| ``get_failed_plugins()``                           | List all FAILED plugins with errors                      |
+----------------------------------------------------+----------------------------------------------------------+

Checking compatibility
======================

Use the validation helpers to verify your plugin before registration:

.. code-block:: python

    from spectrochempy.api.plugins import check_plugin_compatibility

    plugin = MyPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"

See :ref:`plugin-validation` for details.

Testing with optional dependencies
==================================

If your plugin has optional dependencies, test both paths:

.. code-block:: python

    def test_without_optional_dep():
        """Plugin handles missing optional dependency gracefully."""
        harness = PluginTestHarness()
        harness.register(MyPlugin())
        # Plugin should still be registerable, maybe in FAILED state
        # if the dep is required for registration
        ...


    def test_with_optional_dep():
        """Plugin works fully when optional dependency is installed."""
        pytest.importorskip("my_optional_lib")
        harness = PluginTestHarness()
        harness.register(MyPlugin())
        assert harness.get_plugin_state("myplugin") == PluginState.ACTIVE

.. seealso::

    :ref:`plugin-architecture` for the runtime architecture overview and
    :ref:`plugin-author-guide` for the plugin author guide.
