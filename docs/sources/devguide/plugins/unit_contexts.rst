.. _plugin-unit-contexts:

=============
Unit contexts
=============

Unit contexts let a plugin provide metadata-dependent unit conversions without
adding domain-specific traits to core objects.

For example, an NMR plugin can convert between ppm and Hz using acquisition
frequency metadata stored on a coordinate. The plugin registers a context setup
function, an optional predicate, and an optional argument extractor:

.. code-block:: python

    def register_unit_contexts(self) -> list[dict]:
        return [
            {
                "name": "nmr",
                "func": set_nmr_context,
                "predicate": applies_to_coord,
                "argument_extractor": get_larmor_frequency,
                "description": "NMR ppm/frequency conversion context",
            },
        ]

The core asks the registry for an applicable context during unit conversion.
If no plugin context applies, normal Pint conversion is used.
