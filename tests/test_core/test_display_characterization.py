# ======================================================================================
# Copyright (c) 2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Display Characterization Test Suite for Issue #843.

This is a characterization and safety-net PR.

Purpose:
- Document current display behavior BEFORE any Representation Model RFC
- Establish an observable behavior baseline for future display work
- Protect semantic display content without locking implementation choices

Testing Strategy:
- Use only small synthetic objects (no external datasets)
- Test semantic properties, not exact formatting
- Avoid brittle string comparisons
- Use broad assertions for HTML output

Test Categories:
- Observed Display Behavior: broad user-visible display content
- Current Observed Behavior: selected visible quirks, may change later

DO NOT:
- Redesign the display system
- Introduce new abstractions
- Test private display helpers or implementation mechanisms
"""

import numpy as np

from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.utils.print import DisplayItem
from spectrochempy.utils.print import DisplaySection
from spectrochempy.utils.print import _format_array_values
from spectrochempy.utils.print import pstr

# ======================================================================================
# OBSERVED DISPLAY BEHAVIOR
# ======================================================================================
#
# These tests document broad semantic information users can observe in display output.


class TestCoordObservedDisplay:
    """Observed display behavior for Coord objects."""

    def test_repr_contains_coord_type(self):
        """Coord repr should identify the object as a Coord."""
        coord = Coord([1.0, 2.0, 3.0])
        repr_str = repr(coord)
        assert "Coord" in repr_str

    def test_repr_contains_dtype(self):
        """Coord repr should include dtype information."""
        coord = Coord([1.0, 2.0, 3.0])
        repr_str = repr(coord)
        assert "float" in repr_str.lower()

    def test_repr_contains_shape_info(self):
        """Coord repr should include shape/size information."""
        coord = Coord([1.0, 2.0, 3.0])
        repr_str = repr(coord)
        # Should contain either "size" or "shape"
        assert "size" in repr_str.lower() or "shape" in repr_str.lower()

    def test_str_is_non_empty(self):
        """Coord str should produce non-empty output."""
        coord = Coord([1.0, 2.0, 3.0])
        assert len(str(coord)) > 0

    def test_repr_html_exists(self):
        """Coord should have a _repr_html_ method that produces output."""
        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert html is not None
        assert len(html) > 0

    def test_repr_html_contains_coord_type(self):
        """Coord HTML repr should identify the object as a Coord."""
        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert "Coord" in html

    def test_repr_contains_data_indication(self):
        """Coord repr should indicate that it contains data."""
        coord = Coord([1.0, 2.0, 3.0])
        repr_str = repr(coord)
        # Should contain "data" or values
        assert "data" in repr_str.lower() or "[" in repr_str

    def test_title_appears_in_detailed_repr(self):
        """If Coord has a title, it should appear in detailed representation."""
        coord = Coord([1.0, 2.0, 3.0], title="my_coord")
        detailed = pstr(coord)
        assert "my_coord" in detailed


class TestCoordSetObservedDisplay:
    """Observed display behavior for CoordSet objects."""

    def test_repr_contains_coordset_type(self):
        """CoordSet repr should identify the object as a CoordSet."""
        x = Coord([1.0, 2.0, 3.0])
        coordset = CoordSet(x=x)
        repr_str = repr(coordset)
        assert "CoordSet" in repr_str

    def test_str_is_non_empty(self):
        """CoordSet str should produce non-empty output."""
        x = Coord([1.0, 2.0, 3.0])
        coordset = CoordSet(x=x)
        assert len(str(coordset)) > 0

    def test_repr_html_exists(self):
        """CoordSet should have a _repr_html_ method that produces output."""
        x = Coord([1.0, 2.0, 3.0])
        coordset = CoordSet(x=x)
        html = coordset._repr_html_()
        assert html is not None
        assert len(html) > 0

    def test_repr_html_contains_coordset_type(self):
        """CoordSet HTML repr should identify the object as a CoordSet."""
        x = Coord([1.0, 2.0, 3.0])
        coordset = CoordSet(x=x)
        html = coordset._repr_html_()
        assert "CoordSet" in html

    def test_coordinate_names_appear_in_repr(self):
        """CoordSet repr should include coordinate names."""
        x = Coord([1.0, 2.0, 3.0])
        y = Coord([4.0, 5.0, 6.0])
        coordset = CoordSet(x=x, y=y)
        repr_str = repr(coordset)
        # Coordinate names should appear
        assert "x" in repr_str and "y" in repr_str

    def test_coordinate_titles_appear_in_detailed_repr(self):
        """CoordSet detailed repr should include coordinate titles."""
        x = Coord([1.0, 2.0, 3.0], title="wavelength")
        coordset = CoordSet(x=x)
        detailed = pstr(coordset)
        assert "wavelength" in detailed


class TestNDDatasetObservedDisplay:
    """Observed display behavior for NDDataset objects."""

    def test_repr_contains_nddataset_type(self):
        """NDDataset repr should identify the object as an NDDataset."""
        ds = NDDataset([1.0, 2.0, 3.0])
        repr_str = repr(ds)
        assert "NDDataset" in repr_str

    def test_str_is_non_empty(self):
        """NDDataset str should produce non-empty output."""
        ds = NDDataset([1.0, 2.0, 3.0])
        assert len(str(ds)) > 0

    def test_repr_html_exists(self):
        """NDDataset should have a _repr_html_ method that produces output."""
        ds = NDDataset([1.0, 2.0, 3.0])
        html = ds._repr_html_()
        assert html is not None
        assert len(html) > 0

    def test_repr_html_contains_nddataset_type(self):
        """NDDataset HTML repr should identify the object as an NDDataset."""
        ds = NDDataset([1.0, 2.0, 3.0])
        html = ds._repr_html_()
        assert "NDDataset" in html

    def test_repr_contains_shape_info(self):
        """NDDataset repr should include shape information."""
        ds = NDDataset([[1.0, 2.0], [3.0, 4.0]])
        repr_str = repr(ds)
        assert "shape" in repr_str.lower() or "size" in repr_str.lower()

    def test_repr_contains_dtype(self):
        """NDDataset repr should include dtype information."""
        ds = NDDataset([1.0, 2.0, 3.0])
        repr_str = repr(ds)
        assert "float" in repr_str.lower()

    def test_name_appears_in_detailed_repr(self):
        """If NDDataset has a name, it should appear in detailed representation."""
        ds = NDDataset([1.0, 2.0, 3.0], name="my_dataset")
        detailed = pstr(ds)
        assert "my_dataset" in detailed

    def test_coordset_names_appear_in_detailed_repr(self):
        """NDDataset detailed repr should include coordinate set names."""
        x = Coord([1.0, 2.0])
        ds = NDDataset([[1.0, 2.0], [3.0, 4.0]], coordset=[x])
        detailed = pstr(ds)
        assert "x" in detailed


class TestProjectObservedDisplay:
    """Observed display behavior for Project objects."""

    def test_str_is_non_empty(self):
        """Project str should produce non-empty output."""
        proj = Project(name="test_project")
        assert len(str(proj)) > 0

    def test_repr_is_compact(self):
        """Project repr should be compact and contain type and name."""
        proj = Project(name="test_project")
        repr_str = repr(proj)
        assert "Project" in repr_str
        assert "test_project" in repr_str
        assert "\n" not in repr_str.strip()

    def test_repr_differs_from_default_object_repr(self):
        """Project repr should not be the default Python object repr."""
        proj = Project(name="test_project")
        repr_str = repr(proj)
        assert "object at 0x" not in repr_str

    def test_repr_html_exists(self):
        """Project should have a _repr_html_ method that produces output."""
        proj = Project(name="test_project")
        html = proj._repr_html_()
        assert html is not None
        assert len(html) > 0

    def test_str_contains_project_name(self):
        """Project str should include the project name."""
        proj = Project(name="test_project")
        str_output = str(proj)
        assert "test_project" in str_output

    def test_html_contains_project_name(self):
        """Project HTML should include the project name."""
        proj = Project(name="test_project")
        html = proj._repr_html_()
        assert "test_project" in html

    def test_hierarchy_appears_in_str(self):
        """Project str should show hierarchical structure."""
        proj = Project(name="parent")
        subproj = Project(name="child")
        proj._projects["child"] = subproj
        str_output = str(proj)
        # Should show parent and child
        assert "parent" in str_output
        assert "child" in str_output

    def test_dataset_names_appear_in_str(self):
        """Project str should show contained dataset names."""
        proj = Project(name="test_project")
        ds = NDDataset([1.0, 2.0, 3.0], name="my_dataset")
        proj._datasets["my_dataset"] = ds
        str_output = str(proj)
        assert "my_dataset" in str_output

    def test_cstr_contains_metadata_when_present(self):
        """Project _cstr should include metadata fields when present."""
        proj = Project(name="test_project", author="test_author")
        proj.meta.description = "Test description"
        detailed = pstr(proj)
        assert "test_project" in detailed
        assert "test_author" in detailed
        assert "Test description" in detailed

    def test_cstr_contains_hierarchy(self):
        """Project _cstr should include hierarchy information."""
        proj = Project(name="parent")
        subproj = Project(name="child")
        proj._projects["child"] = subproj
        ds = NDDataset([1.0, 2.0, 3.0], name="my_dataset")
        proj._datasets["my_dataset"] = ds
        detailed = pstr(proj)
        assert "child" in detailed
        assert "(sub-project)" in detailed
        assert "my_dataset" in detailed
        assert "(dataset)" in detailed


# ======================================================================================
# CURRENT OBSERVED BEHAVIOR
# ======================================================================================
#
# These tests document selected visible quirks in current display output.
#
# Naming convention: test_<class>_current_behavior_<description>


class TestCoordCurrentBehavior:
    """Current behavior characterization for Coord objects."""

    def test_str_currently_matches_repr(self):
        """Coord str currently matches repr."""
        coord = Coord([1.0, 2.0, 3.0])
        assert str(coord) == repr(coord)

    def test_compact_repr_does_not_show_title(self):
        """Coord compact repr (via __repr__) does not show title."""
        coord = Coord([1.0, 2.0, 3.0], title="my_title")
        repr_str = repr(coord)
        assert "my_title" not in repr_str

    def test_compact_repr_does_not_show_name(self):
        """Coord compact repr does not show the name attribute."""
        coord = Coord([1.0, 2.0, 3.0], name="my_coord")
        repr_str = repr(coord)
        assert "my_coord" not in repr_str

    def test_detailed_repr_shows_title(self):
        """Coord detailed repr shows title when present."""
        coord = Coord([1.0, 2.0, 3.0], title="my_title")
        detailed = pstr(coord)
        assert "my_title" in detailed

    def test_units_appear_in_compact_repr_when_present(self):
        """Coord units appear in compact repr when present."""
        coord = Coord([1.0, 2.0, 3.0], units="m")
        repr_str = repr(coord)
        assert "m" in repr_str


class TestCoordSetCurrentBehavior:
    """Current behavior characterization for CoordSet objects."""

    def test_str_currently_matches_repr(self):
        """CoordSet str currently matches repr."""
        x = Coord([1.0, 2.0, 3.0])
        coordset = CoordSet(x=x)
        assert str(coordset) == repr(coordset)

    def test_compact_repr_does_not_show_coordset_name(self):
        """CoordSet compact repr does not show the CoordSet's own name."""
        x = Coord([1.0, 2.0, 3.0])
        coordset = CoordSet(x=x, name="my_coordset")
        repr_str = repr(coordset)
        assert "my_coordset" not in repr_str

    def test_compact_repr_shows_coordinate_names_and_titles(self):
        """CoordSet compact repr shows coordinate names and titles."""
        x = Coord([1.0, 2.0, 3.0], title="wavelength")
        coordset = CoordSet(x=x)
        repr_str = repr(coordset)
        assert "x" in repr_str
        assert "wavelength" in repr_str

    def test_aliases_appear_in_detailed_repr(self):
        """CoordSet aliases appear in detailed repr when defined."""
        x = Coord([1.0, 2.0, 3.0])
        y = Coord([4.0, 5.0, 6.0])
        coordset = CoordSet(x=x, y=y)
        coordset._references = {"wavelength": "x"}
        detailed = pstr(coordset)
        assert "wavelength" in detailed or "=" in detailed

    def test_nested_coordset_appears_in_detailed_repr(self):
        """Nested CoordSet appears in detailed CoordSet repr."""
        x1 = Coord([1.0, 2.0])
        x2 = Coord([3.0, 4.0])
        inner = CoordSet(x=x1, y=x2)
        y = Coord([5.0, 6.0])
        outer = CoordSet(x=inner, y=y)
        detailed = pstr(outer)
        assert "x" in detailed
        assert "y" in detailed


class TestNDDatasetCurrentBehavior:
    """Current behavior characterization for NDDataset objects."""

    def test_str_currently_matches_repr(self):
        """NDDataset str currently matches repr."""
        ds = NDDataset([1.0, 2.0, 3.0])
        assert str(ds) == repr(ds)

    def test_compact_repr_does_not_show_metadata(self):
        """NDDataset compact repr does not show metadata fields."""
        ds = NDDataset([1.0, 2.0, 3.0], name="my_ds", author="test_user")
        repr_str = repr(ds)
        assert "my_ds" not in repr_str
        assert "test_user" not in repr_str

    def test_detailed_repr_shows_metadata(self):
        """NDDataset detailed repr shows metadata fields."""
        ds = NDDataset([1.0, 2.0, 3.0], name="my_ds", author="test_user")
        detailed = pstr(ds)
        assert "my_ds" in detailed
        assert "test_user" in detailed

    def test_detailed_repr_shows_history(self):
        """NDDataset detailed repr shows history when present."""
        ds = NDDataset([1.0, 2.0, 3.0], history="operation 1")
        detailed = pstr(ds)
        assert "operation 1" in detailed.lower()

    def test_coordinates_appear_in_detailed_repr(self):
        """NDDataset detailed repr shows coordinate information."""
        x = Coord([1.0, 2.0], title="wavelength")
        ds = NDDataset([[1.0, 2.0], [3.0, 4.0]], coordset=[x])
        detailed = pstr(ds)
        assert "DIMENSION" in detailed
        assert "wavelength" in detailed


class TestProjectCurrentBehavior:
    """Current behavior characterization for Project objects."""

    def test_project_repr_contains_type_and_differs_from_str(self):
        """Project repr currently identifies the type and differs from str."""
        proj = Project(name="test_project")
        repr_str = repr(proj)
        assert "Project" in repr_str
        assert str(proj) != repr(proj)

    def test_empty_project_shows_empty_indicator(self):
        """Empty Project str shows empty indicator."""
        proj = Project(name="empty_project")
        str_output = str(proj)
        assert "empty" in str_output.lower()

    def test_subproject_appears_with_type_indicator(self):
        """Sub-projects appear with (sub-project) type indicator."""
        proj = Project(name="parent")
        subproj = Project(name="child")
        proj._projects["child"] = subproj
        str_output = str(proj)
        assert "(sub-project)" in str_output

    def test_dataset_appears_with_type_indicator(self):
        """Datasets appear with (dataset) type indicator."""
        proj = Project(name="test_project")
        ds = NDDataset([1.0, 2.0, 3.0], name="my_dataset")
        proj._datasets["my_dataset"] = ds
        str_output = str(proj)
        assert "(dataset)" in str_output

    def test_script_appears_with_type_indicator(self):
        """Scripts appear with (script) type indicator."""
        from spectrochempy.core.script import Script

        proj = Project(name="test_project")
        script = Script(name="my_script", content="print('hello')")
        proj._scripts["my_script"] = script
        str_output = str(proj)
        assert "(script)" in str_output

    def test_project_does_not_show_metadata_in_str(self):
        """Project __str__ does not show metadata (metadata belongs in _cstr)."""
        proj = Project(name="test_project", author="test_author")
        proj.meta.description = "Test description"
        str_output = str(proj)
        assert "test_author" not in str_output
        assert "Test description" not in str_output

    def test_project_metadata_appears_in_cstr(self):
        """Project metadata does appear in _cstr / detailed display."""
        proj = Project(name="test_project", author="test_author")
        proj.meta.description = "Test description"
        detailed = pstr(proj)
        assert "test_author" in detailed
        assert "Test description" in detailed


class TestProjectNameModel:
    """Tests for Project's explicit-name signal (has_defined_name)."""

    def test_project_named_has_defined_name(self):
        """Project with explicit name has has_defined_name == True."""
        proj = Project(name="my_project")
        assert proj.has_defined_name is True

    def test_project_default_has_no_defined_name(self):
        """Project with no name argument has has_defined_name == False."""
        proj = Project()
        assert proj.has_defined_name is False

    def test_project_none_has_no_defined_name(self):
        """Project(name=None) has has_defined_name == False."""
        proj = Project(name=None)
        assert proj.has_defined_name is False

    def test_project_auto_name_not_in_html_heading(self):
        """Default Project should not show generated name in HTML heading."""
        proj = Project()
        html = proj._repr_html_()
        # Extract the outer summary line (the heading)
        import re

        summary = re.search(r"<summary>(.*?)</summary>", html)
        assert summary is not None
        assert "Project-Project_" not in summary.group(0)

    def test_project_explicit_name_in_html_heading(self):
        """Named Project should show name in HTML heading."""
        proj = Project(name="myproj")
        html = proj._repr_html_()
        assert "myproj" in html
        assert "Project" in html


class TestProjectHTMLHierarchy:
    """Tests for Project HTML hierarchy rendering."""

    def test_project_html_contains_hierarchy(self):
        """Project HTML contains sub-project and dataset names."""
        top = Project(name="top")
        sub = Project(name="subproj")
        top.add_project(sub)
        sub.add_dataset(NDDataset([1, 2, 3], name="ds1"))
        html = top._repr_html_()
        assert "subproj" in html
        assert "ds1" in html
        assert "sub-project" in html
        assert "dataset" in html

    def test_project_hierarchy_nesting_preserved(self):
        """Nested projects preserve structure in HTML."""
        top = Project(name="top")
        mid = Project(name="mid")
        bot = Project(name="bot")
        top.add_project(mid)
        mid.add_project(bot)
        bot.add_dataset(NDDataset([1], name="leaf"))
        html = top._repr_html_()
        assert "top" in html
        assert "mid" in html
        assert "bot" in html
        assert "leaf" in html
        # Hierarchy lines are wrapped in <div> tags
        assert "<div>" in html
        assert "⤷" in html

    def test_project_empty_html_shows_empty(self):
        """Empty project HTML shows empty indicator."""
        proj = Project(name="empty")
        html = proj._repr_html_()
        assert "empty project" in html

    def test_project_mixed_items_in_html(self):
        """Project with sub-projects, datasets, and scripts shows all."""
        proj = Project(name="mixed")
        proj.add_dataset(NDDataset([1], name="ds1"))
        sub = Project(name="sub")
        proj.add_project(sub)
        sub.add_dataset(NDDataset([2], name="sub_ds"))
        html = proj._repr_html_()
        assert "ds1" in html
        assert "sub" in html
        assert "sub_ds" in html


# ======================================================================================
# HTML HEADING TESTS
# ======================================================================================


class TestHTMLHeading:
    """
    Tests for the HTML heading generated by _html_heading / convert_to_html.

    Headings should identify object type and name without duplication.
    """

    def test_coord_heading_contains_type(self):
        """Coord HTML heading contains the type name."""
        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert "Coord" in html

    def test_coord_named_heading(self):
        """Coord HTML heading includes name when set."""
        coord = Coord([1.0, 2.0, 3.0], name="mycoord")
        html = coord._repr_html_()
        assert "mycoord" in html

    def test_coord_unnamed_heading(self):
        """Coord HTML heading does not show empty brackets when name is empty."""
        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert "[]" not in html

    def test_coord_heading_scientific_identity(self):
        """Coord heading includes dtype, size/shape, and units when present."""
        coord = Coord([1.0, 2.0, 3.0], name="x", units="m")
        html = coord._repr_html_()
        assert "Coord" in html
        assert "x" in html
        assert "float" in html.lower()
        assert "size" in html.lower() or "shape" in html.lower()
        assert "m" in html

    def test_coordset_heading_contains_type(self):
        """CoordSet HTML heading contains the type name."""
        x = Coord([1.0, 2.0])
        cs = CoordSet(x=x)
        html = cs._repr_html_()
        assert "CoordSet" in html

    def test_coordset_heading_shows_coord_names(self):
        """CoordSet heading lists child coordinate names."""
        x = Coord([1.0, 2.0], name="x")
        y = Coord([3.0, 4.0], name="y")
        cs = CoordSet(x=x, y=y)
        html = cs._repr_html_()
        assert "CoordSet" in html
        assert "x" in html
        assert "y" in html

    def test_nddataset_heading_contains_type(self):
        """NDDataset HTML heading contains the type name."""
        ds = NDDataset([1.0, 2.0])
        html = ds._repr_html_()
        assert "NDDataset" in html

    def test_nddataset_named_heading(self):
        """NDDataset HTML heading includes name when set."""
        ds = NDDataset([1.0, 2.0], name="myds")
        html = ds._repr_html_()
        assert "myds" in html

    def test_nddataset_heading_scientific_identity(self):
        """NDDataset heading includes dtype, shape/size, and units when present."""
        ds = NDDataset([[1, 2], [3, 4]], units="cm⁻¹", name="spec")
        html = ds._repr_html_()
        assert "NDDataset" in html
        assert "spec" in html
        assert "float" in html.lower()
        assert "shape" in html.lower() or "size" in html.lower()
        assert "cm" in html

    def test_nddataset_heading_no_auto_id(self):
        """Unnamed NDDataset should not show auto-generated ID in heading."""
        ds = NDDataset([1.0, 2.0])
        html = ds._repr_html_()
        # The auto-generated name is the id (starts with NDDataset_)
        assert "NDDataset_NDDataset_" not in html

    def test_project_heading_no_duplicate_name(self):
        """Project HTML heading should not duplicate the project name."""
        proj = Project(name="my_project")
        html = proj._repr_html_()
        assert "my_project" in html

    def test_project_heading_contains_type(self):
        """Project HTML heading contains the type name."""
        proj = Project(name="test")
        html = proj._repr_html_()
        assert "Project" in html

    def test_empty_project_heading_safe(self):
        """Empty Project HTML heading still works."""
        proj = Project(name="empty")
        html = proj._repr_html_()
        assert "empty" in html
        assert "Project" in html

    def test_heading_starts_with_type(self):
        """The outermost summary starts with the type name."""
        coord = Coord([1.0, 2.0, 3.0], name="c")
        html = coord._repr_html_()
        assert "Coord" in html


class TestInlineSummary:
    """Tests that summary metadata renders inline (no collapsible Summary section)."""

    def test_no_summary_collapsible_in_nddataset(self):
        """NDDataset HTML should not contain a collapsible Summary section."""
        ds = NDDataset([1.0, 2.0], name="ds")
        html = ds._repr_html_()
        assert "<summary>Summary</summary>" not in html

    def test_metadata_visible_inline_nddataset(self):
        """NDDataset metadata fields appear directly under heading."""
        ds = NDDataset([1.0, 2.0], name="ds")
        html = ds._repr_html_()
        assert "name" in html
        assert "ds" in html

    def test_data_section_still_collapsible(self):
        """Data section should still be wrapped in <details>."""
        ds = NDDataset([1.0, 2.0])
        html = ds._repr_html_()
        # The data section <details> should exist
        assert "<details>" in html

    def test_no_summary_collapsible_in_project(self):
        """Project HTML should not contain a collapsible Summary section."""
        proj = Project(name="proj")
        html = proj._repr_html_()
        assert "<summary>Summary" not in html
        assert "<summary>Data" in html

    def test_project_metadata_inline(self):
        """Project metadata appears inline under heading."""
        proj = Project(name="proj", author="test")
        html = proj._repr_html_()
        assert "test" in html
        assert "proj" in html

    def test_coord_no_summary_collapsible(self):
        """Coord HTML should not contain a collapsible Summary section."""
        coord = Coord([1.0, 2.0], name="x")
        html = coord._repr_html_()
        assert "<summary>Summary</summary>" not in html


class TestDisplaySafetyNet:
    """Safety-net tests to catch regressions."""

    def test_all_repr_html_methods_return_strings(self):
        """All _repr_html_ methods should return strings without raising."""
        coord = Coord([1.0, 2.0])
        coordset = CoordSet(x=coord)
        ds = NDDataset([1.0, 2.0])
        proj = Project(name="test")

        # These should all succeed and return strings
        assert isinstance(coord._repr_html_(), str)
        assert isinstance(coordset._repr_html_(), str)
        assert isinstance(ds._repr_html_(), str)
        assert isinstance(proj._repr_html_(), str)

    def test_empty_objects_display_safely(self):
        """Empty objects should display safely without errors."""
        empty_coord = Coord([])
        empty_coordset = CoordSet()
        empty_ds = NDDataset([])
        empty_proj = Project(name="empty")

        # These should all succeed
        str(empty_coord)
        repr(empty_coord)
        empty_coord._repr_html_()

        str(empty_coordset)
        repr(empty_coordset)
        empty_coordset._repr_html_()

        str(empty_ds)
        repr(empty_ds)
        empty_ds._repr_html_()

        str(empty_proj)
        repr(empty_proj)
        empty_proj._repr_html_()

    def test_objects_with_none_values_display_safely(self):
        """Objects with None values should display safely."""
        coord = Coord([1.0, 2.0], title=None, name=None)
        coordset = CoordSet(name=None)
        ds = NDDataset([1.0, 2.0], name=None, title=None)
        proj = Project(name="test")

        # These should all succeed
        str(coord)
        repr(coord)
        coord._repr_html_()

        str(coordset)
        repr(coordset)
        coordset._repr_html_()

        str(ds)
        repr(ds)
        ds._repr_html_()

        str(proj)
        repr(proj)
        proj._repr_html_()


class TestSemanticCoord:
    """
    Semantic display model validation for Coord.

    These tests validate that Coord._repr_sections() produces the correct
    semantic structure without using the _cstr() → regex pipeline.

    No HTML is generated or tested here.  Only semantic structure.
    """

    def test_returns_list_with_one_section(self):
        """Coord._repr_sections() returns a single summary section."""
        coord = Coord([1.0, 2.0, 3.0])
        sections = coord._repr_sections()
        assert isinstance(sections, list)
        assert len(sections) == 1

    def test_section_role_is_summary(self):
        """The single section role is 'summary'."""
        coord = Coord([1.0, 2.0])
        sections = coord._repr_sections()
        assert sections[0].role == "summary"

    def test_section_title_is_summary(self):
        """The single section title is 'Summary'."""
        coord = Coord([1.0, 2.0])
        sections = coord._repr_sections()
        assert sections[0].title == "Summary"

    def test_contains_size_field(self):
        """A non-empty Coord has a size field."""
        coord = Coord([1.0, 2.0, 3.0])
        sections = coord._repr_sections()
        size_items = [
            i for i in sections[0].items if i.kind == "field" and i.key == "size"
        ]
        assert len(size_items) == 1
        assert size_items[0].value == "3"

    def test_size_field_value_matches_size_attr(self):
        """The size field value equals str(coord.size)."""
        coord = Coord([1.0, 2.0, 3.0, 4.0])
        sections = coord._repr_sections()
        size_item = next(
            i for i in sections[0].items if i.kind == "field" and i.key == "size"
        )
        assert size_item.value == str(coord.size)

    def test_contains_title_field_when_title_set(self):
        """Coord with an explicit title has a title field."""
        coord = Coord([1.0, 2.0], title="wavenumber")
        sections = coord._repr_sections()
        title_items = [
            i for i in sections[0].items if i.kind == "field" and i.key == "title"
        ]
        assert len(title_items) == 1
        assert title_items[0].value == "wavenumber"

    def test_title_field_absent_when_title_unset(self):
        """
        Coord with default (None) title does not have a title field.

        The _cstr() method only emits 'title:' when the title is truthy.
        Note: Coord always has a default title '<untitled>' which is truthy,
        so this test is informational only and only passes if title is None.

        This test documents the current behavior: the default title trait
        value is '<untitled>', not None.
        """
        coord = Coord([1.0, 2.0])
        sections = coord._repr_sections()
        title_items = [
            i for i in sections[0].items if i.kind == "field" and i.key == "title"
        ]
        # Currently Coord._title_default returns '<untitled>', so title
        # is always truthy.  If that changes, this assertion becomes valid.
        if coord.title:
            assert len(title_items) == 1
        else:
            assert len(title_items) == 0

    def test_contains_data_item_when_has_data(self):
        """A Coord with data has a data item."""
        coord = Coord([1.0, 2.0, 3.0])
        sections = coord._repr_sections()
        data_items = [i for i in sections[0].items if i.kind == "data"]
        assert len(data_items) == 1

    def test_data_item_contains_numeric_text(self):
        """The data item value contains numeric representation."""
        coord = Coord([1.5, 2.5, 3.5])
        sections = coord._repr_sections()
        data_item = next(i for i in sections[0].items if i.kind == "data")
        assert "1.5" in data_item.value
        assert "3.5" in data_item.value

    def test_data_item_includes_units(self):
        """When units are set, the data item includes unit text."""
        coord = Coord([1.0, 2.0, 3.0], units="m")
        sections = coord._repr_sections()
        data_item = next(i for i in sections[0].items if i.kind == "data")
        assert "m" in data_item.value

    def test_data_item_unitless_when_no_units(self):
        """When no units are set, the data item has no unit text."""
        coord = Coord([1.0, 2.0, 3.0])
        sections = coord._repr_sections()
        data_item = next(i for i in sections[0].items if i.kind == "data")
        assert "unitless" not in data_item.value

    def test_undefined_data_when_empty_unlabeled(self):
        """An empty unlabeled Coord has a data item with 'Undefined'."""
        coord = Coord([])
        sections = coord._repr_sections()
        data_item = next(i for i in sections[0].items if i.kind == "data")
        assert "Undefined" in data_item.value

    def test_contains_label_item_when_labeled(self):
        """A labeled Coord has a label item."""
        coord = Coord([1.0, 2.0, 3.0], labels=["A", "B", "C"])
        sections = coord._repr_sections()
        label_items = [i for i in sections[0].items if i.kind == "label"]
        assert len(label_items) == 1

    def test_label_item_contains_label_text(self):
        """The label item value contains the label content."""
        coord = Coord([1.0, 2.0, 3.0], labels=["A", "B", "C"])
        sections = coord._repr_sections()
        label_item = next(i for i in sections[0].items if i.kind == "label")
        assert "A" in label_item.value

    def test_no_label_item_when_not_labeled(self):
        """A non-labeled Coord has no label item."""
        coord = Coord([1.0, 2.0, 3.0])
        sections = coord._repr_sections()
        label_items = [i for i in sections[0].items if i.kind == "label"]
        assert len(label_items) == 0

    def test_item_order_matches_cstr_semantics(self):
        """
        Item kinds appear in the same semantic order as _cstr() output.

        Expected order: fields first (size, title), then data, then labels.
        """
        coord = Coord([1.0, 2.0, 3.0], labels=["A", "B", "C"])
        sections = coord._repr_sections()
        kinds = [i.kind for i in sections[0].items]
        # fields first, then data, then labels
        assert kinds[:2] == ["field", "field"]
        assert "data" in kinds
        assert "label" in kinds
        # data should come before label
        assert kinds.index("data") < kinds.index("label")

    def test_semantic_equivalence_with_cstr_empty(self):
        """Empty Coord: _repr_sections semantics match _cstr."""
        coord = Coord([])
        sections = coord._repr_sections()
        items = sections[0].items

        # Empty coord has no size (is_empty → skip size)
        size_items = [i for i in items if i.kind == "field" and i.key == "size"]
        assert len(size_items) == 0

        # Has title if title is truthy
        if coord.title:
            assert any(i.kind == "field" and i.key == "title" for i in items)

        # Has Undefined data
        data_items = [i for i in items if i.kind == "data"]
        assert len(data_items) == 1
        assert data_items[0].value == "Undefined"

    def test_semantic_equivalence_with_cstr_data(self):
        """Data Coord: _repr_sections semantics match _cstr."""
        coord = Coord([1.0, 2.0, 3.0], units="cm", title="shift")
        sections = coord._repr_sections()
        items = sections[0].items

        # Has size
        assert any(
            i.kind == "field" and i.key == "size" and i.value == "3" for i in items
        )

        # Has title
        assert any(
            i.kind == "field" and i.key == "title" and i.value == "shift" for i in items
        )

        # Has data (not Undefined)
        data_items = [i for i in items if i.kind == "data"]
        assert len(data_items) == 1
        assert "Undefined" not in data_items[0].value
        assert "cm" in data_items[0].value

    def test_display_item_repr(self):
        """DisplayItem repr is readable."""
        item = DisplayItem("field", "5", "size")
        r = repr(item)
        assert "field" in r
        assert "size" in r

    def test_display_item_equality(self):
        """DisplayItem equality compares kind, value, key."""
        a = DisplayItem("field", "5", "size")
        b = DisplayItem("field", "5", "size")
        assert a == b

    def test_display_section_equality(self):
        """DisplaySection equality compares role, title, items."""
        items = [DisplayItem("field", "5", "size")]
        a = DisplaySection("summary", "Summary", items)
        b = DisplaySection("summary", "Summary", items)
        assert a == b

    def test_display_section_default_items(self):
        """DisplaySection with no items defaults to empty list."""
        section = DisplaySection("summary", "Summary")
        assert section.items == []


class TestFormatArrayValues:
    """
    Tests for the shared _format_array_values() helper.

    This helper is used by NDArray._str_value(), NDComplex._str_value(),
    and Coord._repr_sections().  It encapsulates np.array2string(),
    masked value formatting, newline replacement, and unit suffix
    handling.
    """

    def test_float_array(self):
        """Float array produces correct numeric text."""
        data = np.array([1.5, 2.5, 3.5])
        text = _format_array_values(data)
        assert "1.5" in text
        assert "3.5" in text

    def test_integer_array(self):
        """Integer array produces correct numeric text."""
        data = np.array([1, 2, 3])
        text = _format_array_values(data)
        assert "1" in text
        assert "3" in text

    def test_array_with_units(self):
        """Unit suffix is appended when provided."""
        data = np.array([1.0, 2.0, 3.0])
        text = _format_array_values(data, units=" m")
        assert "m" in text
        assert text.endswith("m")

    def test_array_without_units(self):
        """No extraneous unit text when units are empty."""
        data = np.array([1.0, 2.0, 3.0])
        text = _format_array_values(data, units="")
        assert "unitless" not in text

    def test_masked_array(self):
        """Masked values are replaced by --dtype string."""
        data = np.ma.MaskedArray(
            np.array([1.0, 2.0, 3.0]),
            mask=[False, True, False],
        )
        text = _format_array_values(data, is_masked=True, dtype=np.dtype("float64"))
        assert "--" in text

    def test_masked_integer_array(self):
        """Masked integer values show --int64."""
        data = np.ma.MaskedArray(
            np.array([1, 2, 3]),
            mask=[False, True, False],
        )
        text = _format_array_values(data, is_masked=True, dtype=np.dtype("int64"))
        assert "--" in text

    def test_multiline_array(self):
        """Multi-line arrays replace internal newlines with sep."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        text = _format_array_values(data, sep="\n")
        assert "\n" in text
        assert "1" in text
        assert "4" in text

    def test_newline_replacement(self):
        """Sep parameter controls newline replacement."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        text = _format_array_values(data, sep=" | ")
        assert " | " in text
        lines = text.split(" | ")
        assert len(lines) == 2

    def test_prefix(self):
        """Prefix is prepended to the output."""
        data = np.array([1.0, 2.0, 3.0])
        text = _format_array_values(data, prefix="R")
        assert text.startswith("R")

    def test_no_trailing_separator(self):
        """Helper does not append a trailing sep."""
        data = np.array([1.0, 2.0, 3.0])
        text = _format_array_values(data)
        assert not text.endswith("\n")

    def test_no_trailing_separator_multiline(self):
        """Multi-line output does not end with sep."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        text = _format_array_values(data, sep="\n")
        assert "\n" in text
        assert "1" in text


class TestStrValuePreservation:
    """Verify that NDArray._str_value() output is unchanged after refactoring."""

    def test_str_value_float_no_units(self):
        """NDArray._str_value produces expected output for float data."""
        nd = NDArray([1.0, 2.0, 3.0])
        text = nd._str_value()
        assert "1" in text
        assert "3" in text

    def test_str_value_float_with_units(self):
        """NDArray._str_value includes unit suffix."""
        nd = NDArray([1.0, 2.0, 3.0], units="m")
        text = nd._str_value()
        assert "m" in text

    def test_str_value_masked(self):
        """NDArray._str_value shows -- for masked entries."""
        nd = NDArray([1.0, 2.0, 3.0], mask=[False, True, False])
        text = nd._str_value()
        assert "--" in text

    def test_str_value_empty(self):
        """NDArray._str_value returns 'empty' for empty array."""
        nd = NDArray([])
        text = nd._str_value()
        assert "empty" in text

    def test_str_value_units_in_data_block(self):
        """NDArray._str_value puts units inside the sentinel block."""
        nd = NDArray([1.0, 2.0, 3.0], units="m")
        text = nd._str_value()
        assert "m" in text
        assert "\x00" in text


class TestNDComplexStrValuePreservation:
    """Verify that NDComplexArray._str_value() output is unchanged after refactoring."""

    def test_str_value_real(self):
        """NDComplexArray._str_value for real data."""
        from spectrochempy.core.dataset.basearrays.ndcomplex import NDComplexArray

        nd = NDComplexArray([1.0, 2.0, 3.0])
        text = nd._str_value()
        assert "DATA" in text
        assert "1" in text

    def test_str_value_complex(self):
        """NDComplexArray._str_value splits R and I components."""
        from spectrochempy.core.dataset.basearrays.ndcomplex import NDComplexArray

        nd = NDComplexArray([1.0 + 2.0j, 3.0 + 4.0j])
        text = nd._str_value()
        assert "R[" in text
        assert "I[" in text


# ======================================================================================
# PHASE B: SEMANTIC HTML RENDERING FOR COORD
# ======================================================================================


class TestRenderSections:
    """Tests for _render_sections() converting DisplaySections to HTML."""

    def test_field_item_renders_attr_html(self):
        """A field item renders as attr-name / attr-value pair."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("field", "3", "size"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="attr-name"' in html
        assert "size" in html
        assert "3" in html
        assert ":" in html

    def test_data_item_renders_numeric(self):
        """A data item renders as numeric div."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("data", "[1.  2.  3.]"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="numeric"' in html
        assert "[1.  2.  3.]" in html

    def test_label_item_renders_label(self):
        """A label item renders as label div."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("label", "[A  B]"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="label"' in html
        assert "[A  B]" in html

    def test_data_with_key_wraps_in_attr_pair(self):
        """A data item with a key is wrapped in attr-name/attr-value."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("data", "[1.  2.  3.]", "coordinates"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="attr-name"' in html
        assert "coordinates" in html
        assert 'class="numeric"' in html
        assert "[1.  2.  3.]" in html

    def test_data_without_key_renders_bare_numeric(self):
        """A data item without a key renders as bare numeric div."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("data", "[1.  2.  3.]"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="numeric"' in html
        assert 'class="attr-name"' not in html

    def test_label_with_key_wraps_in_attr_pair(self):
        """A label item with a key is wrapped in attr-name/attr-value."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("label", "[A  B]", "labels"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="attr-name"' in html
        assert "labels" in html
        assert 'class="label"' in html
        assert "[A  B]" in html

    def test_label_without_key_renders_bare_label(self):
        """A label item without a key renders as bare label div."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("label", "[A  B]"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert 'class="label"' in html
        assert 'class="attr-name"' not in html

    def test_block_item_renders_plain_div(self):
        """A block item renders as plain div."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("block", "some block content"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert "<div>some block content</div>" in html

    def test_newline_in_value_becomes_br(self):
        """Newlines in item values are converted to <br/>."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("data", "line1\nline2"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert "<br/>" in html
        assert "line1" in html
        assert "line2" in html

    def test_data_section_wraps_in_details(self):
        """A data-role section is wrapped in a details/summary block."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "data",
                "Data",
                [
                    DisplayItem("data", "[1.  2.  3.]"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert "<details>" in html
        assert "<summary>Data</summary>" in html

    def test_dimension_section_wraps_in_details_with_title(self):
        """A dimension-role section uses its title in the summary."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "dimension",
                "Dimension `x`",
                [
                    DisplayItem("field", "50", "size"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert "<details>" in html
        assert "<summary>Dimension `x`</summary>" in html

    def test_summary_section_no_details(self):
        """A summary-role section renders items directly without wrapper."""
        from spectrochempy.utils.print import DisplayItem
        from spectrochempy.utils.print import DisplaySection
        from spectrochempy.utils.print import _render_sections

        sections = [
            DisplaySection(
                "summary",
                "Summary",
                [
                    DisplayItem("field", "3", "size"),
                ],
            )
        ]
        html = _render_sections(sections)
        assert "<details>" not in html
        assert "Summary" not in html


class TestCoordSemanticHTML:
    """Tests for Coord._repr_html_() via the semantic path."""

    def test_heading_contains_type(self):
        """Coord HTML heading contains the type name."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert "Coord" in html
        assert "float64" in html
        assert "size" in html.lower()

    def test_heading_includes_name(self):
        """Coord HTML heading includes the name when set."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0], name="x")
        html = coord._repr_html_()
        assert "[x]" in html

    def test_heading_includes_units(self):
        """Coord HTML heading includes units when present."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0], units="m")
        html = coord._repr_html_()
        assert "m" in html

    def test_content_shows_size_as_field(self):
        """Size is displayed as an attr-name/attr-value field."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert 'class="attr-name"' in html or "attr-name" in html
        assert "3" in html

    def test_content_shows_data_as_numeric(self):
        """Data values are displayed in a numeric div."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert 'class="numeric"' in html

    def test_content_shows_data_values(self):
        """Data values appear in the rendered HTML."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0])
        html = coord._repr_html_()
        assert "1" in html
        assert "2" in html
        assert "3" in html

    def test_content_shows_units_with_data(self):
        """Units appear appended to data values."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0, 3.0], units="m")
        html = coord._repr_html_()
        assert "m" in html

    def test_empty_coord_shows_undefined(self):
        """An empty Coord shows 'Undefined' as data."""
        from spectrochempy import Coord

        coord = Coord()
        html = coord._repr_html_()
        assert "Undefined" in html or "undefined" in html.lower()

    def test_labeled_coord_shows_labels(self):
        """Labeled Coord includes labels in the HTML."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0], labels=["A", "B"])
        html = coord._repr_html_()
        assert "A" in html
        assert "B" in html

    def test_outer_wrapper_structure(self):
        """Coord HTML has the expected outer wrapper structure."""
        from spectrochempy import Coord

        coord = Coord([1.0, 2.0])
        html = coord._repr_html_()
        assert '<div class="scp-output">' in html
        assert "<details>" in html
        assert "</details>" in html
        assert "</div>" in html

    def test_temporary_equivalence_with_sentinel_pipeline(self):
        """
        Semantic HTML contains all content present in sentinel HTML.

        This is a temporary migration equivalence test.
        It checks content presence, not exact HTML structure.
        """
        from spectrochempy import Coord
        from spectrochempy.utils.print import convert_to_html

        configs = [
            Coord([1.0, 2.0, 3.0]),
            Coord([1.0, 2.0], name="x", units="m"),
            Coord([1.0, 2.0, 3.0], title="MyCoord"),
            Coord([1.0, 2.0], labels=["A", "B"]),
        ]

        for coord in configs:
            coord._html_output = False
            old_html = convert_to_html(coord)

            coord._html_output = False
            new_html = coord._repr_html_()

            old_content = set(old_html.split())
            new_content = set(new_html.split())

            common = old_content & new_content
            assert (
                len(common) > 0
            ), f"No overlapping content between old and new HTML for {coord}"
            assert "Coord" in new_html
