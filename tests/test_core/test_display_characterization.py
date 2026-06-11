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

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
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

    def test_project_repr_contains_object_identity(self):
        """Project repr currently shows object identity."""
        proj = Project(name="test_project")
        repr_str = repr(proj)
        assert "Project" in repr_str
        assert " object at " in repr_str

    def test_project_repr_different_from_str(self):
        """Project repr is currently different from str."""
        proj = Project(name="test_project")
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

    def test_project_does_not_show_metadata(self):
        """Project currently does not show metadata in str output."""
        proj = Project(name="test_project")
        proj.author = "test_author"
        proj.description = "Test description"
        str_output = str(proj)
        assert "test_author" not in str_output
        assert "Test description" not in str_output


# ======================================================================================
# ADDITIONAL SAFETY-NET TESTS
# ======================================================================================


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
