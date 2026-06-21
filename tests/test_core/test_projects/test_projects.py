# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
import base64
import json
import pickle
import zipfile

import numpy as np
import pytest

from spectrochempy.application.preferences import preferences
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.utils.constants import INPLACE
from spectrochempy.utils.exceptions import SpectroChemPyError

prefs = preferences


def _rewrite_project_dataset_payload_as_legacy_pickle(filename):
    current = Project.load(filename)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js.pop("__format__", None)
    js.pop("__version__", None)
    js["datasets"][0]["data"] = {
        "__class__": "NUMPY_ARRAY",
        "base64": base64.b64encode(
            pickle.dumps(np.array(current.datasets[0].data)),
        ).decode(),
    }

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))


def _assert_dataset_membership(project, dataset, key, *, present):
    if present:
        assert key in project.datasets_names
        assert project[key] is dataset
        assert dataset in project.datasets
    else:
        assert key not in project.datasets_names
        assert dataset not in project.datasets


def _assert_project_membership(project, child, key, *, present):
    if present:
        assert key in project.projects_names
        assert project[key] is child
        assert child in project.projects
    else:
        assert key not in project.projects_names
        assert child not in project.projects


# =============================================================================
# Integration tests
# =============================================================================


def test_project(ds1, ds2, dsm):
    myp = Project(name="AGIR processing", method="stack")

    ds1.name = "toto"
    ds2.name = "tata"
    dsm.name = "titi"

    ds = ds1[:, 10, INPLACE]
    assert ds1.shape == ds.shape
    assert ds is ds1

    myp.add_datasets(ds1, ds2, dsm)

    print(myp.datasets_names)
    assert myp.datasets_names[-1] == "titi"
    assert ds1.parent == myp

    # iteration
    d = []
    for item in myp:
        d.append(item)

    assert d[1][0] == "tata"

    # add sub project
    msp1 = Project(name="AGIR ATG")
    msp1.add_dataset(ds1)
    assert ds1.parent == msp1
    assert ds1.name not in myp.datasets_names

    msp2 = Project(name="AGIR IR")

    myp.add_projects(msp1, msp2)

    print(myp)
    assert "tata" in myp.allnames
    myp["titi"]
    assert myp["titi"] == dsm

    # import multiple objects in Project
    myp2 = Project(msp1, msp2, ds1, ds2)

    print(myp2)


def test_empty_project():
    proj = Project(name="XXX")
    assert proj.name == "XXX"
    assert str(proj).strip() == "Project XXX:\n    (empty project)"


def test_save_and_load_project(ds1, ds2):
    myp = Project(name="process")

    ds1.name = "toto"
    ds2.name = "tata"

    myp.add_datasets(ds1, ds2)

    fn = myp.save()
    proj = Project.load(fn)

    assert str(proj["toto"]) == "NDDataset: [float64] a.u. (shape: (z:10, y:100, x:3))"


# =============================================================================
# Test classes
# =============================================================================


class TestProjectInit:
    """Tests for Project initialization."""

    def test_init_with_argnames(self, ds1, ds2):
        ds1.name = "first"
        ds2.name = "second"
        proj = Project(ds1, ds2, argnames=["renamed_first", "renamed_second"])
        assert "renamed_first" in proj.datasets_names
        assert "renamed_second" in proj.datasets_names
        assert "first" not in proj.datasets_names
        assert "second" not in proj.datasets_names

    def test_init_with_meta(self):
        proj = Project(name="meta_test", description="A test project", version="1.0")
        assert proj.meta.description == "A test project"
        assert proj.meta.version == "1.0"

    def test_init_with_named_objects(self):
        ds = NDDataset([1, 2, 3], name="mydataset")
        proj = Project(ds)
        assert "mydataset" in proj.datasets_names


class TestSetFromType:
    """Tests for _set_from_type method."""

    def test_set_from_type_rejects_unsupported_type(self):
        proj = Project(name="test")

        with pytest.raises(TypeError, match="does not accept"):
            proj._set_from_type(42)

        with pytest.raises(TypeError, match="does not accept"):
            proj._set_from_type("not a supported object")


class TestGetItem:
    """Tests for __getitem__ method."""

    def test_getitem_non_string_key(self):
        proj = Project(name="test")
        with pytest.raises(KeyError, match="must be a string"):
            proj[123]

    def test_getitem_no_project(self):
        proj = Project(name="test")
        assert proj["No project"] is None

    def test_getitem_invalid_key(self):
        proj = Project(name="test")
        with pytest.raises(KeyError, match="does not exist"):
            proj["nonexistent"]

    def test_getitem_with_slash_but_not_in_subproject(self):
        proj = Project(name="main")
        proj.add_project(Project(name="sub"))
        with pytest.raises(KeyError):
            proj["sub/nonexistent"]

    def test_getitem_composed_key_subproject_dataset(self):
        proj = Project(name="main")
        sub = Project(name="sub")
        ds = NDDataset([1, 2, 3], name="data")
        sub.add_dataset(ds)
        proj.add_project(sub)

        result = proj["sub/data"]
        assert result is ds


class TestSetItem:
    """Tests for __setitem__ method."""

    def test_setitem_non_string_key(self):
        proj = Project(name="test")
        with pytest.raises(KeyError, match="must be a string"):
            proj[123] = NDDataset([1, 2])

    def test_setitem_type_mismatch_raises(self):
        proj = Project(name="test")
        ds = NDDataset([1, 2, 3], name="my_data")
        proj.add_dataset(ds)
        wrong_type = Project(name="my_data")
        with pytest.raises(ValueError, match="different type"):
            proj["my_data"] = wrong_type

    def test_setitem_new_assignment(self):
        proj = Project(name="test")

        # dataset
        ds = NDDataset([1, 2, 3], name="data")
        proj["data"] = ds
        assert "data" in proj.datasets_names
        assert ds.parent == proj

        # project
        sub = Project(name="sub")
        proj["sub"] = sub
        assert "sub" in proj.projects_names
        assert sub.parent == proj

    @pytest.mark.parametrize("item_type", ["dataset", "project"])
    def test_setitem_update_existing(self, item_type):
        """Test updating existing items via __setitem__."""
        proj = Project(name="test")

        # Create initial items
        if item_type == "dataset":
            proj.add_dataset(NDDataset([1, 2, 3], name="item"))
            old = proj["item"]
            initial = NDDataset([4, 5, 6], name="item")
        else:
            proj.add_project(Project(name="item"))
            old = proj["item"]
            initial = Project(name="item")

        # Update
        proj["item"] = initial

        assert proj["item"] is initial
        assert initial.parent == proj
        assert old.parent is None


class TestGetAttr:
    """Tests for __getattr__ method."""

    def test_getattr_meta_attribute(self):
        proj = Project(name="test", custom_attr="value")
        assert proj.custom_attr == "value"

    def test_getattr_invalid_attribute(self):
        proj = Project(name="test")
        with pytest.raises(AttributeError):
            _ = proj.nonexistent_attribute

    def test_getattr_access_items_via_attribute(self):
        proj = Project(name="test")
        proj.add_dataset(NDDataset([1, 2, 3], name="data"))
        proj.add_project(Project(name="sub"))

        assert proj.data.name == "data"
        assert proj.sub.name == "sub"


class TestRepr:
    """Tests for string representation."""

    def test_repr_html(self):
        proj = Project(name="test")
        html = proj._repr_html_()
        assert '<div class="scp-output">' in html
        assert "test" in html

    def test_str_empty_project(self):
        proj = Project(name="root")
        output = str(proj)
        assert "empty project" in output

    def test_str_with_nested_projects(self):
        proj = Project(name="root")
        proj.add_project(Project(name="sub1"))
        proj.add_project(Project(name="sub2"))
        output = str(proj)
        assert "sub1" in output
        assert "sub2" in output
        assert "(sub-project)" in output


class TestCopy:
    """Tests for copy functionality."""

    def test_copy_project(self, ds1):
        ds1.name = "data"
        proj = Project(ds1, name="original")
        copied = proj.__copy__()
        assert copied.name == "original"
        assert "data" in copied.datasets_names
        # modifying copy shouldn't affect original
        del copied._datasets["data"]
        assert "data" in proj.datasets_names

    def test_copy_method(self, ds1):
        ds1.name = "data"
        proj = Project(ds1, name="original")
        copied = proj.copy()
        assert copied.name == "original"
        assert "data" in copied.datasets_names


class TestDuplicateNamesRFC:
    """RFC-compliant duplicate name handling."""

    def test_add_dataset_duplicate_raises_valueerror(self):
        proj = Project(name="test")
        first = NDDataset([1], name="data")
        proj.add_dataset(first)

        second = NDDataset([2], name="data")
        with pytest.raises(ValueError, match="already exists"):
            proj.add_dataset(second)

    def test_add_project_duplicate_raises_valueerror(self):
        proj = Project(name="root")
        first = Project(name="child")
        proj.add_project(first)

        second = Project(name="child")
        with pytest.raises(ValueError, match="already exists"):
            proj.add_project(second)

    def test_duplicate_dataset_dataset(self):
        proj = Project(name="test")
        proj.add_dataset(NDDataset([1], name="data"))
        with pytest.raises(ValueError, match="already exists"):
            proj.add_dataset(NDDataset([2], name="data"))

    def test_duplicate_project_project(self):
        proj = Project(name="test")
        proj.add_project(Project(name="sub"))
        with pytest.raises(ValueError, match="already exists"):
            proj.add_project(Project(name="sub"))

    def test_duplicate_dataset_project_collision(self):
        proj = Project(name="test")
        proj.add_dataset(NDDataset([1], name="shared"))
        with pytest.raises(ValueError, match="already exists"):
            proj.add_project(Project(name="shared"))

    def test_duplicate_project_dataset_collision(self):
        proj = Project(name="test")
        proj.add_project(Project(name="shared"))
        with pytest.raises(ValueError, match="already exists"):
            proj.add_dataset(NDDataset([1], name="shared"))

    def test_constructor_duplicate_raises_valueerror(self):
        ds1 = NDDataset([1], name="data")
        ds2 = NDDataset([2], name="data")
        with pytest.raises(ValueError, match="already exists"):
            Project(ds1, ds2)

    def test_duplicate_does_not_modify_ownership_on_failure(self):
        proj = Project(name="test")
        ds = NDDataset([1], name="data")
        proj.add_dataset(ds)

        original_parent = ds.parent
        with pytest.raises(ValueError):
            proj.add_dataset(NDDataset([2], name="data"))

        assert ds.parent is original_parent
        assert ds.name == "data"
        _assert_dataset_membership(proj, ds, "data", present=True)

    def test_duplicate_does_not_partially_modify_parent_on_failure(self):
        proj = Project(name="test")
        first = NDDataset([1], name="data")
        proj.add_dataset(first)

        with pytest.raises(ValueError):
            proj.add_dataset(NDDataset([2], name="data"))

        assert len(proj.datasets) == 1
        assert first.parent is proj


class TestRemoveMethods:
    """Tests for remove methods."""

    def test_remove_dataset(self):
        proj = Project(name="test")
        ds = NDDataset([1, 2, 3], name="my_dataset")
        proj.add_dataset(ds)
        assert "my_dataset" in proj.datasets_names
        proj.remove_dataset("my_dataset")
        assert "my_dataset" not in proj.datasets_names
        assert ds.parent is None

    def test_remove_project(self):
        proj = Project(name="test")
        sub = Project(name="sub")
        proj.add_project(sub)
        assert "sub" in proj.projects_names
        proj.remove_project("sub")
        assert "sub" not in proj.projects_names
        assert sub.parent is None

    def test_clear_datasets(self, ds1, ds2):
        proj = Project(name="test")
        proj.add_datasets(ds1, ds2)
        assert len(proj.datasets_names) == 2
        assert ds1.parent == proj
        assert ds2.parent == proj
        proj.clear_datasets()
        assert len(proj.datasets_names) == 0
        assert ds1.parent is None
        assert ds2.parent is None

    def test_clear_projects(self):
        proj = Project(name="test")
        sub1 = Project(name="sub1")
        sub2 = Project(name="sub2")
        proj.add_projects(sub1, sub2)
        assert len(proj.projects_names) == 2
        assert sub1.parent == proj
        assert sub2.parent == proj
        proj.clear_projects()
        assert len(proj.projects_names) == 0
        assert sub1.parent is None
        assert sub2.parent is None

    def test_remove_all_dataset_deprecated(self, ds1, ds2):
        proj = Project(name="test")
        proj.add_datasets(ds1, ds2)
        assert len(proj.datasets_names) == 2
        with pytest.warns(DeprecationWarning, match="remove_all_dataset.*deprecated"):
            proj.remove_all_dataset()
        assert len(proj.datasets_names) == 0

    def test_remove_all_project_deprecated(self):
        proj = Project(name="test")
        proj.add_projects(Project(name="sub1"), Project(name="sub2"))
        assert len(proj.projects_names) == 2
        with pytest.warns(DeprecationWarning, match="remove_all_project.*deprecated"):
            proj.remove_all_project()
        assert len(proj.projects_names) == 0


class TestAddProject:
    """Tests for add_project method."""

    def test_add_project_rename(self):
        proj = Project(name="main")
        sub = Project(name="original_name")
        proj.add_project(sub, name="renamed")
        assert "renamed" in proj.projects_names
        assert "original_name" not in proj.projects_names
        assert sub.name == "renamed"

    def test_add_project_with_name_none(self):
        proj = Project(name="main")
        sub = Project(name="my_subproject")
        proj.add_project(sub, name=None)
        assert "my_subproject" in proj.projects_names


class TestParentAssignment:
    """Tests for parent property assignment."""

    def test_parent_setter_changes_parent(self):
        proj1 = Project(name="parent1")
        proj2 = Project(name="parent2")
        sub = Project(name="child")

        proj1.add_project(sub)
        assert sub.parent == proj1

        proj2.add_project(sub)
        assert sub.parent == proj2
        assert "child" in proj2.projects_names


class TestProjectOwnershipCharacterization:
    """Characterization tests for current ownership semantics."""

    def test_add_dataset_sets_bidirectional_membership(self):
        proj = Project(name="root")
        ds = NDDataset([1, 2, 3], name="data")

        proj.add_dataset(ds)

        _assert_dataset_membership(proj, ds, "data", present=True)
        assert ds.parent is proj

    def test_add_subproject_sets_bidirectional_membership(self):
        proj = Project(name="root")
        child = Project(name="child")

        proj.add_project(child)

        _assert_project_membership(proj, child, "child", present=True)
        assert child.parent is proj

    def test_move_dataset_between_projects_updates_membership_in_both_projects(self):
        source = Project(name="source")
        destination = Project(name="destination")
        ds = NDDataset([1, 2, 3], name="data")

        source.add_dataset(ds)
        destination.add_dataset(ds)

        _assert_dataset_membership(source, ds, "data", present=False)
        _assert_dataset_membership(destination, ds, "data", present=True)
        assert ds.parent is destination

    def test_move_subproject_between_projects_updates_membership_in_both_projects(self):
        source = Project(name="source")
        destination = Project(name="destination")
        child = Project(name="child")

        source.add_project(child)
        destination.add_project(child)

        _assert_project_membership(source, child, "child", present=False)
        _assert_project_membership(destination, child, "child", present=True)
        assert child.parent is destination

    def test_remove_dataset_breaks_bidirectional_membership(self):
        proj = Project(name="root")
        ds = NDDataset([1, 2, 3], name="data")

        proj.add_dataset(ds)
        proj.remove_dataset("data")

        _assert_dataset_membership(proj, ds, "data", present=False)
        assert ds.parent is None

    def test_remove_subproject_breaks_bidirectional_membership(self):
        proj = Project(name="root")
        child = Project(name="child")

        proj.add_project(child)
        proj.remove_project("child")

        _assert_project_membership(proj, child, "child", present=False)
        assert child.parent is None


class TestProjectReplacementCharacterization:
    """Characterization tests for current replacement semantics."""

    def test_replace_dataset_detaches_old_child_parent(self):
        proj = Project(name="root")
        old = NDDataset([1, 2, 3], name="item")
        new = NDDataset([4, 5, 6], name="replacement")

        proj.add_dataset(old)
        proj["item"] = new

        _assert_dataset_membership(proj, new, "item", present=True)
        assert proj["item"] is not old
        assert old not in proj.datasets
        assert new.parent is proj
        assert old.parent is None
        assert new.name == "item"
        assert proj["item"].name == "item"

    def test_replace_subproject_detaches_old_child_parent(self):
        proj = Project(name="root")
        old = Project(name="item")
        new = Project(name="replacement")

        proj.add_project(old)
        proj["item"] = new

        _assert_project_membership(proj, new, "item", present=True)
        assert proj["item"] is not old
        assert old not in proj.projects
        assert new.parent is proj
        assert old.parent is None
        assert new.name == "item"
        assert proj["item"].name == "item"

    def test_replace_dataset_old_child_detached_even_with_prior_parent(self):
        proj = Project(name="root")
        other = Project(name="other")
        old = NDDataset([1, 2, 3], name="item")
        new = NDDataset([4, 5, 6], name="replacement")

        other.add_dataset(old)
        proj.add_dataset(new)
        proj["item"] = old
        proj["item"] = new

        assert new.parent is proj
        assert old.parent is None

    def test_replace_subproject_old_child_detached_even_with_prior_parent(self):
        proj = Project(name="root")
        other = Project(name="other")
        old = Project(name="item")
        new = Project(name="replacement")

        other.add_project(old)
        proj.add_project(new)
        proj["item"] = old
        proj["item"] = new

        assert new.parent is proj
        assert old.parent is None

    def test_replace_dataset_does_not_affect_unrelated_children(self):
        proj = Project(name="root")
        old = NDDataset([1, 2, 3], name="item")
        unrelated = NDDataset([7, 8, 9], name="other")
        new = NDDataset([4, 5, 6], name="replacement")

        proj.add_dataset(old)
        proj.add_dataset(unrelated)
        proj["item"] = new

        assert old.parent is None
        assert unrelated.parent is proj

    def test_replace_subproject_does_not_affect_unrelated_children(self):
        proj = Project(name="root")
        old = Project(name="item")
        unrelated = Project(name="other")
        new = Project(name="replacement")

        proj.add_project(old)
        proj.add_project(unrelated)
        proj["item"] = new

        assert old.parent is None
        assert unrelated.parent is proj


class TestProjectNameMutationCharacterization:
    """Characterization tests for child name mutation after insertion."""

    def test_dataset_name_mutation_breaks_reparenting_by_current_name_lookup(self):
        source = Project(name="source")
        destination = Project(name="destination")
        ds = NDDataset([1, 2, 3], name="original")

        source.add_dataset(ds)
        ds.name = "mutated"

        with pytest.raises(KeyError, match="mutated"):
            destination.add_dataset(ds)

        _assert_dataset_membership(source, ds, "original", present=True)
        _assert_dataset_membership(destination, ds, "original", present=False)
        assert ds.parent is source
        assert ds.name == "mutated"


class TestProjectCycleRejection:
    """Cycle rejection tests per Project Invariants RFC."""

    def test_self_insertion_raises_valueerror(self):
        proj = Project(name="self")
        with pytest.raises(ValueError, match="cycle"):
            proj.add_project(proj)

    def test_direct_cycle_raises_valueerror(self):
        parent = Project(name="parent")
        child = Project(name="child")
        parent.add_project(child)
        with pytest.raises(ValueError, match="cycle"):
            child.add_project(parent)

    def test_indirect_cycle_raises_valueerror(self):
        a = Project(name="a")
        b = Project(name="b")
        c = Project(name="c")
        a.add_project(b)
        b.add_project(c)
        with pytest.raises(ValueError, match="cycle"):
            c.add_project(a)

    def test_parent_setter_self_cycle_raises_valueerror(self):
        proj = Project(name="self")
        with pytest.raises(ValueError, match="cycle"):
            proj.parent = proj

    def test_parent_setter_direct_cycle_raises_valueerror(self):
        parent = Project(name="parent")
        child = Project(name="child")
        parent.add_project(child)
        with pytest.raises(ValueError, match="cycle"):
            parent.parent = child

    def test_parent_setter_indirect_cycle_raises_valueerror(self):
        a = Project(name="a")
        b = Project(name="b")
        c = Project(name="c")
        a.add_project(b)
        b.add_project(c)
        with pytest.raises(ValueError, match="cycle"):
            a.parent = c

    def test_state_unchanged_after_self_insertion_failure(self):
        proj = Project(name="self")
        with pytest.raises(ValueError):
            proj.add_project(proj)
        assert proj.parent is None
        assert "self" not in proj.projects_names

    def test_state_unchanged_after_direct_cycle_failure(self):
        parent = Project(name="parent")
        child = Project(name="child")
        parent.add_project(child)
        with pytest.raises(ValueError):
            child.add_project(parent)
        assert child.parent is parent
        assert parent.parent is None
        _assert_project_membership(parent, child, "child", present=True)

    def test_state_unchanged_after_indirect_cycle_failure(self):
        a = Project(name="a")
        b = Project(name="b")
        c = Project(name="c")
        a.add_project(b)
        b.add_project(c)
        with pytest.raises(ValueError):
            c.add_project(a)
        assert c.parent is b
        assert b.parent is a
        assert a.parent is None

    def test_normal_tree_structure_not_affected_by_cycle_protection(self):
        parent = Project(name="parent")
        child = Project(name="child")
        parent.add_project(child)
        assert child.parent is parent
        _assert_project_membership(parent, child, "child", present=True)

    def test_normal_parent_setter_not_affected_by_cycle_protection(self):
        proj = Project(name="root")
        child = Project(name="child")
        proj.add_project(child)
        child.parent = None
        assert child.parent is None
        assert "child" not in proj.projects_names


class TestProjectKeyNameIdentityRFC:
    """RFC-compliant key/name identity tests."""

    def test_setitem_replacement_enforces_key_name_identity_for_dataset(self):
        proj = Project(name="test")
        ds = NDDataset([1, 2, 3], name="original")
        proj.add_dataset(NDDataset([0], name="existing"))
        proj["existing"] = ds
        assert proj["existing"].name == "existing"
        assert ds.name == "existing"

    def test_setitem_replacement_enforces_key_name_identity_for_project(self):
        proj = Project(name="test")
        sub = Project(name="original")
        proj.add_project(Project(name="existing"))
        proj["existing"] = sub
        assert proj["existing"].name == "existing"
        assert sub.name == "existing"

    def test_setitem_new_key_enforces_identity_via_add_dataset(self):
        proj = Project(name="test")
        ds = NDDataset([1, 2, 3], name="original")
        proj["renamed"] = ds
        assert proj["renamed"].name == "renamed"
        assert ds.name == "renamed"

    def test_setitem_new_key_enforces_identity_via_add_project(self):
        proj = Project(name="test")
        sub = Project(name="original")
        proj["renamed"] = sub
        assert proj["renamed"].name == "renamed"
        assert sub.name == "renamed"

    def test_move_between_projects_via_add_dataset_preserves_identity(self):
        source = Project(name="source")
        dest = Project(name="dest")
        ds = NDDataset([1, 2, 3], name="data")
        source.add_dataset(ds)
        dest.add_dataset(ds)
        assert dest["data"].name == "data"

    def test_move_between_projects_via_add_project_preserves_identity(self):
        source = Project(name="source")
        dest = Project(name="dest")
        sub = Project(name="child")
        source.add_project(sub)
        dest.add_project(sub)
        assert dest["child"].name == "child"

    def test_constructor_with_argnames_enforces_identity(self):
        ds = NDDataset([1, 2, 3], name="original")
        proj = Project(ds, argnames=["renamed"])
        assert proj["renamed"].name == "renamed"
        assert ds.name == "renamed"

    def test_parent_setter_moves_child_and_add_project_restores_identity(self):
        source = Project(name="source")
        dest = Project(name="dest")
        sub = Project(name="child")
        source.add_project(sub)
        sub.parent = None
        dest.add_project(sub)
        assert dest["child"].name == "child"

    def test_ownership_and_duplicate_and_cycle_still_work(self):
        """Verify prior invariants are not broken by key/name identity changes."""
        proj = Project(name="root")
        ds = NDDataset([1], name="data")
        proj.add_dataset(ds)
        assert ds.parent is proj
        assert proj["data"] is ds

        with pytest.raises(ValueError, match="already exists"):
            proj.add_dataset(NDDataset([2], name="data"))

        with pytest.raises(ValueError, match="cycle"):
            proj.add_project(proj)


class TestProjectKeyNameIdentityCharacterization:
    """Characterization tests for current key/name identity rules."""

    def test_dataset_insertion_without_rename_keeps_key_equal_to_name(self):
        proj = Project(name="root")
        ds = NDDataset([1, 2, 3], name="data")

        proj.add_dataset(ds)

        assert proj["data"].name == "data"

    def test_dataset_insertion_with_explicit_name_mutates_child_name(self):
        proj = Project(name="root")
        ds = NDDataset([1, 2, 3], name="original")

        proj.add_dataset(ds, name="renamed")

        assert "renamed" in proj.datasets_names
        assert proj["renamed"].name == "renamed"
        assert ds.name == "renamed"

    def test_save_load_restores_child_key_name_identity(self):
        proj = Project(name="root")
        ds = NDDataset([1, 2, 3], name="data")
        sub = Project(name="sub")
        nested = NDDataset([4, 5, 6], name="nested")
        sub.add_dataset(nested)
        proj.add_dataset(ds)
        proj.add_project(sub)

        filename = proj.save_as("project-key-name-roundtrip")
        loaded = Project.load(filename)

        assert loaded["data"].name == "data"
        assert loaded["sub"].name == "sub"
        assert loaded["sub/nested"].name == "nested"


class TestProjectCopyCharacterization:
    """Characterization tests for current shallow-copy semantics."""

    def test_copy_duplicates_dataset_child_and_resets_parent_pointer(self):
        proj = Project(name="root")
        ds = NDDataset([1, 2, 3], name="data")
        proj.add_dataset(ds)

        copied = proj.copy()

        assert copied["data"] is not ds
        assert copied["data"].name == ds.name
        assert copied["data"].parent is None

    def test_copy_shares_subprojects_and_keeps_parent_on_original(self):
        proj = Project(name="root")
        child = Project(name="child")
        proj.add_project(child)

        copied = proj.copy()

        assert copied["child"] is child
        assert copied["child"].parent is proj
        assert copied["child"].parent is not copied


class TestImplements:
    """Tests for _implements static method."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            (None, "Project"),
            ("Project", True),
            ("NDDataset", False),
        ],
    )
    def test_implements(self, name, expected):
        assert Project._implements(name) == expected


class TestAttributes:
    """Tests for _attributes_ method."""

    def test_attributes_returns_expected_list(self):
        proj = Project(name="test")
        attrs = proj._attributes_()
        expected = ["name", "meta", "parent", "datasets", "projects"]
        assert attrs == expected


class TestPrivateMethods:
    """Tests for private methods."""

    def test_get_from_type_is_noop(self):
        proj = Project(name="test")
        assert proj._get_from_type("any_name") is None


class TestProjectProperties:
    """Tests for project properties."""

    def test_allnames_and_allitems(self):
        proj = Project(name="test")
        proj.add_dataset(NDDataset([1, 2, 3], name="data"))
        proj.add_project(Project(name="sub"))

        all_names = proj.allnames
        assert "data" in all_names
        assert "sub" in all_names

        items = proj.allitems
        item_names = [item[0] for item in items]
        assert "data" in item_names
        assert "sub" in item_names

    def test_readonly_id_property(self):
        proj = Project(name="test")
        assert proj.id is not None
        assert proj.id.startswith("Project_")

    def test_directory_property(self):
        proj = Project(name="test")
        assert proj.directory is None


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_name_auto_generated(self):
        proj = Project()
        assert proj.name is not None

    def test_dataset_with_empty_name(self):
        proj = Project(name="test")
        ds = NDDataset([1, 2, 3])
        ds.name = ""
        proj.add_dataset(ds)
        assert len(proj.datasets_names) == 1

    def test_nested_subprojects(self):
        proj = Project(name="root")
        level1 = Project(name="level1")
        level2 = Project(name="level2")

        proj.add_project(level1)
        level1.add_project(level2)

        assert "level1" in proj.projects_names
        assert "level2" in level1.projects_names

    def test_dataset_parent_chain(self):
        proj = Project(name="root")
        sub1 = Project(name="sub1")
        sub2 = Project(name="sub2")
        ds = NDDataset([1, 2, 3], name="data")

        proj.add_project(sub1)
        sub1.add_project(sub2)
        sub2.add_dataset(ds)

        assert ds.parent == sub2
        assert sub2.parent == sub1
        assert sub1.parent == proj

    def test_iterate_empty_project(self):
        proj = Project(name="empty")
        assert list(proj) == []


class TestSerializationRoundtrip:
    """Tests for save/load roundtrip."""

    def test_save_load_with_nested_projects(self, ds1):
        proj = Project(name="nested_test")
        sub = Project(name="subproject")
        ds1.name = "data"
        sub.add_dataset(ds1)
        proj.add_project(sub)

        filename = proj.save()
        loaded = Project.load(filename)

        assert "subproject" in loaded.projects_names
        assert "data" in loaded.subproject.datasets_names

    def test_save_load_overwrites_root_name_from_filename(self):
        proj = Project(name="logical_name", description="saved metadata")
        proj.add_dataset(NDDataset([1, 2, 3], name="data"))

        filename = proj.save_as("filename_name_wins")
        loaded = Project.load(filename)

        assert loaded.name == "filename_name_wins"
        assert loaded.meta.description == "saved metadata"

    def test_save_as_creates_new_file(self, ds1):
        proj = Project(name="overwrite_test")
        proj.add_dataset(ds1)

        filename1 = proj.save()
        filename2 = proj.save_as("overwrite_test_v2")

        assert filename1.exists()
        assert filename2.exists()
        assert filename1 != filename2

    def test_save_load_ignores_legacy_fields(self, ds1):
        proj = Project(name="legacy_fields_test")
        ds1.name = "data"
        proj.add_dataset(ds1)

        filename = proj.save()

        with zipfile.ZipFile(filename, "r") as zipf:
            member = zipf.namelist()[0]
            js = json.loads(zipf.read(member).decode("utf-8"))

        js["datasets"][0]["roi"] = [0.0, 1.0]
        js["datasets"][0]["modeldata"] = [42.0]
        js["_others"] = {"unused": "should be ignored"}

        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(member, json.dumps(js, indent=2))

        loaded = Project.load(filename)

        assert not hasattr(loaded.data, "roi")
        assert not hasattr(loaded.data, "modeldata")
        assert not hasattr(loaded, "_others")

    def test_project_load_requires_explicit_opt_in(self, ds1):
        proj = Project(name="legacy_project")
        ds1.name = "data"
        proj.add_dataset(ds1)

        filename = proj.save()
        _rewrite_project_dataset_payload_as_legacy_pickle(filename)

        with pytest.raises(
            SpectroChemPyError,
            match="trusted legacy loading",
        ):
            Project.load(filename)

        loaded = Project.load(filename, allow_unsafe_legacy=True)
        assert "data" in loaded.datasets_names


class TestArgNamesEdgeCases:
    """Test argnames edge cases."""

    def test_argnames_wrong_length_raises(self, ds1, ds2):
        ds1.name = "first"
        ds2.name = "second"
        with pytest.raises(IndexError):
            Project(ds1, ds2, argnames=["only_one"])
