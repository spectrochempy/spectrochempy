# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import pytest

from spectrochempy.application.preferences import preferences
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.core.project.project import makescript
from spectrochempy.core.script import Script
from spectrochempy.core.script import run_script
from spectrochempy.utils.constants import INPLACE

prefs = preferences


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


def test_project_with_script():
    proj = Project(
        Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
        Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
        Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
        name="HIZECOKE_TEST",
    )

    assert proj.projects_names == ["P350", "A350", "B350"]

    # add a dataset to a subproject
    ir = NDDataset([1, 2, 3])
    tg = NDDataset([1, 3, 4])
    proj.A350["IR"] = ir
    proj["TG"] = tg

    print(proj.A350)
    print(proj)
    print(proj.A350.label)

    f = proj.save()

    newproj = Project.load("HIZECOKE_TEST")
    assert str(newproj) == str(proj)
    assert newproj.A350.label == proj.A350.label

    script_source = (
        "set_loglevel(INFO)\n"
        'info_("samples contained in the project are : '
        '%s"%proj.projects_names)'
    )

    proj["print_info"] = Script("print_info", script_source)

    # save but do not change the original data
    proj.save_as("HIZECOKE_TEST", overwrite_data=False)

    newproj = Project.load("HIZECOKE_TEST")

    # execute
    run_script(newproj.print_info, locals())
    newproj.print_info.execute(locals())
    newproj.print_info(locals())
    newproj.print_info()

    proj.save_as("HIZECOKE_TEST")
    newproj = Project.load("HIZECOKE_TEST")


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

    def test_set_from_type_with_other_object_having_name(self):
        proj = Project(name="test")

        class NamedObject:
            name = "custom_object"

        obj = NamedObject()
        proj._set_from_type(obj)
        assert "custom_object" in proj._others
        assert proj._others["custom_object"] is obj

    def test_set_from_type_invalid_object(self):
        proj = Project(name="test")

        class UnnamedObject:
            pass

        obj = UnnamedObject()
        with pytest.raises(ValueError, match="has no name"):
            proj._set_from_type(obj)


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

    def test_getitem_composed_key_subproject_script(self):
        proj = Project(name="main")
        sub = Project(name="sub")
        script = Script("myscript", "x = 1")
        sub.add_script(script)
        proj.add_project(sub)

        result = proj["sub/myscript"]
        assert result is script
        assert result.content == "x = 1"

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

        # script
        script = Script("script", "x = 1")
        proj["script"] = script
        assert "script" in proj.scripts_names
        assert script.parent == proj

        # project
        sub = Project(name="sub")
        proj["sub"] = sub
        assert "sub" in proj.projects_names
        assert sub.parent == proj

    @pytest.mark.parametrize("item_type", ["dataset", "script", "project"])
    def test_setitem_update_existing(self, item_type):
        """Test updating existing items via __setitem__."""
        proj = Project(name="test")

        # Create initial items
        if item_type == "dataset":
            proj.add_dataset(NDDataset([1, 2, 3], name="item"))
            initial = NDDataset([4, 5, 6], name="item")
        elif item_type == "script":
            proj.add_script(Script("item", "x = 1"))
            initial = Script("item", "y = 2")
        else:
            proj.add_project(Project(name="item"))
            initial = Project(name="item")

        # Update
        proj["item"] = initial

        assert proj["item"] is initial
        assert initial.parent == proj


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
        proj.add_script(Script("script", "x = 1"))

        assert proj.data.name == "data"
        assert proj.sub.name == "sub"
        assert proj.script.name == "script"


class TestRepr:
    """Tests for string representation."""

    def test_repr_html(self):
        proj = Project(name="test")
        html = proj._repr_html_()
        assert "<br/>" in html
        assert "&nbsp;" in html

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

    def test_str_with_scripts(self):
        proj = Project(name="scripts_only")
        proj.add_scripts(
            Script("first", "x = 1"),
            Script("second", "y = 2"),
        )
        output = str(proj)
        assert "first" in output
        assert "second" in output
        assert "(script)" in output


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


class TestDuplicateNames:
    """Tests for duplicate name handling."""

    @pytest.mark.parametrize(
        "count,expected_name",
        [
            (1, "data"),
            (2, "data-1"),
            (3, "data-2"),
        ],
    )
    def test_add_dataset_duplicate_names(self, ds1, count, expected_name):
        proj = Project(name="test")
        ds1.name = "data"
        proj.add_dataset(ds1)

        for i in range(1, count):
            ds = NDDataset([i], name="data")
            proj.add_dataset(ds)

        assert "data" in proj.datasets_names
        assert expected_name in proj.datasets_names


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

    def test_remove_script(self):
        proj = Project(name="test")
        script = Script("myscript", "x = 1")
        proj.add_script(script)
        assert "myscript" in proj.scripts_names
        proj.remove_script("myscript")
        assert "myscript" not in proj.scripts_names
        assert script.parent is None

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

    def test_clear_scripts(self):
        proj = Project(name="test")
        script1 = Script("script1", "x = 1")
        script2 = Script("script2", "y = 2")
        proj.add_scripts(script1, script2)
        assert len(proj.scripts_names) == 2
        assert script1.parent == proj
        assert script2.parent == proj
        proj.clear_scripts()
        assert len(proj.scripts_names) == 0
        assert script1.parent is None
        assert script2.parent is None

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

    def test_remove_all_script_deprecated(self):
        proj = Project(name="test")
        proj.add_scripts(
            Script("script1", "x = 1"),
            Script("script2", "y = 2"),
        )
        assert len(proj.scripts_names) == 2
        with pytest.warns(DeprecationWarning, match="remove_all_script.*deprecated"):
            proj.remove_all_script()
        assert len(proj.scripts_names) == 0


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


class TestMakescriptDecorator:
    """Tests for makescript decorator."""

    @pytest.mark.parametrize("priority", [None, 75])
    def test_makescript_decorator(self, priority):
        kwargs = {"priority": priority} if priority else {}

        @makescript(**kwargs)
        def test_func():
            return 42

        assert test_func() == 42


class TestAttributes:
    """Tests for _attributes_ method."""

    def test_attributes_returns_expected_list(self):
        proj = Project(name="test")
        attrs = proj._attributes_()
        expected = ["name", "meta", "parent", "datasets", "projects", "scripts"]
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
        proj.add_script(Script("script", "x = 1"))

        all_names = proj.allnames
        assert "data" in all_names
        assert "sub" in all_names
        assert "script" in all_names

        items = proj.allitems
        item_names = [item[0] for item in items]
        assert "data" in item_names
        assert "sub" in item_names
        assert "script" in item_names

    def test_readonly_id_property(self):
        proj = Project(name="test")
        assert proj.id is not None
        assert proj.id.startswith("Project_")

    def test_directory_property(self):
        proj = Project(name="test")
        assert proj.directory is None


class TestOthersDict:
    """Tests for _others dictionary functionality."""

    def test_others_dict_populated(self):
        proj = Project(name="test")

        class CustomObject:
            name = "custom"

        obj = CustomObject()
        proj._set_from_type(obj)
        assert "custom" in proj._others
        assert proj._others["custom"] is obj


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

    def test_multiple_scripts(self):
        proj = Project(name="test")
        proj.add_scripts(
            Script("script1", "x = 1"),
            Script("script2", "y = 2"),
            Script("script3", "z = 3"),
        )
        assert len(proj.scripts_names) == 3

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

    def test_save_load_with_scripts(self):
        proj = Project(name="script_test")
        proj.add_script(Script("test_script", "x = 42"))

        filename = proj.save()
        loaded = Project.load(filename)

        assert "test_script" in loaded.scripts_names
        assert loaded.test_script.content == "x = 42"

    def test_save_as_creates_new_file(self, ds1):
        proj = Project(name="overwrite_test")
        proj.add_dataset(ds1)

        filename1 = proj.save()
        filename2 = proj.save_as("overwrite_test_v2")

        assert filename1.exists()
        assert filename2.exists()
        assert filename1 != filename2


class TestArgNamesEdgeCases:
    """Test argnames edge cases."""

    def test_argnames_wrong_length_raises(self, ds1, ds2):
        ds1.name = "first"
        ds2.name = "second"
        with pytest.raises(IndexError):
            Project(ds1, ds2, argnames=["only_one"])
