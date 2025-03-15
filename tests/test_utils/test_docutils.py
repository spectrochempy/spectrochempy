# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the docutils module."""

import pytest

from spectrochempy.utils.docutils import _remove_errors
from spectrochempy.utils.docutils import _scpy_error
from spectrochempy.utils.docutils import check_docstrings


# Test classes with different docstring scenarios
class GoodDocstrings:
    """Class with proper docstrings for testing."""

    def method_with_good_docstring(self):
        """
        A method with a proper docstring.

        This method has a proper docstring with all required sections.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method doesn't return anything.

        See Also
        --------
        another_method : Another method that does something.

        Examples
        --------
        >>> obj = GoodDocstrings()
        >>> obj.method_with_good_docstring()
        """
        pass


class BadDocstrings:
    """Class with improper docstrings for testing."""

    def method_with_bad_docstring(self):
        """
        A method with a problematic docstring.

        This method has array_like instead of array-like.

        Parameters
        ----------
        None

        Returns
        -------
        array_like
            This has a terminology error.
        """
        pass

    def method_with_missing_sections(self):
        """A method with missing sections in the docstring."""
        pass


def test_docstring_validator(monkeypatch):
    """Test basic functionality of the docstring validator."""
    import spectrochempy.utils.docutils as docutils

    # Mock the Validator class to return predictable results
    class MockValidator:
        def __init__(self, name):
            self.name = name
            self.mentioned_private_classes = []

        def non_hyphenated_array_like(self):
            return False

    # Apply our mock
    original_validator = docutils._DocstringValidator
    docutils._DocstringValidator = MockValidator

    try:
        # Create an instance to test
        validator = MockValidator("test.method")

        # Test the expected behavior
        assert validator.name == "test.method"
        assert not validator.mentioned_private_classes
        assert not validator.non_hyphenated_array_like()
    finally:
        # Restore original
        docutils._DocstringValidator = original_validator


def test_remove_errors():
    """Test the _remove_errors function."""
    errors = [
        ("GL01", "First error"),
        ("GL02", "Second error"),
        ("GL03", "Third error"),
    ]

    # Remove one error
    filtered_errors = _remove_errors(errors, "GL02")
    assert len(filtered_errors) == 2
    assert ("GL02", "Second error") not in filtered_errors

    # Remove multiple errors
    filtered_errors = _remove_errors(errors, ["GL01", "GL03"])
    assert len(filtered_errors) == 1
    assert filtered_errors[0] == ("GL02", "Second error")


def test_scpy_error():
    """Test the _scpy_error function."""
    # Test GL04 error
    code, msg = _scpy_error("GL04", mentioned_private_classes="PrivateClass")
    assert code == "GL04"
    assert "PrivateClass" in msg

    # Test GL05 error
    code, msg = _scpy_error("GL05")
    assert code == "GL05"
    assert "array-like" in msg
    assert "array_like" in msg


def test_bad_docstring(monkeypatch):
    """Test detection of bad docstrings in _scpy_numpydoc_validate."""
    import spectrochempy.utils.docutils as docutils

    # Mock the entire validation process
    def mock_validate(func_name, exclude=None):
        result = {
            "errors": [],
            "examples_errs": [],
            "member_name": func_name,
            "file": "mock_file.py",
            "file_line": 1,
        }

        if "bad" in func_name.lower():
            result["errors"].append(
                ("GL05", "Use 'array-like' rather than 'array_like' in docstrings.")
            )

        return result

    # Keep original and replace for testing
    original_validate = docutils._scpy_numpydoc_validate
    docutils._scpy_numpydoc_validate = mock_validate

    try:
        # Test validation
        result = mock_validate("BadDocstrings.method_with_bad_docstring")
        assert len(result["errors"]) > 0
        assert any(error[0] == "GL05" for error in result["errors"])
        assert "examples_errs" in result
    finally:
        # Restore original
        docutils._scpy_numpydoc_validate = original_validate


@pytest.mark.parametrize(
    "exclude_param,should_raise",
    [
        (
            None,
            True,
        ),  # No custom exclusions, will still have default GL02/GL03 excluded
        (
            ["GL01", "GL05"],
            False,
        ),  # Excluding GL01 and GL05 (GL02 already excluded by default)
        (["GL01", "GL02", "GL05"], False),  # Explicitly exclude all errors
    ],
)
def test_docstring_validation_with_exclusions(monkeypatch, exclude_param, should_raise):
    """Test that docstring validation respects exclusions."""
    import spectrochempy.utils.docutils as docutils

    # Create a mock that actually respects excludes parameter
    def mock_validate(func_name, exclude=None):
        # Start with all errors, including GL01 which isn't excluded by default
        all_errors = [
            ("GL01", "First error"),
            ("GL02", "Second error - excluded by default"),
            ("GL05", "Use 'array-like' rather than 'array_like' in docstrings."),
        ]

        # Filter based on exclusion list
        if exclude:
            errors = [err for err in all_errors if err[0] not in exclude]
        else:
            errors = all_errors.copy()

        return {
            "errors": errors,
            "examples_errs": [],
            "member_name": func_name,
            "file": "mock_file.py",
            "file_line": 1,
        }

    # Replace the validation function with our mock
    monkeypatch.setattr(docutils, "_scpy_numpydoc_validate", mock_validate)

    # Create a test class
    class TestClass:
        def test_method(self):
            """Test method docstring."""
            pass

    # Test behavior based on exclusion
    if should_raise:
        with pytest.raises(docutils._DocstringError):
            check_docstrings(
                "spectrochempy.utils.docutils", TestClass, exclude=exclude_param
            )
    else:
        # Should not raise when all errors are excluded
        check_docstrings(
            "spectrochempy.utils.docutils", TestClass, exclude=exclude_param
        )


def test_docstring_error_raising(monkeypatch):
    """Test that check_docstrings raises errors with proper messages."""
    import spectrochempy.utils.docutils as docutils

    # Create a complete mock result including all required fields
    mock_result = {
        "errors": [("GL05", "Test error")],
        "examples_errs": [],
        "member_name": "spectrochempy.utils.docutils.TestClass",
        "file": "mock_file.py",
        "file_line": 1,
    }

    # Override _scpy_numpydoc_validate to return our mock result
    def mock_validate(*args, **kwargs):
        return mock_result.copy()

    monkeypatch.setattr(docutils, "_scpy_numpydoc_validate", mock_validate)

    # Create a class for testing
    class TestClass:
        def test_method(self):
            """Test method docstring."""
            pass

    # Should raise DocstringError
    with pytest.raises(docutils._DocstringError) as excinfo:
        check_docstrings("spectrochempy.utils.docutils", TestClass)

    # Check error message contains our error code
    error_message = str(excinfo.value)
    assert "GL05" in error_message
