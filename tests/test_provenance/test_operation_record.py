import json
from dataclasses import FrozenInstanceError
from datetime import UTC
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pytest

import spectrochempy as scp
from spectrochempy.provenance import OPERATION_RECORD_SCHEMA_ID
from spectrochempy.provenance import OPERATION_RECORD_SCHEMA_VERSION
from spectrochempy.provenance import ObjectRef
from spectrochempy.provenance import OperationRecord
from spectrochempy.provenance import OperationRef
from spectrochempy.provenance import ProvenanceValidationError
from spectrochempy.provenance import ReferenceLink
from spectrochempy.provenance import ResultRef
from spectrochempy.provenance import StateRef


def _record(**overrides):
    source = StateRef("state-000001", ObjectRef("object-000001"), 0)
    result = StateRef("state-000002", ObjectRef("object-000002"), 0)
    values = {
        "id": "op-000001",
        "operation_id": "org.spectrochempy.test.transform",
        "category": "transform",
        "implementation": "spectrochempy.testing.transform",
        "provider_name": "spectrochempy",
        "provider_version": "0.13.dev0",
        "started_at": datetime(2026, 9, 14, 8, 0, tzinfo=UTC),
        "inputs": (ReferenceLink("source", source),),
        "outputs": (ReferenceLink("result", result),),
        "requested_parameters": {"dim": "x"},
        "resolved_parameters": {"dim": "x", "inplace": False},
    }
    values.update(overrides)
    return OperationRecord(**values)


def test_reference_models_are_validated_and_immutable():
    object_ref = ObjectRef("object-000001")
    state_ref = StateRef("state-000001", object_ref, 2)
    result_ref = ResultRef("result-000001")
    operation_ref = OperationRef("op-000001")

    assert state_ref.object is object_ref
    assert state_ref.version == 2
    assert ReferenceLink("input", state_ref).to_dict() == {
        "role": "input",
        "kind": "state",
        "id": "state-000001",
    }
    assert ReferenceLink("score", result_ref).to_dict()["kind"] == "result"
    assert operation_ref.kind == "operation"
    with pytest.raises(FrozenInstanceError):
        object_ref.id = "changed"


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ObjectRef(""), "non-empty"),
        (lambda: StateRef("state", ObjectRef("object"), -1), "non-negative"),
        (lambda: StateRef("state", ObjectRef("object"), True), "integer"),
        (lambda: ReferenceLink("", ResultRef("result")), "non-empty"),
    ],
)
def test_reference_models_reject_invalid_values(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_operation_record_has_versioned_detached_json_compatible_view():
    record = _record(parent_operation=OperationRef("op-parent"))
    payload = record.to_dict()

    assert payload["schema"] == {
        "id": OPERATION_RECORD_SCHEMA_ID,
        "version": OPERATION_RECORD_SCHEMA_VERSION,
    }
    assert "sequence" not in payload
    assert payload["operation"] == {
        "id": "org.spectrochempy.test.transform",
        "category": "transform",
        "implementation": "spectrochempy.testing.transform",
    }
    assert payload["inputs"][0]["role"] == "source"
    assert payload["parent_operation"] == {"kind": "operation", "id": "op-parent"}
    json.dumps(payload, allow_nan=False)

    payload["parameters"]["requested"]["values"]["dim"] = "changed"
    assert record.to_dict()["parameters"]["requested"]["values"]["dim"] == "x"


def test_diagnostic_implementation_is_optional():
    record = OperationRecord(
        id="op-without-implementation",
        operation_id="org.spectrochempy.test.transform",
        category="transform",
        provider_name="spectrochempy",
        started_at=datetime(2026, 9, 14, 8, 0, tzinfo=UTC),
        outputs=(ReferenceLink("result", ResultRef("result-optional")),),
    )
    payload = record.to_dict()
    assert payload["operation"] == {
        "id": "org.spectrochempy.test.transform",
        "category": "transform",
    }


def test_record_cannot_be_its_own_parent():
    with pytest.raises(ValueError, match="cannot reference the record itself"):
        _record(parent_operation=OperationRef("op-000001"))


def test_operation_record_is_deeply_immutable():
    parameters = {"options": ["a", "b"]}
    record = _record(requested_parameters=parameters)
    parameters["options"].append("changed")

    with pytest.raises(FrozenInstanceError):
        record.status = "failed"
    with pytest.raises(TypeError):
        record.requested_parameters["new"] = 1
    assert record.to_dict()["parameters"]["requested"]["values"] == {
        "options": ["a", "b"]
    }


def test_explicit_record_construction_is_strict_by_default():
    with pytest.raises(ProvenanceValidationError, match="Unsupported"):
        _record(requested_parameters={"unsafe": object()})


def test_best_effort_record_marks_unsupported_parameter_as_partial():
    record = _record(
        requested_parameters={"unsafe": object()},
        strict=False,
    )
    payload = record.to_dict()

    assert payload["parameters"]["requested"]["values"]["unsafe"] == {
        "status": "unsupported",
        "reason": "unsupported_type",
        "type": "builtins.object",
    }
    assert payload["capture"]["status"] == "partial"
    assert payload["capture"]["omissions"] == [
        {"field": "parameters.requested.unsafe", "status": "unsupported"}
    ]
    assert payload["parameters"]["requested"]["fields"] == {
        "unsafe": {"status": "unsupported"}
    }


@pytest.mark.parametrize("capture_status", ["partial", "unknown"])
def test_detected_omissions_merge_with_existing_capture(capture_status):
    duplicate = {
        "field": "parameters.requested.unsafe",
        "status": "unsupported",
    }
    record = _record(
        requested_parameters={"unsafe": object()},
        strict=False,
        capture={
            "status": capture_status,
            "omissions": [
                {"field": "provider.entry_point", "status": "unknown"},
                duplicate,
            ],
        },
    )
    payload = record.to_dict()

    assert payload["capture"]["status"] == capture_status
    assert payload["capture"]["omissions"] == [
        duplicate,
        {"field": "provider.entry_point", "status": "unknown"},
    ]


def test_redacted_path_parameter_marks_capture_partial():
    record = _record(requested_parameters={"source": Path("/not/a/path")})
    payload = record.to_dict()

    assert payload["parameters"]["requested"]["values"]["source"] == {
        "type": "path",
        "value": "path",
        "redaction": "basename",
    }
    assert payload["capture"]["status"] == "partial"


def test_omitted_coordinate_labels_mark_capture_partial():
    coordinate = scp.Coord([1.0, 2.0], labels=["first", "second"])
    record = _record(requested_parameters={"coordinate": coordinate})
    payload = record.to_dict()

    assert payload["parameters"]["requested"]["fields"] == {}
    assert payload["capture"]["status"] == "partial"
    assert {
        "field": "parameters.requested.coordinate.labels",
        "status": "omitted",
    } in payload["capture"]["omissions"]


@pytest.mark.parametrize("status", [{"enabled": True}, ["enabled"]])
def test_non_string_status_value_is_treated_as_user_data(status):
    record = _record(
        requested_parameters={"options": {"status": status}},
        strict=False,
    )
    payload = record.to_dict()

    assert payload["parameters"]["requested"]["values"]["options"] == {"status": status}
    assert payload["parameters"]["requested"]["fields"] == {}
    assert payload["capture"] == {"status": "complete", "omissions": []}


def test_complete_capture_with_omissions_is_downgraded_to_partial():
    omission = {"field": "provider.entry_point", "status": "unknown"}
    record = _record(capture={"status": "complete", "omissions": [omission]})
    payload = record.to_dict()

    assert payload["capture"]["status"] == "partial"
    assert payload["capture"]["omissions"] == [omission]


def test_unsupported_warning_value_marks_capture_partial():
    record = _record(warnings=({"note": object()},), strict=False)
    payload = record.to_dict()

    assert payload["capture"]["status"] == "partial"
    assert payload["capture"]["omissions"] == [
        {"field": "warnings[0].note", "status": "unsupported"}
    ]


def test_unknown_provider_version_is_explicit():
    record = _record(provider_name="unknown-provider", provider_version=None)
    assert record.to_dict()["provider"]["version"] == {
        "status": "unknown",
        "reason": "not_discoverable",
    }


def test_core_provider_version_is_discovered_when_omitted():
    record = _record(provider_version=None)
    assert record.to_dict()["provider"]["version"] == version("spectrochempy")


def test_failed_record_cannot_claim_outputs():
    with pytest.raises(ValueError, match="cannot contain outputs"):
        _record(status="failed")


def test_successful_record_requires_an_output():
    with pytest.raises(ValueError, match="must contain outputs"):
        _record(outputs=())


def test_failed_mutating_record_can_report_unknown_state_effect():
    record = _record(
        status="failed",
        outputs=(),
        state_effect={"effect": "unknown", "reconstructable": False},
        capture={
            "status": "partial",
            "omissions": [
                {"field": "state_effect", "status": "unknown"},
            ],
        },
    )
    assert record.to_dict()["state_effect"] == {
        "effect": "unknown",
        "reconstructable": False,
    }


def test_successful_fit_has_summary_or_explicit_omission():
    omitted = _record(category="fit").to_dict()
    summarized = _record(
        category="fit",
        fitted_state_summary={"components": {"status": "omitted"}},
    ).to_dict()

    assert omitted["fitted_state_summary"] == {
        "status": "omitted",
        "reason": "not_captured",
    }
    assert omitted["capture"]["status"] == "partial"
    assert summarized["fitted_state_summary"] == {"components": {"status": "omitted"}}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"operation_id": "transform"}, "namespaced"),
        ({"category": "unknown"}, "category"),
        ({"implementation": "transform"}, "fully qualified"),
        ({"status": "unknown"}, "status"),
        ({"capture": {"status": "invalid"}}, "capture status"),
        ({"capture": {"status": "complete", "omissions": "invalid"}}, "omissions"),
        ({"requested_parameters": []}, "requested_parameters"),
        ({"strict": 1}, "strict"),
    ],
)
def test_operation_record_rejects_invalid_contract_fields(overrides, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _record(**overrides)
