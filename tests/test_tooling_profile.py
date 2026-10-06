"""Exercise deferred annotations on the maintained record and a forward reference."""

import annotationlib

from paa.opportunity_records import ActionKind, ActionRequirements, RequiredCapability


def _deferred(record: _DefinedAfterFunction) -> _DefinedAfterFunction:
    return record


class _DefinedAfterFunction:
    pass


def test_python_314_deferred_annotations_are_values_at_the_consumer():
    annotations = annotationlib.get_annotations(_deferred, format=annotationlib.Format.VALUE)
    assert annotations == {"record": _DefinedAfterFunction, "return": _DefinedAfterFunction}
    fields = annotationlib.get_annotations(ActionRequirements, format=annotationlib.Format.VALUE)
    assert fields["action_kind"] is ActionKind
    assert fields["required_capability"] is RequiredCapability
    assert fields["evidence_ids"] == tuple[str, ...]
