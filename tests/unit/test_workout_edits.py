"""Unit tests for in-place workout editing helpers.

These cover the change-spec logic in isolation. The behaviours they assert were
established against the live Garmin API: stepOrder is global across the whole
workout, a repeat group stores its count twice, and Garmin persists whatever
estimate it is sent rather than deriving one from the steps.
"""
import pytest

from garmin_mcp.workouts import (
    _apply_workout_changes,
    _index_steps_by_order,
    _prepare_workout_payload,
    _resolve_editable_workout_id,
)


def _walk_run_dto():
    """A DTO shaped like the one Garmin returns for a walk/run workout.

    stepOrder runs globally: the repeat group is 2 and its children are 3 and
    4, so the cooldown continues at 5.
    """
    return {
        "workoutId": 111,
        "workoutName": "Walk Run",
        "estimatedDurationInSecs": 1860,
        "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
        "workoutSegments": [{
            "segmentOrder": 1,
            "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
            "workoutSteps": [
                {
                    "type": "ExecutableStepDTO",
                    "stepOrder": 1,
                    "stepType": {"stepTypeId": 1, "stepTypeKey": "warmup"},
                    "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                    "endConditionValue": 600.0,
                    "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"},
                    "zoneNumber": None,
                    "targetValueOne": None,
                    "targetValueTwo": None,
                },
                {
                    "type": "RepeatGroupDTO",
                    "stepOrder": 2,
                    "stepType": {"stepTypeId": 6, "stepTypeKey": "repeat"},
                    "numberOfIterations": 4,
                    "endConditionValue": 4.0,
                    "endCondition": {"conditionTypeId": 7, "conditionTypeKey": "iterations"},
                    "workoutSteps": [
                        {
                            "type": "ExecutableStepDTO",
                            "stepOrder": 3,
                            "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                            "endConditionValue": 120.0,
                            "targetType": {"workoutTargetTypeId": 4, "workoutTargetTypeKey": "heart.rate.zone"},
                            "zoneNumber": 3,
                            "targetValueOne": None,
                            "targetValueTwo": None,
                        },
                        {
                            "type": "ExecutableStepDTO",
                            "stepOrder": 4,
                            "stepType": {"stepTypeId": 4, "stepTypeKey": "recovery"},
                            "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                            "endConditionValue": 120.0,
                            "targetType": {"workoutTargetTypeId": 4, "workoutTargetTypeKey": "heart.rate.zone"},
                            "zoneNumber": 3,
                            "targetValueOne": None,
                            "targetValueTwo": None,
                        },
                    ],
                },
                {
                    "type": "ExecutableStepDTO",
                    "stepOrder": 5,
                    "stepType": {"stepTypeId": 2, "stepTypeKey": "cooldown"},
                    "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                    "endConditionValue": 300.0,
                    "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"},
                    "zoneNumber": None,
                    "targetValueOne": None,
                    "targetValueTwo": None,
                },
            ],
        }],
    }


def _steps(dto):
    return dto["workoutSegments"][0]["workoutSteps"]


def _nested(dto, position):
    return _steps(dto)[1]["workoutSteps"][position]


class TestStepIndexing:
    def test_indexes_nested_steps_by_global_order(self):
        index = _index_steps_by_order(_walk_run_dto())
        assert sorted(index) == [1, 2, 3, 4, 5]
        assert index[3]["stepType"]["stepTypeKey"] == "interval"
        assert index[2]["type"] == "RepeatGroupDTO"

    def test_duplicate_order_is_rejected(self):
        dto = _walk_run_dto()
        _steps(dto)[2]["stepOrder"] = 1
        with pytest.raises(ValueError, match="duplicate stepOrder"):
            _index_steps_by_order(dto)


class TestApplyChanges:
    def test_renames_workout(self):
        dto = _walk_run_dto()
        applied = _apply_workout_changes(dto, {"name": "Renamed"})
        assert dto["workoutName"] == "Renamed"
        assert applied == ["name -> 'Renamed'"]

    def test_edits_nested_step_by_order(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [{"order": 3, "end_condition_value": 90}]})
        assert _nested(dto, 0)["endConditionValue"] == 90.0
        # Siblings are untouched
        assert _nested(dto, 1)["endConditionValue"] == 120.0

    def test_repeat_count_updates_both_stored_copies(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [{"order": 2, "repeat_count": 6}]})
        group = _steps(dto)[1]
        # Garmin keeps the count in both fields and honours whichever it reads
        # first, so they must move together.
        assert group["numberOfIterations"] == 6
        assert group["endConditionValue"] == 6.0

    def test_setting_zone_clears_custom_range(self):
        dto = _walk_run_dto()
        _nested(dto, 0).update({"zoneNumber": None, "targetValueOne": 130.0, "targetValueTwo": 145.0})
        _apply_workout_changes(dto, {"steps": [{"order": 3, "target_zone": 2}]})
        step = _nested(dto, 0)
        assert step["zoneNumber"] == 2
        assert step["targetValueOne"] is None
        assert step["targetValueTwo"] is None

    def test_setting_custom_range_clears_zone(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [
            {"order": 3, "target_value_low": 136, "target_value_high": 148},
        ]})
        step = _nested(dto, 0)
        assert step["zoneNumber"] is None
        assert (step["targetValueOne"], step["targetValueTwo"]) == (136.0, 148.0)

    def test_target_type_resolves_canonical_id(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [{"order": 3, "target_type": "power.between"}]})
        assert _nested(dto, 0)["targetType"] == {
            "workoutTargetTypeId": 6,
            "workoutTargetTypeKey": "power.between",
        }

    def test_no_target_clears_target_values(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [{"order": 3, "target_type": "no.target"}]})
        step = _nested(dto, 0)
        assert step["zoneNumber"] is None
        assert step["targetValueOne"] is None

    def test_end_condition_resolves_canonical_id(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [
            {"order": 3, "end_condition": "distance", "end_condition_value": 400},
        ]})
        step = _nested(dto, 0)
        assert step["endCondition"] == {"conditionTypeId": 3, "conditionTypeKey": "distance"}
        assert step["endConditionValue"] == 400.0

    def test_step_type_resolves_canonical_id(self):
        dto = _walk_run_dto()
        _apply_workout_changes(dto, {"steps": [{"order": 1, "type": "rest"}]})
        assert _steps(dto)[0]["stepType"] == {"stepTypeId": 5, "stepTypeKey": "rest"}


class TestApplyChangesRejects:
    @pytest.mark.parametrize("changes,message", [
        ({}, "nothing to update"),
        ({"nome": "typo"}, "Unknown change field"),
        ({"name": "  "}, "non-empty string"),
        ({"steps": [{"description": "no order"}]}, "missing 'order'"),
        ({"steps": [{"order": 99, "description": "x"}]}, "no step with order 99"),
        ({"steps": [{"order": 1, "duration": 5}]}, "unknown field"),
        ({"steps": [{"order": 1, "type": "sprint"}]}, "unknown type"),
        ({"steps": [{"order": 1, "end_condition": "vibes"}]}, "unknown end_condition"),
        ({"steps": [{"order": 1, "target_type": "mood.zone"}]}, "unknown target_type"),
        ({"steps": [{"order": 1, "repeat_count": 3}]}, "applies to a repeat group"),
        ({"steps": [{"order": 2, "repeat_count": 0}]}, "at least 1"),
        ({"steps": [{"order": 3, "target_value_low": 100}]}, "must be given together"),
        ({"steps": [{"order": 3, "target_value_low": 150, "target_value_high": 120}]}, "must be less than"),
        ({"steps": [{"order": 3, "target_zone": 2, "target_value_low": 1, "target_value_high": 2}]},
         "not both"),
        ({"steps": "not a list"}, "must be a list"),
    ])
    def test_invalid_change_spec(self, changes, message):
        with pytest.raises(ValueError, match=message):
            _apply_workout_changes(_walk_run_dto(), changes)

    def test_bad_order_lists_available_orders(self):
        with pytest.raises(ValueError, match=r"Available orders: 1, 2, 3, 4, 5"):
            _apply_workout_changes(_walk_run_dto(), {"steps": [{"order": 99, "description": "x"}]})


class TestPreparePayload:
    def test_strips_derived_estimates(self):
        dto = _walk_run_dto()
        dto["avgTrainingSpeed"] = 2.5
        dto["workoutSegments"][0]["estimatedDurationInSecs"] = 1860
        _prepare_workout_payload(dto, 111)
        # Garmin persists any estimate it is sent without checking it against
        # the steps, so an edited workout must carry none and let Garmin derive.
        assert "estimatedDurationInSecs" not in dto
        assert "avgTrainingSpeed" not in dto
        assert "estimatedDurationInSecs" not in dto["workoutSegments"][0]

    def test_sets_body_id_to_target_id(self):
        dto = _walk_run_dto()
        dto["workoutId"] = 999
        _prepare_workout_payload(dto, 111)
        assert dto["workoutId"] == 111

    def test_validation_still_applies(self):
        dto = _walk_run_dto()
        # conditionTypeId 4 is "calories"; Garmin would silently reinterpret it.
        _steps(dto)[0]["endCondition"] = {"conditionTypeId": 4, "conditionTypeKey": "heart.rate"}
        with pytest.raises(ValueError, match="conditionTypeId"):
            _prepare_workout_payload(dto, 111)


class TestResolveEditableId:
    def test_accepts_numeric_id(self):
        assert _resolve_editable_workout_id("123") == 123
        assert _resolve_editable_workout_id(123) == 123

    def test_rejects_training_plan_uuid(self):
        with pytest.raises(ValueError, match="cannot be edited"):
            _resolve_editable_workout_id("fd3d0a6b-1234-4c1e-9c2a-aaaabbbbcccc")

    @pytest.mark.parametrize("bad", ["", "abc", 0, -5])
    def test_rejects_invalid_id(self, bad):
        with pytest.raises(ValueError):
            _resolve_editable_workout_id(bad)
