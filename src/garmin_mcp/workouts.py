"""
Workout-related functions for Garmin Connect MCP Server
"""
import json
import re
import datetime
from typing import Any, Dict, List, Optional, Union

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _validate_date(value: str, field: str = "date") -> str:
    if not _DATE_RE.match(value):
        raise ValueError(f"Invalid {field} '{value}': expected YYYY-MM-DD")

# The garmin_client will be set by the main file
garmin_client = None

END_CONDITION_TYPE_IDS = {
    "lap.button": 1,
    "time": 2,
    "distance": 3,
    "calories": 4,
    "power": 5,
    "heart.rate": 6,
    "iterations": 7,
    "fixed.rest": 8,
    "fixed.repetition": 9,
    "reps": 10,
    "training.peaks.tss": 11,
}
END_CONDITION_TYPE_KEYS = {
    condition_id: condition_key
    for condition_key, condition_id in END_CONDITION_TYPE_IDS.items()
}

# Verified from Garmin-created workouts and live upload/fetch probes. Unknown
# target type IDs are allowed so we do not block valid Garmin targets that are
# not in this partial mapping yet.
KNOWN_TARGET_TYPE_IDS = {
    1: frozenset(["no.target"]),
    2: frozenset(["power.zone"]),
    4: frozenset(["heart.rate.zone"]),
    # ID 6 is sport-context-dependent:
    #   - running / swimming: "pace.zone"
    #   - cycling: "power.between" (absolute watt range, uses targetValueOne/targetValueTwo)
    6: frozenset(["pace.zone", "power.between"]),
}

# Reverse map: workoutTargetTypeKey -> workoutTargetTypeId (each key maps to exactly one ID).
KNOWN_TARGET_TYPE_KEYS = {
    key: target_id
    for target_id, keys in KNOWN_TARGET_TYPE_IDS.items()
    for key in keys
}

# Verified against DTOs Garmin returns for workouts created by this server.
STEP_TYPE_IDS = {
    "warmup": 1,
    "cooldown": 2,
    "interval": 3,
    "recovery": 4,
    "rest": 5,
    "repeat": 6,
}

def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client


_TARGET_FIELD_LAYOUTS = {
    'targetType': {
        'bounds': ('targetValueOne', 'targetValueTwo'),
        'zone': 'zoneNumber',
    },
    'secondaryTargetType': {
        'bounds': ('secondaryTargetValueOne', 'secondaryTargetValueTwo'),
        'zone': 'secondaryZoneNumber',
    },
}


def _iter_step_tree(step: dict, path: str):
    """Yield a workout step and every nested step with its request path."""
    yield step, path
    for index, nested in enumerate(step.get('workoutSteps', [])):
        yield from _iter_step_tree(
            nested,
            f"{path}.workoutSteps[{index}]",
        )


def _iter_workout_steps(workout_data: dict):
    """Yield every workout step with a stable request path."""
    for segment_index, segment in enumerate(
        workout_data.get('workoutSegments', [])
    ):
        for step_index, step in enumerate(segment.get('workoutSteps', [])):
            path = (
                f"workoutSegments[{segment_index}]"
                f".workoutSteps[{step_index}]"
            )
            yield from _iter_step_tree(step, path)


def _validate_nested_target_fields(step: dict, path: str) -> None:
    """Reject conflicting or ambiguous target fields before any repair."""
    for target_field, layout in _TARGET_FIELD_LAYOUTS.items():
        target_type = step.get(target_field)
        if not isinstance(target_type, dict):
            continue

        fields = (*layout['bounds'], layout['zone'])
        for field in fields:
            nested_value = target_type.get(field)
            if (
                nested_value is not None
                and step.get(field) is not None
                and step[field] != nested_value
            ):
                raise ValueError(
                    f"{path}.{field}={step[field]!r} conflicts with "
                    f"{path}.{target_field}.{field}={nested_value!r}; "
                    f"keep only the step-level {field}"
                )

        zone_field = layout['zone']
        zone_value = step.get(zone_field)
        if zone_value is None:
            zone_value = target_type.get(zone_field)

        bound_values = []
        for field in layout['bounds']:
            value = step.get(field)
            if value is None:
                value = target_type.get(field)
            if value is not None:
                bound_values.append((field, value))

        if zone_value is not None and bound_values:
            bounds = ", ".join(
                f"{field}={value!r}"
                for field, value in bound_values
            )
            raise ValueError(
                f"{path} mixes {zone_field}={zone_value!r} with custom "
                f"range fields ({bounds}); use either a named zone or a "
                f"custom range"
            )


def _move_nested_target_fields(step: dict) -> None:
    """Move target fields to the step level, where Garmin reads them."""
    for target_field, layout in _TARGET_FIELD_LAYOUTS.items():
        target_type = step.get(target_field)
        if not isinstance(target_type, dict):
            continue

        fields = (*layout['bounds'], layout['zone'])
        for field in fields:
            if field not in target_type:
                continue
            value = target_type.pop(field)
            if value is not None and step.get(field) is None:
                step[field] = value


def _fix_hr_zone_step(step: dict) -> None:
    """Fix a common mistake where HR zone targets use targetValueOne instead of zoneNumber.

    When targetType is heart.rate.zone and a named zone is intended, Garmin expects
    zoneNumber (1-5). If targetValueOne is set to a small integer (1-5) and zoneNumber
    is missing, this is almost certainly a zone number, not an absolute HR value.

    Custom HR bpm ranges (e.g. targetValueOne=105, targetValueTwo=143) are left
    unchanged — these are legitimate custom heart rate targets in Garmin Connect.
    """
    target_type = step.get('targetType')
    target_key = (
        target_type.get('workoutTargetTypeKey', '')
        if isinstance(target_type, dict)
        else ''
    )

    if target_key == 'heart.rate.zone' and 'zoneNumber' not in step:
        zone = step.get('targetValueOne')
        if zone is not None and 1 <= zone <= 5:
            step['zoneNumber'] = int(zone)
            step.pop('targetValueOne', None)
            step.pop('targetValueTwo', None)

    # Recurse into nested steps (RepeatGroupDTO)
    for nested in step.get('workoutSteps', []):
        _fix_hr_zone_step(nested)


def _fix_repeat_group_step(step: dict) -> None:
    """Ensure RepeatGroupDTO steps have a valid endCondition and numberOfIterations.

    The Garmin API silently corrupts a RepeatGroupDTO when conditionTypeId is
    missing from its endCondition — it falls back to an unrelated condition type
    (observed: "heart.rate") and drops numberOfIterations entirely.

    This function:
    - Adds conditionTypeId: 7 ("iterations") when conditionTypeKey is "iterations"
      but conditionTypeId is absent.
    - Backfills numberOfIterations from endConditionValue when the former is missing.
    - Recurses into nested workoutSteps so nested repeat groups are also fixed.
    """
    if step.get('type') != 'RepeatGroupDTO':
        for nested in step.get('workoutSteps', []):
            _fix_repeat_group_step(nested)
        return

    end_condition = step.get('endCondition')
    if isinstance(end_condition, dict):
        if (
            end_condition.get('conditionTypeKey') == 'iterations'
            and 'conditionTypeId' not in end_condition
        ):
            end_condition['conditionTypeId'] = 7

    if 'numberOfIterations' not in step:
        value = step.get('endConditionValue')
        if value is not None:
            step['numberOfIterations'] = int(value)

    for nested in step.get('workoutSteps', []):
        _fix_repeat_group_step(nested)


def _normalize_workout_steps(workout_data: dict) -> None:
    """Repair recoverable step-shape mistakes before validation and upload."""
    steps = list(_iter_workout_steps(workout_data))

    # Preflight the complete workout so a later conflict cannot leave an
    # earlier step partially repaired.
    for step, path in steps:
        _validate_nested_target_fields(step, path)
    for step, _ in steps:
        _move_nested_target_fields(step)

    # These helpers recurse, so invoke them only for top-level steps.
    for segment in workout_data.get('workoutSegments', []):
        for step in segment.get('workoutSteps', []):
            _fix_hr_zone_step(step)
            _fix_repeat_group_step(step)


def _validate_end_condition_step(step: dict, path: str) -> None:
    """Reject endCondition id/key pairs Garmin would silently reinterpret."""
    end_condition = step.get('endCondition')
    if isinstance(end_condition, dict):
        condition_key = end_condition.get('conditionTypeKey')
        condition_id = end_condition.get('conditionTypeId')

        expected_id = END_CONDITION_TYPE_IDS.get(condition_key)
        expected_key = END_CONDITION_TYPE_KEYS.get(condition_id)

        if expected_id is not None:
            if condition_id is None:
                raise ValueError(
                    f"{path}.endCondition conditionTypeKey '{condition_key}' "
                    f"requires conditionTypeId {expected_id}"
                )
            if condition_id != expected_id:
                actual = expected_key or "unknown"
                raise ValueError(
                    f"{path}.endCondition conditionTypeKey '{condition_key}' "
                    f"requires conditionTypeId {expected_id}, got {condition_id} "
                    f"({actual})"
                )
        elif expected_key is not None and condition_key is not None:
            raise ValueError(
                f"{path}.endCondition conditionTypeId {condition_id} "
                f"requires conditionTypeKey '{expected_key}', got '{condition_key}'"
            )

    for index, nested in enumerate(step.get('workoutSteps', [])):
        _validate_end_condition_step(nested, f"{path}.workoutSteps[{index}]")


def _validate_end_condition_steps(workout_data: dict) -> None:
    """Validate all workout step endCondition blocks before upload."""
    for segment_index, segment in enumerate(workout_data.get('workoutSegments', [])):
        for step_index, step in enumerate(segment.get('workoutSteps', [])):
            path = f"workoutSegments[{segment_index}].workoutSteps[{step_index}]"
            _validate_end_condition_step(step, path)


def _validate_target_type_block(step: dict, path: str, target_field: str) -> None:
    """Reject a target type id/key pair Garmin would silently reinterpret."""
    target_type = step.get(target_field)
    if isinstance(target_type, dict):
        target_key = target_type.get('workoutTargetTypeKey')
        target_id = target_type.get('workoutTargetTypeId')

        if target_id is not None:
            try:
                target_id = int(target_id)
            except (TypeError, ValueError):
                raise ValueError(f"{path}.{target_field}.workoutTargetTypeId must be numeric")

        valid_keys = KNOWN_TARGET_TYPE_IDS.get(target_id)
        if valid_keys is not None and target_key is not None and target_key not in valid_keys:
            if len(valid_keys) == 1:
                (only_key,) = valid_keys
                raise ValueError(
                    f"{path}.{target_field} mismatch: workoutTargetTypeId {target_id} is "
                    f"{only_key!r}, not {target_key!r}"
                )
            else:
                valid_list = ", ".join(sorted(repr(k) for k in valid_keys))
                raise ValueError(
                    f"{path}.{target_field} mismatch: workoutTargetTypeId {target_id} is "
                    f"one of ({valid_list}), not {target_key!r}"
                )

        expected_id = KNOWN_TARGET_TYPE_KEYS.get(target_key)
        if expected_id is not None and target_id is not None and target_id != expected_id:
            raise ValueError(
                f"{path}.{target_field} mismatch: workoutTargetTypeKey {target_key!r} "
                f"requires workoutTargetTypeId {expected_id}, not {target_id}"
            )


def _validate_target_type_step(step: dict, path: str) -> None:
    """Reject targetType id/key pairs Garmin would silently reinterpret."""
    _validate_target_type_block(step, path, 'targetType')
    _validate_target_type_block(step, path, 'secondaryTargetType')

    for index, nested in enumerate(step.get('workoutSteps', [])):
        _validate_target_type_step(nested, f"{path}.workoutSteps[{index}]")


def _validate_target_type_steps(workout_data: dict) -> None:
    """Walk all workout steps and validate known targetType id/key pairs."""
    for segment_index, segment in enumerate(workout_data.get('workoutSegments', [])):
        for step_index, step in enumerate(segment.get('workoutSteps', [])):
            path = f"workoutSegments[{segment_index}].workoutSteps[{step_index}]"
            _validate_target_type_step(step, path)


# =============================================================================
# IN-PLACE EDITING
#
# Garmin updates a workout with PUT /workout-service/workout/{workoutId},
# carrying the complete workout DTO. Partial bodies are rejected with
# "There is an error with the workout segments", so every edit is a
# read-modify-write of the DTO Garmin itself returns.
#
# Editing in place keeps the workout id, so calendar entries that already
# point at the workout survive and follow the new content. Delete-and-
# re-upload does not: it mints a new id and orphans the schedule.
# =============================================================================

# Fields Garmin derives from the steps. Garmin does not validate them against
# the steps it is sent: a PUT carrying estimatedDurationInSecs 99999 for a
# 2700-second workout stores 99999. Dropping them makes Garmin recompute from
# the steps, which is the only way an edited workout cannot end up advertising
# a duration it does not have.
_ESTIMATE_FIELDS = (
    'estimatedDurationInSecs',
    'estimatedDistanceInMeters',
    'estimatedDuration',
    'estimatedDistance',
    'avgTrainingSpeed',
)

_STEP_CHANGE_KEYS = frozenset({
    'order', 'step', 'description', 'type', 'end_condition',
    'end_condition_value', 'target_type', 'target_zone',
    'target_value_low', 'target_value_high', 'repeat_count',
})


def _index_steps_by_order(workout_data: dict) -> Dict[int, dict]:
    """Map every step in the workout to its stepOrder.

    Garmin numbers stepOrder globally across the whole workout rather than
    per list: a repeat group at order 2 is followed by its own children at
    orders 3 and 4, and the next top-level step continues at 5. That makes
    stepOrder a unique address for any step, nested ones included, and it is
    the same "order" value get_workout_by_id reports.
    """
    index: Dict[int, dict] = {}
    for step, _ in _iter_workout_steps(workout_data):
        order = step.get('stepOrder')
        if order is None:
            continue
        order = int(order)
        if order in index:
            raise ValueError(
                f"Workout has duplicate stepOrder {order}; cannot address "
                f"steps by order. Use replace_workout instead."
            )
        index[order] = step
    return index


def _set_target_type(step: dict, target_key: str, path: str) -> None:
    """Set a step's target type, resolving the id Garmin treats as canonical."""
    target_id = KNOWN_TARGET_TYPE_KEYS.get(target_key)
    if target_id is None:
        known = ", ".join(sorted(KNOWN_TARGET_TYPE_KEYS))
        raise ValueError(
            f"{path}: unknown target_type {target_key!r}. Known types: {known}. "
            f"Use replace_workout to set a target type outside this list."
        )
    step['targetType'] = {
        "workoutTargetTypeId": target_id,
        "workoutTargetTypeKey": target_key,
    }
    if target_key == 'no.target':
        step['zoneNumber'] = None
        step['targetValueOne'] = None
        step['targetValueTwo'] = None


def _apply_step_change(step: dict, change: dict, path: str) -> None:
    """Apply one curated change to a single workout step, in place."""
    unknown = set(change) - _STEP_CHANGE_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown field(s) {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(_STEP_CHANGE_KEYS - {'order', 'step'}))}"
        )

    if 'description' in change:
        step['description'] = change['description']

    if 'type' in change:
        step_key = change['type']
        step_id = STEP_TYPE_IDS.get(step_key)
        if step_id is None:
            known = ", ".join(sorted(STEP_TYPE_IDS))
            raise ValueError(f"{path}: unknown type {step_key!r}. Known: {known}")
        step['stepType'] = {"stepTypeId": step_id, "stepTypeKey": step_key}

    if 'end_condition' in change:
        condition_key = change['end_condition']
        condition_id = END_CONDITION_TYPE_IDS.get(condition_key)
        if condition_id is None:
            known = ", ".join(sorted(END_CONDITION_TYPE_IDS))
            raise ValueError(
                f"{path}: unknown end_condition {condition_key!r}. Known: {known}"
            )
        step['endCondition'] = {
            "conditionTypeId": condition_id,
            "conditionTypeKey": condition_key,
        }

    if 'end_condition_value' in change:
        step['endConditionValue'] = float(change['end_condition_value'])

    if 'repeat_count' in change:
        if step.get('type') != 'RepeatGroupDTO':
            raise ValueError(
                f"{path}: repeat_count applies to a repeat group, but this "
                f"step is {step.get('type')!r}"
            )
        repeats = int(change['repeat_count'])
        if repeats < 1:
            raise ValueError(f"{path}: repeat_count must be at least 1, got {repeats}")
        # Garmin stores the count twice and honours whichever it reads first;
        # leaving them out of sync makes the watch and the web UI disagree.
        step['numberOfIterations'] = repeats
        step['endConditionValue'] = float(repeats)

    if 'target_type' in change:
        _set_target_type(step, change['target_type'], path)

    has_zone = 'target_zone' in change
    has_range = 'target_value_low' in change or 'target_value_high' in change
    if has_zone and has_range:
        raise ValueError(
            f"{path}: set either target_zone or target_value_low/high, not both. "
            f"Garmin silently discards the custom range when a zone is present."
        )

    if has_zone:
        zone = change['target_zone']
        step['zoneNumber'] = None if zone is None else int(zone)
        # A named zone and a custom range are mutually exclusive; clear the
        # range so a leftover value cannot win.
        step['targetValueOne'] = None
        step['targetValueTwo'] = None

    if has_range:
        low = change.get('target_value_low')
        high = change.get('target_value_high')
        if low is None or high is None:
            raise ValueError(
                f"{path}: target_value_low and target_value_high must be given together"
            )
        if float(low) >= float(high):
            raise ValueError(
                f"{path}: target_value_low ({low}) must be less than "
                f"target_value_high ({high})"
            )
        step['targetValueOne'] = float(low)
        step['targetValueTwo'] = float(high)
        step['zoneNumber'] = None


def _apply_workout_changes(workout_data: dict, changes: dict) -> List[str]:
    """Apply a curated change spec to a full workout DTO, in place.

    Returns a human-readable list of what changed, so the tool can report the
    edit without the caller diffing two DTOs.
    """
    known_top = {'name', 'description', 'steps'}
    unknown = set(changes) - known_top
    if unknown:
        raise ValueError(
            f"Unknown change field(s): {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(known_top))}"
        )
    if not changes:
        raise ValueError("changes is empty; nothing to update")

    applied: List[str] = []

    if 'name' in changes:
        name = changes['name']
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        workout_data['workoutName'] = name
        applied.append(f"name -> {name!r}")

    if 'description' in changes:
        workout_data['description'] = changes['description']
        applied.append("description updated")

    step_changes = changes.get('steps') or []
    if 'steps' in changes and not isinstance(step_changes, list):
        raise ValueError("steps must be a list of step change objects")

    if step_changes:
        index = _index_steps_by_order(workout_data)

        for position, change in enumerate(step_changes):
            if not isinstance(change, dict):
                raise ValueError(f"steps[{position}] must be an object")
            order = change.get('order', change.get('step'))
            if order is None:
                raise ValueError(
                    f"steps[{position}] is missing 'order' (the step's order "
                    f"from get_workout_by_id)"
                )
            try:
                order = int(order)
            except (TypeError, ValueError):
                raise ValueError(f"steps[{position}]: order must be an integer, got {order!r}")

            step = index.get(order)
            if step is None:
                available = ", ".join(str(o) for o in sorted(index))
                raise ValueError(
                    f"steps[{position}]: no step with order {order}. "
                    f"Available orders: {available}"
                )

            _apply_step_change(step, change, f"step[order={order}]")
            applied.append(f"step {order} updated")

    return applied


def _prepare_workout_payload(workout_data: dict, workout_id: int) -> dict:
    """Validate and normalize a workout DTO for a PUT to workout_id."""
    _normalize_workout_steps(workout_data)
    _validate_end_condition_steps(workout_data)
    _validate_target_type_steps(workout_data)
    # Never send a derived estimate: Garmin stores whatever it is given without
    # checking it against the steps, so any carried-over value can outlive the
    # edit that invalidated it. Absent, Garmin derives a correct one.
    for container in (workout_data, *workout_data.get('workoutSegments', [])):
        for field in _ESTIMATE_FIELDS:
            container.pop(field, None)
    # Garmin takes the id from the URL and ignores the body, but a mismatched
    # body id makes the payload confusing to read back in a log.
    workout_data['workoutId'] = workout_id
    return workout_data


def _put_workout(workout_id: int, workout_data: dict) -> None:
    """PUT a complete workout DTO, replacing the workout in place.

    Garmin answers with an empty body, so success is confirmed by re-reading
    the workout rather than by inspecting the response.
    """
    garmin_client.client.put(
        "connectapi",
        f"/workout-service/workout/{workout_id}",
        json=workout_data,
        api=True,
    )


def _resolve_editable_workout_id(workout_id: Union[int, str]) -> int:
    """Return a numeric workout id, rejecting ids that cannot be edited."""
    workout_id_str = str(workout_id).strip()
    if '-' in workout_id_str:
        raise ValueError(
            f"{workout_id_str} is a training-plan/Garmin Coach workout UUID. "
            f"Those are generated by Garmin and cannot be edited; copy it into "
            f"your own workout with upload_workout instead."
        )
    try:
        numeric_id = int(workout_id_str)
    except ValueError:
        raise ValueError(f"Invalid workout_id {workout_id!r}: expected a numeric id")
    if numeric_id <= 0:
        raise ValueError(f"Invalid workout_id {numeric_id}: must be positive")
    return numeric_id


def _curate_workout_summary(workout: dict) -> dict:
    """Extract essential workout metadata for list views"""
    sport_type = workout.get('sportType', {})

    summary = {
        "id": workout.get('workoutId'),
        "name": workout.get('workoutName'),
        "sport": sport_type.get('sportTypeKey'),
        "provider": workout.get('workoutProvider'),
        "created_date": workout.get('createdDate'),
        "updated_date": workout.get('updatedDate'),
    }

    # Add optional fields if present
    if workout.get('description'):
        summary['description'] = workout.get('description')

    if workout.get('estimatedDuration'):
        summary['estimated_duration_seconds'] = workout.get('estimatedDuration')

    if workout.get('estimatedDistance'):
        summary['estimated_distance_meters'] = workout.get('estimatedDistance')

    # Remove None values
    return {k: v for k, v in summary.items() if v is not None}


def _curate_step_target(
    curated: dict,
    step: dict,
    target_field: str,
    value_one_field: str,
    value_two_field: str,
    zone_field: str,
    prefix: str = "",
) -> None:
    """Curate a workout target block, handling Garmin null target payloads safely."""
    target_type = step.get(target_field)
    if not isinstance(target_type, dict):
        target_type = {}
    target_key = target_type.get('workoutTargetTypeKey')

    if not target_key or target_key == 'no.target':
        return

    curated[f'{prefix}target_type'] = target_key

    if step.get(value_one_field) is not None:
        curated[f'{prefix}target_value_low'] = step.get(value_one_field)
    if step.get(value_two_field) is not None:
        curated[f'{prefix}target_value_high'] = step.get(value_two_field)
    if step.get(zone_field) is not None:
        curated[f'{prefix}target_zone'] = step.get(zone_field)


def _curate_workout_step(step: dict) -> dict:
    """Extract essential workout step information"""
    step_type = step.get('stepType') or {}
    end_condition = step.get('endCondition') or {}

    curated = {
        "order": step.get('stepOrder'),
        "type": step_type.get('stepTypeKey'),  # warmup, interval, cooldown, rest, recover
    }

    # Description
    if step.get('description'):
        curated['description'] = step.get('description')

    # End condition (duration/distance/lap press)
    if end_condition.get('conditionTypeKey'):
        curated['end_condition'] = end_condition.get('conditionTypeKey')
    if step.get('endConditionValue'):
        # Value meaning depends on condition type (seconds for time, meters for distance)
        curated['end_condition_value'] = step.get('endConditionValue')

    # Primary target (heart rate, pace, power, etc.)
    _curate_step_target(
        curated,
        step,
        target_field='targetType',
        value_one_field='targetValueOne',
        value_two_field='targetValueTwo',
        zone_field='zoneNumber',
    )

    # Swim workouts often store pace prescriptions as secondary targets.
    _curate_step_target(
        curated,
        step,
        target_field='secondaryTargetType',
        value_one_field='secondaryTargetValueOne',
        value_two_field='secondaryTargetValueTwo',
        zone_field='secondaryZoneNumber',
        prefix='secondary_',
    )
    # Swim stroke / equipment / drill info (Garmin returns these as nested dicts;
    # previously dropped entirely -- strokeType/equipmentType/drillType are real
    # fields Garmin provides for swim steps, unrelated to secondaryTargetValueOne).
    stroke_type = step.get('strokeType')
    if isinstance(stroke_type, dict) and stroke_type.get('strokeTypeKey'):
        curated['stroke_type'] = stroke_type.get('strokeTypeKey')
    equipment_type = step.get('equipmentType')
    if isinstance(equipment_type, dict) and equipment_type.get('equipmentTypeKey'):
        curated['equipment_type'] = equipment_type.get('equipmentTypeKey')
    drill_type = step.get('drillType')
    if isinstance(drill_type, dict) and drill_type.get('drillTypeKey'):
        curated['drill_type'] = drill_type.get('drillTypeKey')
    # Strength training exercise info
    if step.get('category'):
        curated['category'] = step.get('category')
    if step.get('exerciseName'):
        curated['exercise_name'] = step.get('exerciseName')
    if step.get('weightValue') is not None:
        curated['weight_value'] = step.get('weightValue')
        weight_unit = step.get('weightUnit', {})
        if weight_unit and weight_unit.get('unitKey'):
            curated['weight_unit'] = weight_unit.get('unitKey')

    # Repeat info for repeat steps
    if step.get('type') == 'RepeatGroupDTO':
        curated['repeat_count'] = step.get('numberOfIterations')
        nested_steps = step.get('workoutSteps', [])
        if nested_steps:
            curated['steps'] = [_curate_workout_step(s) for s in nested_steps]
            curated['step_count'] = len(nested_steps)

    return {k: v for k, v in curated.items() if v is not None}


def _curate_workout_segment(segment: dict) -> dict:
    """Extract essential segment information including workout steps"""
    sport_type = segment.get('sportType', {})

    curated = {
        "order": segment.get('segmentOrder'),
        "sport": sport_type.get('sportTypeKey'),
    }

    # Estimated metrics
    if segment.get('estimatedDurationInSecs'):
        curated['estimated_duration_seconds'] = segment.get('estimatedDurationInSecs')
    if segment.get('estimatedDistanceInMeters'):
        curated['estimated_distance_meters'] = segment.get('estimatedDistanceInMeters')

    # Workout steps - the actual content of the segment
    steps = segment.get('workoutSteps', [])
    if steps:
        curated['steps'] = [_curate_workout_step(s) for s in steps]
        curated['step_count'] = len(steps)

    return {k: v for k, v in curated.items() if v is not None}


def _curate_workout_details(workout: dict) -> dict:
    """Extract detailed workout information with segments

    Handles both regular workouts (from get_workout_by_id) and training plan workouts
    (from fbt-adaptive endpoint) which use slightly different field names.
    """
    sport_type = workout.get('sportType') or {}

    details = {
        "id": workout.get('workoutId'),
        "uuid": workout.get('workoutUuid'),
        "name": workout.get('workoutName'),
        "sport": sport_type.get('sportTypeKey') if sport_type else None,
        "provider": workout.get('workoutProvider'),
        "created_date": workout.get('createdDate'),
        "updated_date": workout.get('updatedDate'),
    }

    # Optional fields
    if workout.get('description'):
        details['description'] = workout.get('description')

    # Handle both field name variants (regular vs training plan workouts)
    duration = workout.get('estimatedDuration') or workout.get('estimatedDurationInSecs')
    if duration:
        details['estimated_duration_seconds'] = duration

    distance = workout.get('estimatedDistance') or workout.get('estimatedDistanceInMeters')
    if distance:
        details['estimated_distance_meters'] = distance

    if workout.get('avgTrainingSpeed'):
        details['avg_training_speed_mps'] = workout.get('avgTrainingSpeed')

    # Training plan specific fields
    if workout.get('workoutPhrase'):
        details['workout_type'] = workout.get('workoutPhrase')

    if workout.get('trainingEffectLabel'):
        details['training_effect_label'] = workout.get('trainingEffectLabel')

    if workout.get('estimatedTrainingEffect'):
        details['estimated_training_effect'] = workout.get('estimatedTrainingEffect')

    # Curate segments with workout steps
    segments = workout.get('workoutSegments', [])
    if segments:
        details['segments'] = [_curate_workout_segment(seg) for seg in segments]
        details['segment_count'] = len(segments)

    # Remove None values
    return {k: v for k, v in details.items() if v is not None}


def _curate_scheduled_workout(scheduled: dict) -> dict:
    """Extract essential scheduled workout information from GraphQL response"""
    # GraphQL response has workout data at top level (not nested)
    # Completed is determined by presence of associatedActivityId
    is_completed = scheduled.get('associatedActivityId') is not None

    summary = {
        "date": scheduled.get('scheduleDate'),
        # Calendar-entry id (distinct from workout_id). Pass this to
        # unschedule_workout to remove the entry from the calendar.
        "scheduled_workout_id": scheduled.get('scheduledWorkoutId'),
        "workout_uuid": scheduled.get('workoutUuid'),
        "workout_id": scheduled.get('workoutId'),
        "training_plan_id": scheduled.get('trainingPlanId'),
        "fbt_adaptive_plan_id": scheduled.get('fbtAdaptivePlanId'),
        "tp_type": scheduled.get('tpType'),
        "name": scheduled.get('workoutName'),
        "sport": scheduled.get('workoutType'),
        "completed": is_completed,
    }

    # Training plan info
    if scheduled.get('tpPlanName'):
        summary['training_plan'] = scheduled.get('tpPlanName')

    # Workout type description (e.g., "AEROBIC_LOW_SHORTAGE_BASE", "ANAEROBIC_SPEED", "LONG_WORKOUT")
    # This describes the intent/type of the workout from Garmin Coach
    if scheduled.get('workoutPhrase'):
        summary['workout_type'] = scheduled.get('workoutPhrase')

    # Rest day and race day flags
    if scheduled.get('isRestDay'):
        summary['is_rest_day'] = True
    if scheduled.get('race'):
        summary['is_race_day'] = True

    # Optional fields
    if scheduled.get('estimatedDurationInSecs'):
        summary['estimated_duration_seconds'] = scheduled.get('estimatedDurationInSecs')

    if scheduled.get('estimatedDistanceInMeters'):
        summary['estimated_distance_meters'] = scheduled.get('estimatedDistanceInMeters')

    # If completed, include the activity ID
    if is_completed:
        summary['activity_id'] = scheduled.get('associatedActivityId')

    # Remove None values
    return {k: v for k, v in summary.items() if v is not None}


def _is_already_scheduled(workout_id: int, calendar_date: str) -> bool:
    """Return True if workout_id is already scheduled on calendar_date.

    Used to make schedule_workout / schedule_workouts idempotent. The Garmin
    schedule endpoint is not idempotent: a second POST creates a second
    calendar entry on the same day. Querying first avoids the duplicate.
    """
    try:
        _validate_date(calendar_date, "calendar_date")
        query = {
            "query": (
                f'query{{workoutScheduleSummariesScalar('
                f'startDate:"{calendar_date}", endDate:"{calendar_date}")}}'
            )
        }
        result = garmin_client.query_garmin_graphql(query) or {}
        existing = (
            result.get("data", {}).get("workoutScheduleSummariesScalar", []) or []
        )
        for entry in existing:
            if (
                entry.get("workoutId") == workout_id
                and entry.get("scheduleDate") == calendar_date
            ):
                return True
    except Exception:
        # If the pre-check itself fails, fall through to the normal POST
        # path so we don't block a legitimate scheduling attempt.
        return False
    return False


def _get_garmin_coach_workouts(calendar_date: str) -> str:
    """Return curated workouts from the active Garmin Coach/training plan."""
    _validate_date(calendar_date, "calendar_date")
    query = {
        "query": (
            f'query{{trainingPlanScalar(calendarDate:"{calendar_date}", '
            f'lang:"en-US", firstDayOfWeek:"monday")}}'
        )
    }
    result = garmin_client.query_garmin_graphql(query)

    if not isinstance(result, dict) or not isinstance(result.get("data"), dict):
        return "No training plan data found or error querying data."

    plan_data = result["data"].get("trainingPlanScalar") or {}
    if not isinstance(plan_data, dict):
        return "No training plan data found or error querying data."

    training_plans = plan_data.get("trainingPlanWorkoutScheduleDTOS") or []
    if not isinstance(training_plans, list) or not training_plans:
        return f"No training plan workouts scheduled for {calendar_date}."

    all_workouts = []
    plan_names = []
    plans = []
    valid_plan_count = 0
    for plan in training_plans:
        if not isinstance(plan, dict):
            continue
        valid_plan_count += 1

        plan_name = plan.get("planName")
        if plan_name and plan_name not in plan_names:
            plan_names.append(plan_name)

        plan_details = plan.get("trainingPlanDetailsDTO")
        if not isinstance(plan_details, dict):
            plan_details = {}
        plan_summary = {
            "name": plan_name,
            "training_plan_id": plan.get("trainingPlanId"),
            "classification": plan.get("trainingPlanClassification"),
            "training_type": plan_details.get("trainingType"),
        }
        plan_summary = {
            key: value for key, value in plan_summary.items()
            if value is not None
        }
        if plan_summary:
            plans.append(plan_summary)

        workout_summaries = plan.get("workoutScheduleSummaries") or []
        if not isinstance(workout_summaries, list):
            continue
        all_workouts.extend(
            _curate_scheduled_workout(workout)
            for workout in workout_summaries
            if isinstance(workout, dict)
        )

    if valid_plan_count == 0:
        return f"No training plan workouts scheduled for {calendar_date}."

    curated = {
        "date": calendar_date,
        "training_plans": plan_names if plan_names else None,
        "plans": plans if plans else None,
        "count": len(all_workouts),
        "workouts": all_workouts,
    }
    return json.dumps(
        {key: value for key, value in curated.items() if value is not None},
        indent=2,
    )


def register_tools(app):
    """Register all workout-related tools with the MCP server app"""

    @app.tool()
    async def get_workouts() -> str:
        """Get all workouts with curated summary list

        Returns a count and list of workout summaries with essential metadata only.
        For detailed workout information including segments, use get_workout_by_id.
        """
        try:
            workouts = garmin_client.get_workouts()
            if not workouts:
                return "No workouts found."

            # Curate the workout list
            curated = {
                "count": len(workouts),
                "workouts": [_curate_workout_summary(w) for w in workouts]
            }

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving workouts: {str(e)}"

    @app.tool()
    async def get_workout_by_id(workout_id: Union[int, str]) -> str:
        """Get detailed information for a specific workout

        Returns workout details including segments and step structure.

        Accepts either:
        - Numeric workout ID (from get_workouts, get_scheduled_workouts, or
          training-plan families that expose workout_id)
        - Workout UUID (from adaptive Garmin Coach/training-plan workouts)

        Rest-day UUIDs can resolve to a minimal record without a workout name
        or segments.

        Args:
            workout_id: Workout ID (numeric) or UUID (for training plan workouts)
        """
        try:
            workout_id_str = str(workout_id)
            # Detect if this is a UUID (contains dashes) or numeric ID
            is_uuid = '-' in workout_id_str

            if is_uuid:
                # Training plan / Garmin Coach workout - use fbt-adaptive endpoint
                url = f"workout-service/fbt-adaptive/{workout_id_str}"
                workout = garmin_client.connectapi(url)
            else:
                # Regular workout - use standard endpoint
                workout = garmin_client.get_workout_by_id(int(workout_id_str))

            if not workout:
                return f"No workout found with ID {workout_id_str}."

            # Return curated details with segments
            curated = _curate_workout_details(workout)
            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving workout: {str(e)}"

    @app.tool()
    async def download_workout(workout_id: int) -> str:
        """Download a workout as a FIT file

        Downloads the workout in FIT format. The binary data cannot be returned
        directly through the MCP interface, but this confirms the workout is available.

        Args:
            workout_id: ID of the workout to download
        """
        try:
            workout_data = garmin_client.download_workout(workout_id)
            if not workout_data:
                return f"No workout data found for workout with ID {workout_id}."

            # Return information about the download
            data_size = len(workout_data) if isinstance(workout_data, (bytes, bytearray)) else 0
            return json.dumps({
                "workout_id": workout_id,
                "format": "FIT",
                "size_bytes": data_size,
                "message": "Workout data is available in FIT format. Use Garmin Connect API to save to file."
            }, indent=2)
        except Exception as e:
            return f"Error downloading workout: {str(e)}"

    @app.tool()
    async def upload_workout(workout_data: dict) -> str:
        """Upload a workout from JSON data

        Creates a new workout in Garmin Connect from structured workout data.

        IMPORTANT: Step types must use Garmin's DTO format:
        - Use "ExecutableStepDTO" for regular steps (warmup, interval, cooldown, recovery)
        - Use "RepeatGroupDTO" for repeat/interval groups with numberOfIterations.
          Always include endCondition with conditionTypeId 7 and conditionTypeKey
          "iterations"; omitting conditionTypeId causes the API to silently corrupt
          the repeat count.

        IMPORTANT: Heart rate targets come in two forms:
        - Named zone (e.g. Zone 2): set targetType to "heart.rate.zone" and use "zoneNumber" (1-5).
          Do NOT put the zone number in targetValueOne.
        - Custom HR range (e.g. 105-143 bpm): set targetType to "heart.rate.zone" and use
          "targetValueOne" (low bpm) / "targetValueTwo" (high bpm). Do NOT set "zoneNumber".
          This matches Garmin Connect's "Custom" heart rate target.
        For non-HR targets (pace, power, cadence), use targetValueOne/targetValueTwo directly.
        Target values are fields on the workout step, alongside targetType; do not put
        targetValueOne, targetValueTwo, or zoneNumber inside the targetType object.
        Use either zoneNumber or targetValueOne/targetValueTwo, not both. Garmin silently
        discards a custom range when a named zone is also present.

        Note: a safety check converts targetValueOne 1-5 to zoneNumber when zoneNumber is missing,
        to catch the common mistake of putting a zone index in targetValueOne. Typical bpm values
        (e.g. 105, 143) are not affected.

        IMPORTANT: Target type IDs and keys must match Garmin's canonical mapping.
        Garmin treats workoutTargetTypeId as authoritative, so mismatches are rejected
        before upload.  Known mappings:
        - workoutTargetTypeId 1  -> "no.target"
        - workoutTargetTypeId 2  -> "power.zone"  (cycling power zone 1-7, use zoneNumber)
        - workoutTargetTypeId 4  -> "heart.rate.zone"
        - workoutTargetTypeId 6  -> "pace.zone" (running/swim) OR "power.between" (cycling)

        IMPORTANT: For cycling power targets use the correct target type:
        - Power zone (zone 1-7 based on FTP %): use workoutTargetTypeId 2, key "power.zone",
          and "zoneNumber" (1-7).
        - Absolute watt range (e.g. 200-250 W): use workoutTargetTypeId 6, key "power.between",
          and "targetValueOne" (low watts) / "targetValueTwo" (high watts).
        Using workoutTargetTypeId 2 with key "power.between" is a silent Garmin bug: the
        workout uploads but Garmin stores it as "power.zone" and the intent is lost.

        Use {"workoutTargetTypeId": 4, "workoutTargetTypeKey": "heart.rate.zone"} with
        targetValueOne/targetValueTwo for custom heart-rate ranges.

        IMPORTANT: Sport type IDs for workouts (different from activity API!):
        - 1 = running, 2 = cycling, 5 = strength_training, 6 = cardio, 11 = walking

        IMPORTANT: End condition IDs and keys must match Garmin's canonical mapping.
        Garmin treats conditionTypeId as authoritative, so mismatches such as
        {"conditionTypeId": 4, "conditionTypeKey": "heart.rate"} are rejected before
        upload because Garmin would interpret them as "calories". Use
        {"conditionTypeId": 6, "conditionTypeKey": "heart.rate"} for heart-rate
        end conditions.

        **Available Templates:**
        Instead of building workout JSON from scratch, you can use these MCP resources as starting points:
        - workout://templates/simple-run - Basic warmup/run/cooldown structure
        - workout://templates/interval-running - Interval training with repeat groups
        - workout://templates/tempo-run - Tempo run with heart rate zone targets
        - workout://templates/strength-circuit - Strength training with exercises, reps, rest
        - workout://reference/structure - Complete JSON structure reference with all fields

        Access these resources using your MCP client's resource reading capability, modify the template
        as needed, and pass the resulting JSON as the workout_data parameter.

        **Strength training workouts** require these additional fields on each exercise step:
        - "category": exercise category (e.g. "BENCH_PRESS", "PULL_UP", "CURL", "SHOULDER_PRESS",
          "ROW", "SQUAT", "DEADLIFT", "TRICEPS_EXTENSION", "PLANK", "LUNGE", "CARDIO")
        - "exerciseName": specific exercise (e.g. "BARBELL_BENCH_PRESS", "PULL_UP",
          "DUMBBELL_BICEPS_CURL", "DUMBBELL_SHOULDER_PRESS", "BENT_OVER_ROW_WITH_DUMBELL",
          "BODY_WEIGHT_DIP", "BARBELL_SQUAT", "BARBELL_DEADLIFT")
        - "weightValue" (optional): weight as number (e.g. 24.0)
        - "weightUnit" (optional): {"unitId": 8, "unitKey": "kilogram", "factor": 1000.0}
        Use endCondition reps (conditionTypeId: 10) for exercises, rest (stepTypeId: 5) between sets.

        Example strength exercise step:
        {
            "type": "ExecutableStepDTO",
            "stepOrder": 1,
            "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
            "endCondition": {"conditionTypeId": 10, "conditionTypeKey": "reps"},
            "endConditionValue": 10.0,
            "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target"},
            "category": "BENCH_PRESS",
            "exerciseName": "BARBELL_BENCH_PRESS",
            "weightValue": 60.0,
            "weightUnit": {"unitId": 8, "unitKey": "kilogram", "factor": 1000.0}
        }

        Example running workout with HR zone target:
        {
            "workoutName": "My Workout",
            "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
            "workoutSegments": [{
                "segmentOrder": 1,
                "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
                "workoutSteps": [{
                    "type": "ExecutableStepDTO",
                    "stepOrder": 1,
                    "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                    "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                    "endConditionValue": 1200.0,
                    "targetType": {"workoutTargetTypeId": 4, "workoutTargetTypeKey": "heart.rate.zone"},
                    "zoneNumber": 3
                }]
            }]
        }

        Args:
            workout_data: Dictionary containing workout structure (name, sport type, segments, etc.)
        """
        try:
            _normalize_workout_steps(workout_data)
            _validate_end_condition_steps(workout_data)
            _validate_target_type_steps(workout_data)

            # Pass dict directly - library handles conversion
            result = garmin_client.upload_workout(workout_data)

            # Curate the response
            if isinstance(result, dict):
                curated = {
                    "status": "success",
                    "workout_id": result.get('workoutId'),
                    "name": result.get('workoutName'),
                    "message": "Workout uploaded successfully"
                }
                # Remove None values
                curated = {k: v for k, v in curated.items() if v is not None}
                return json.dumps(curated, indent=2)

            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error uploading workout: {str(e)}"

    @app.tool()
    async def upload_workouts(workouts: list[dict]) -> str:
        """Upload multiple workouts from JSON data in a single call

        Creates multiple new workouts in Garmin Connect. Each item in the list
        uses the same structure as upload_workout.

        IMPORTANT: Step types must use Garmin's DTO format:
        - Use "ExecutableStepDTO" for regular steps (warmup, interval, cooldown, recovery)
        - Use "RepeatGroupDTO" for repeat/interval groups with numberOfIterations.
          Always include endCondition with conditionTypeId 7 and conditionTypeKey
          "iterations"; omitting conditionTypeId causes the API to silently corrupt
          the repeat count.

        IMPORTANT: For named heart rate zone targets, use "zoneNumber" (1-5), NOT targetValueOne/targetValueTwo.
        For custom heart-rate ranges, use targetType {"workoutTargetTypeId": 4,
        "workoutTargetTypeKey": "heart.rate.zone"} with targetValueOne/targetValueTwo.
        Target values belong on the workout step, alongside targetType, not inside it.
        For cycling power zone targets (zone-based), use workoutTargetTypeId 2, key "power.zone".
        For cycling absolute watt range targets, use workoutTargetTypeId 6, key "power.between",
        with targetValueOne (low watts) and targetValueTwo (high watts).
        Target type IDs and keys must match Garmin's canonical mapping.

        IMPORTANT: End condition IDs and keys must match Garmin's canonical mapping.
        Garmin treats conditionTypeId as authoritative, so mismatches are rejected before upload.

        Args:
            workouts: List of workout dictionaries, each containing workout structure
                      (name, sport type, segments, etc.) — same format as upload_workout.
        """
        results = []
        for workout_data in workouts:
            try:
                _normalize_workout_steps(workout_data)
                _validate_end_condition_steps(workout_data)
                _validate_target_type_steps(workout_data)
                result = garmin_client.upload_workout(workout_data)
                if isinstance(result, dict):
                    entry = {
                        "status": "success",
                        "workout_id": result.get('workoutId'),
                        "name": result.get('workoutName'),
                        "message": "Workout uploaded successfully"
                    }
                    results.append({k: v for k, v in entry.items() if v is not None})
                else:
                    results.append({"status": "success", "message": "Workout uploaded successfully"})
            except Exception as e:
                results.append({
                    "status": "error",
                    "name": workout_data.get('workoutName'),
                    "message": f"Error uploading workout: {str(e)}"
                })

        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results
        }, indent=2)

    @app.tool()
    async def update_workout(workout_id: Union[int, str], changes: dict) -> str:
        """Edit an existing workout in place, keeping its ID and calendar entries

        Use this instead of delete + upload_workout when changing a workout that
        already exists. The workout keeps its ID, so any calendar entry already
        pointing at it stays valid and picks up the new content. Deleting and
        re-uploading mints a new ID and leaves the old calendar entry orphaned.

        The workout is read, patched, and written back, so only the fields you
        name change; everything else is preserved exactly as Garmin has it.

        Steps are addressed by "order" — the same value get_workout_by_id
        reports for each step. Garmin numbers stepOrder globally across the
        whole workout, so nested steps inside a repeat group have their own
        unique orders (a repeat group at order 2 is followed by its children at
        orders 3 and 4). Call get_workout_by_id first to see the orders.

        This edits your own workouts. Garmin Coach / training-plan workouts are
        identified by UUID and are generated by Garmin; they cannot be edited.

        To add, remove, or reorder steps, use replace_workout — this tool edits
        existing steps only.

        changes accepts:
            name (str): new workout name
            description (str): new workout description
            steps (list): per-step edits, each an object with:
                order (int, required): the step's order from get_workout_by_id
                description (str): step description
                type (str): warmup, cooldown, interval, recovery, rest, repeat
                end_condition (str): time, distance, lap.button, calories,
                    heart.rate, iterations, reps, ...
                end_condition_value (number): seconds for time, meters for
                    distance, count for reps
                target_type (str): no.target, heart.rate.zone, power.zone,
                    pace.zone, power.between
                target_zone (int): named zone number (HR 1-5, power 1-7).
                    Setting this clears any custom range.
                target_value_low / target_value_high (number): custom range,
                    e.g. 105/143 bpm or 200/250 watts. Must be given together,
                    and they clear any named zone.
                repeat_count (int): iterations, for a repeat group step only

        target_zone and target_value_low/high are mutually exclusive — Garmin
        silently discards a custom range when a named zone is also present, so
        passing both is rejected here.

        Note: the workout itself updates immediately, but Garmin's calendar
        summary caches the duration it recorded when the workout was scheduled.
        get_scheduled_workouts may keep reporting the old
        estimated_duration_seconds for an edited workout. The workout content
        the watch receives is correct.

        Examples:
            Rename and stretch the main interval to 30 minutes:
            update_workout(1234567890, {
                "name": "Tempo 30min",
                "steps": [{"order": 2, "end_condition_value": 1800}]
            })

            Change a repeat group to 6 reps and drop the recovery to Z1:
            update_workout(1234567890, {
                "steps": [
                    {"order": 2, "repeat_count": 6},
                    {"order": 4, "target_zone": 1}
                ]
            })

            Swap a named zone for an exact bpm range:
            update_workout(1234567890, {
                "steps": [{"order": 3, "target_value_low": 136,
                           "target_value_high": 148}]
            })

        Args:
            workout_id: Numeric ID of the workout to edit (from get_workouts)
            changes: Change spec as described above
        """
        try:
            numeric_id = _resolve_editable_workout_id(workout_id)
            workout = garmin_client.get_workout_by_id(numeric_id)
            if not workout:
                return json.dumps({
                    "status": "failed",
                    "workout_id": numeric_id,
                    "message": f"No workout found with ID {numeric_id}",
                }, indent=2)

            applied = _apply_workout_changes(workout, changes)
            _prepare_workout_payload(workout, numeric_id)
            _put_workout(numeric_id, workout)

            # Garmin's PUT returns an empty body, so read the workout back to
            # report what it actually stored rather than what we sent.
            updated = garmin_client.get_workout_by_id(numeric_id)
            return json.dumps({
                "status": "success",
                "workout_id": numeric_id,
                "applied": applied,
                "message": (
                    f"Workout {numeric_id} updated in place; ID and any "
                    f"calendar entries preserved"
                ),
                "workout": _curate_workout_details(updated or {}),
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "failed",
                "workout_id": workout_id,
                "message": f"Error updating workout: {str(e)}",
            }, indent=2)

    @app.tool()
    async def update_workouts(updates: list[dict]) -> str:
        """Edit multiple existing workouts in place in a single call

        Each workout keeps its ID and calendar entries. See update_workout for
        the full change spec, step addressing rules, and caveats.

        Each update is applied independently: one failure does not stop the
        others, and every result is reported.

        Args:
            updates: List of objects, each with:
                - workout_id (int): numeric ID of the workout to edit
                - changes (dict): change spec, same format as update_workout

        Example:
            [{"workout_id": 123, "changes": {"name": "Week 1 Tempo"}},
             {"workout_id": 456, "changes": {
                 "steps": [{"order": 2, "end_condition_value": 2400}]}}]
        """
        results = []
        for position, update in enumerate(updates):
            workout_id = update.get("workout_id") if isinstance(update, dict) else None
            try:
                if not isinstance(update, dict):
                    raise ValueError(f"updates[{position}] must be an object")
                if workout_id is None:
                    raise ValueError(f"updates[{position}] is missing workout_id")
                changes = update.get("changes")
                if not isinstance(changes, dict):
                    raise ValueError(
                        f"updates[{position}] is missing a 'changes' object"
                    )

                numeric_id = _resolve_editable_workout_id(workout_id)
                workout = garmin_client.get_workout_by_id(numeric_id)
                if not workout:
                    raise ValueError(f"No workout found with ID {numeric_id}")

                applied = _apply_workout_changes(workout, changes)
                _prepare_workout_payload(workout, numeric_id)
                _put_workout(numeric_id, workout)

                updated = garmin_client.get_workout_by_id(numeric_id)
                results.append({
                    "status": "success",
                    "workout_id": numeric_id,
                    "name": (updated or {}).get("workoutName"),
                    "applied": applied,
                    "message": f"Workout {numeric_id} updated in place",
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "workout_id": workout_id,
                    "message": f"Error updating workout: {str(e)}",
                })

        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results,
        }, indent=2)

    @app.tool()
    async def replace_workout(workout_id: Union[int, str], workout_data: dict) -> str:
        """Replace an existing workout's entire content, keeping its ID

        Overwrites the workout with the structure you provide, while keeping the
        workout ID — so calendar entries pointing at it stay valid and follow
        the new content. Use this when the step list itself changes: adding,
        removing, or reordering steps, or switching sport type.

        For editing values on existing steps (durations, targets, names), prefer
        update_workout — it patches the workout Garmin already has instead of
        requiring you to restate the whole thing.

        workout_data uses exactly the same structure as upload_workout, and the
        same rules apply for step DTO types, end-condition IDs, target-type IDs,
        and heart-rate zone versus custom range. Anything you omit is dropped
        from the workout, so pass the complete intended structure.

        Note: Garmin's calendar summary caches the duration recorded when the
        workout was scheduled, so get_scheduled_workouts may keep reporting the
        old estimated_duration_seconds. The workout content is correct.

        Args:
            workout_id: Numeric ID of the workout to overwrite (from get_workouts)
            workout_data: Complete workout structure, same format as upload_workout
        """
        try:
            numeric_id = _resolve_editable_workout_id(workout_id)
            if not isinstance(workout_data, dict) or not workout_data.get('workoutSegments'):
                raise ValueError(
                    "workout_data must be a complete workout including "
                    "workoutSegments; Garmin rejects a partial body"
                )

            _prepare_workout_payload(workout_data, numeric_id)
            _put_workout(numeric_id, workout_data)

            updated = garmin_client.get_workout_by_id(numeric_id)
            return json.dumps({
                "status": "success",
                "workout_id": numeric_id,
                "message": (
                    f"Workout {numeric_id} replaced in place; ID and any "
                    f"calendar entries preserved"
                ),
                "workout": _curate_workout_details(updated or {}),
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "failed",
                "workout_id": workout_id,
                "message": f"Error replacing workout: {str(e)}",
            }, indent=2)

    @app.tool()
    async def delete_workout(workout_id: int) -> str:
        """Delete a workout from Garmin Connect

        Permanently removes a workout from your Garmin Connect workout library.

        Args:
            workout_id: ID of the workout to delete (get IDs from get_workouts)
        """
        try:
            # Use the high-level garminconnect method. In garminconnect 0.3.2,
            # client.delete(..., api=True) returns resp.json() (a dict), not a
            # Response, so checking response.status_code raises AttributeError.
            # Delegate to the library and rely on exceptions to signal failure.
            garmin_client.delete_workout(workout_id)
            return json.dumps({
                "status": "success",
                "workout_id": workout_id,
                "message": f"Workout {workout_id} deleted successfully"
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "failed",
                "workout_id": workout_id,
                "message": f"Failed to delete workout: {str(e)}"
            }, indent=2)

    @app.tool()
    async def delete_workouts(workout_ids: list[int]) -> str:
        """Delete multiple workouts from Garmin Connect in a single call

        Permanently removes multiple workouts from your Garmin Connect workout library.

        Args:
            workout_ids: List of workout IDs to delete (get IDs from get_workouts)
        """
        results = []
        for workout_id in workout_ids:
            try:
                # See note in delete_workout: high-level call avoids the
                # garminconnect 0.3.2 dict-vs-Response trap.
                garmin_client.delete_workout(workout_id)
                results.append({
                    "status": "success",
                    "workout_id": workout_id,
                    "message": f"Workout {workout_id} deleted successfully"
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "workout_id": workout_id,
                    "message": f"Error deleting workout: {str(e)}"
                })

        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results
        }, indent=2)

    @app.tool()
    async def get_scheduled_workouts(start_date: str, end_date: str) -> str:
        """Get scheduled workouts between two dates with curated summary list

        Returns workouts that have been scheduled on the Garmin Connect calendar,
        including their scheduled dates and completion status.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        try:
            _validate_date(start_date, "start_date")
            _validate_date(end_date, "end_date")
            # Query for scheduled workouts using GraphQL
            query = {
                "query": f'query{{workoutScheduleSummariesScalar(startDate:"{start_date}", endDate:"{end_date}")}}'
            }
            result = garmin_client.query_garmin_graphql(query)

            if not result or "data" not in result:
                return "No scheduled workouts found or error querying data."

            scheduled = result.get("data", {}).get("workoutScheduleSummariesScalar", [])

            if not scheduled:
                return f"No workouts scheduled between {start_date} and {end_date}."

            # Curate the scheduled workout list
            curated = {
                "count": len(scheduled),
                "date_range": {"start": start_date, "end": end_date},
                "scheduled_workouts": [_curate_scheduled_workout(s) for s in scheduled]
            }

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving scheduled workouts: {str(e)}"

    @app.tool()
    async def get_garmin_coach_workouts(calendar_date: str) -> str:
        """Get Garmin Coach workouts around the given date

        Returns workouts from the active Garmin Coach/training plan, including
        plan metadata, workout identifiers, dates, sport, duration, completion
        status, rest days, race days, and workout intent when Garmin provides
        them. Adaptive plans expose only Garmin's currently generated window,
        typically the current week; future dates may return no workouts even
        while a plan is active. The count includes rest-day entries.

        Garmin's standalone Daily Suggested Workouts are generated on compatible
        devices. As of July 31, 2026, no supported or known Garmin Connect
        web/API endpoint, including those exposed by this project's
        python-garminconnect dependency, returns the device's upcoming DSW
        schedule. This tool returns Garmin Coach/training-plan workouts and does
        not synthesize device-generated suggestions.

        This is the preferred tool for Garmin Coach requests. The legacy
        get_training_plan_workouts tool returns the same data; do not call both.

        Adaptive Coach plans typically expose workout_uuid; other plan families
        may expose numeric workout_id. Pass whichever identifier is present to
        get_workout_by_id. Rest-day UUIDs may return minimal detail without
        workout segments.

        Args:
            calendar_date: Reference date in YYYY-MM-DD format (returns week's workouts)
        """
        try:
            return _get_garmin_coach_workouts(calendar_date)
        except Exception as e:
            return f"Error retrieving Garmin Coach workouts: {str(e)}"

    @app.tool()
    async def get_training_plan_workouts(calendar_date: str) -> str:
        """Compatibility alias for get_garmin_coach_workouts

        Prefer get_garmin_coach_workouts for new requests. This legacy tool
        returns the same Garmin Coach/training-plan data; do not call both for
        one request. Adaptive plans expose only Garmin's currently generated
        window, typically the current week; future dates may return no workouts
        even while a plan is active.

        Adaptive training plans typically expose workout_uuid; other plan
        families may expose numeric workout_id. Pass whichever identifier is
        present to get_workout_by_id. The returned count includes rest days.

        Args:
            calendar_date: Reference date in YYYY-MM-DD format (returns week's workouts)
        """
        try:
            return _get_garmin_coach_workouts(calendar_date)
        except Exception as e:
            return f"Error retrieving training plan workouts: {str(e)}"

    @app.tool()
    async def schedule_workout(workout_id: int, calendar_date: str) -> str:
        """Schedule a workout to a specific calendar date

        This adds an existing workout from your Garmin workout library
        to your Garmin Connect calendar on the specified date.

        Idempotent: if the workout is already scheduled for that date, this
        is a no-op that reports success without creating a duplicate entry.

        Args:
            workout_id: ID of the workout to schedule (get IDs from get_workouts)
            calendar_date: Date to schedule the workout in YYYY-MM-DD format
        """
        try:
            _validate_date(calendar_date, "calendar_date")
        except ValueError as e:
            return json.dumps({
                "status": "failed",
                "workout_id": workout_id,
                "scheduled_date": calendar_date,
                "message": str(e),
            }, indent=2)

        try:
            if _is_already_scheduled(workout_id, calendar_date):
                return json.dumps({
                    "status": "success",
                    "workout_id": workout_id,
                    "scheduled_date": calendar_date,
                    "idempotent": True,
                    "message": (
                        f"Workout {workout_id} already scheduled for "
                        f"{calendar_date} — no action taken"
                    )
                }, indent=2)

            url = f"workout-service/schedule/{workout_id}"
            response = garmin_client.client.post("connectapi", url, json={"date": calendar_date})

            if response.status_code == 200:
                return json.dumps({
                    "status": "success",
                    "workout_id": workout_id,
                    "scheduled_date": calendar_date,
                    "message": f"Successfully scheduled workout {workout_id} for {calendar_date}"
                }, indent=2)
            else:
                return json.dumps({
                    "status": "failed",
                    "workout_id": workout_id,
                    "scheduled_date": calendar_date,
                    "http_status": response.status_code,
                    "message": f"Failed to schedule workout: HTTP {response.status_code}"
                }, indent=2)
        except Exception as e:
            return f"Error scheduling workout: {str(e)}"

    @app.tool()
    async def schedule_workouts(schedules: list[dict]) -> str:
        """Schedule multiple workouts to specific calendar dates

        This adds workouts to your Garmin Connect calendar in a single call.
        Each item can either reference an existing workout by ID, or provide
        inline workout_data to upload-and-schedule in one step.

        Args:
            schedules: List of workout schedules, each with:
                - calendar_date (str): Date to schedule the workout in YYYY-MM-DD format (required)
                - workout_id (int): ID of an existing workout to schedule (required unless workout_data is provided)
                - workout_data (dict): Inline workout JSON to upload first, then schedule (optional).
                  When provided, workout_id is not required. Uses the same structure and
                  target-value rules as upload_workout.

        Examples:
            Schedule existing workouts by ID:
            [{"workout_id": 123456, "calendar_date": "2024-01-15"},
             {"workout_id": 789012, "calendar_date": "2024-01-17"}]

            Upload and schedule inline:
            [{"calendar_date": "2024-01-15", "workout_data": {"workoutName": "Easy Run", ...}},
             {"workout_id": 789012, "calendar_date": "2024-01-17"}]
        """
        results = []
        for item in schedules:
            workout_id = item.get("workout_id")
            calendar_date = item.get("calendar_date")
            workout_data = item.get("workout_data")

            if calendar_date is None:
                results.append({
                    "status": "failed",
                    "workout_id": workout_id,
                    "scheduled_date": calendar_date,
                    "message": "Missing required field: calendar_date"
                })
                continue

            try:
                _validate_date(calendar_date, "calendar_date")
            except ValueError as e:
                results.append({
                    "status": "failed",
                    "workout_id": workout_id,
                    "scheduled_date": calendar_date,
                    "message": str(e),
                })
                continue

            if workout_id is None and workout_data is None:
                results.append({
                    "status": "failed",
                    "workout_id": None,
                    "scheduled_date": calendar_date,
                    "message": "Missing required fields: provide either workout_id or workout_data"
                })
                continue

            try:
                workout_name = None

                if workout_data is not None:
                    # Upload the workout first, then use the returned ID to schedule
                    _normalize_workout_steps(workout_data)
                    _validate_end_condition_steps(workout_data)
                    _validate_target_type_steps(workout_data)
                    upload_result = garmin_client.upload_workout(workout_data)
                    if not isinstance(upload_result, dict) or upload_result.get('workoutId') is None:
                        results.append({
                            "status": "failed",
                            "scheduled_date": calendar_date,
                            "message": "Upload succeeded but no workout_id returned"
                        })
                        continue
                    workout_id = upload_result['workoutId']
                    workout_name = upload_result.get('workoutName')

                if _is_already_scheduled(workout_id, calendar_date):
                    entry = {
                        "status": "success",
                        "workout_id": workout_id,
                        "scheduled_date": calendar_date,
                        "idempotent": True,
                        "message": (
                            f"Workout {workout_id} already scheduled for "
                            f"{calendar_date} — no action taken"
                        )
                    }
                    if workout_name:
                        entry["workout_name"] = workout_name
                    results.append(entry)
                    continue

                url = f"workout-service/schedule/{workout_id}"
                response = garmin_client.client.post("connectapi", url, json={"date": calendar_date})

                if response.status_code == 200:
                    entry = {
                        "status": "success",
                        "workout_id": workout_id,
                        "scheduled_date": calendar_date,
                        "message": f"Successfully scheduled workout {workout_id} for {calendar_date}"
                    }
                    if workout_name:
                        entry["workout_name"] = workout_name
                    results.append(entry)
                else:
                    results.append({
                        "status": "failed",
                        "workout_id": workout_id,
                        "scheduled_date": calendar_date,
                        "http_status": response.status_code,
                        "message": f"Failed to schedule workout: HTTP {response.status_code}"
                    })
            except Exception as e:
                results.append({
                    "status": "error",
                    "workout_id": workout_id,
                    "scheduled_date": calendar_date,
                    "message": f"Error scheduling workout: {str(e)}"
                })

        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results
        }, indent=2)

    @app.tool()
    async def unschedule_workout(scheduled_workout_id: int) -> str:
        """Remove a scheduled workout from the Garmin Connect calendar

        Deletes a calendar entry without deleting the underlying workout
        template — the workout stays in your library and can be re-scheduled.

        IMPORTANT: scheduled_workout_id is the calendar-entry id, which is
        different from the workout's id. Get it from get_scheduled_workouts
        (the "scheduled_workout_id" field), not from get_workouts.

        Note: the scheduled-workouts listing is an eventually-consistent index.
        If you just scheduled this workout, allow a moment before unscheduling
        so the id is available.

        Args:
            scheduled_workout_id: Calendar-entry id from get_scheduled_workouts
        """
        try:
            # Delegate to the high-level garminconnect method. Its client.delete
            # returns a dict ({}), not a Response, so we rely on exceptions to
            # signal failure rather than checking a status code — same pattern
            # as delete_workout.
            garmin_client.unschedule_workout(scheduled_workout_id)
            return json.dumps({
                "status": "success",
                "scheduled_workout_id": scheduled_workout_id,
                "message": f"Scheduled workout {scheduled_workout_id} removed from calendar"
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "failed",
                "scheduled_workout_id": scheduled_workout_id,
                "message": f"Failed to unschedule workout: {str(e)}"
            }, indent=2)

    @app.tool()
    async def unschedule_workouts(scheduled_workout_ids: list[int]) -> str:
        """Remove multiple scheduled workouts from the Garmin Connect calendar

        Deletes multiple calendar entries in a single call. The underlying
        workout templates are left intact in your library.

        IMPORTANT: each id is a calendar-entry id (the "scheduled_workout_id"
        field from get_scheduled_workouts), not a workout id.

        Args:
            scheduled_workout_ids: List of calendar-entry ids from get_scheduled_workouts
        """
        results = []
        for scheduled_workout_id in scheduled_workout_ids:
            try:
                # See note in unschedule_workout: high-level call returns a dict,
                # so rely on exceptions to signal failure.
                garmin_client.unschedule_workout(scheduled_workout_id)
                results.append({
                    "status": "success",
                    "scheduled_workout_id": scheduled_workout_id,
                    "message": f"Scheduled workout {scheduled_workout_id} removed from calendar"
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "scheduled_workout_id": scheduled_workout_id,
                    "message": f"Error unscheduling workout: {str(e)}"
                })

        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results
        }, indent=2)

    return app
