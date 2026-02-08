"""Tests for slipcore.intent -- Factorized intent model: Force + Object."""

from __future__ import annotations

import pytest

from slipcore import (
    CORE_OBJECTS,
    ForceToken,
    Intent,
    ObjectToken,
    V2_TO_V3,
    from_v2_mnemonic,
    resolve_intent,
)


# ===================================================================
# ForceToken enum
# ===================================================================


class TestForceToken:
    """Tests for the ForceToken enumeration."""

    EXPECTED_MEMBERS: list[str] = [
        "OBSERVE",
        "INFORM",
        "ASK",
        "REQUEST",
        "PROPOSE",
        "COMMIT",
        "EVAL",
        "META",
        "ACCEPT",
        "REJECT",
        "ERROR",
        "FALLBACK",
    ]

    def test_all_12_members_exist(self) -> None:
        assert len(ForceToken) == 12

    @pytest.mark.parametrize("name", EXPECTED_MEMBERS)
    def test_member_exists(self, name: str) -> None:
        assert hasattr(ForceToken, name)
        member = ForceToken[name]
        assert isinstance(member, ForceToken)

    @pytest.mark.parametrize("token", list(ForceToken))
    def test_action_coord_in_range(self, token: ForceToken) -> None:
        coord = token.action_coord
        assert isinstance(coord, int)
        assert 0 <= coord <= 7, f"{token.name}.action_coord = {coord}, out of 0-7"

    def test_specific_action_coords(self) -> None:
        assert ForceToken.OBSERVE.action_coord == 0
        assert ForceToken.INFORM.action_coord == 1
        assert ForceToken.ASK.action_coord == 2
        assert ForceToken.REQUEST.action_coord == 3
        assert ForceToken.PROPOSE.action_coord == 4
        assert ForceToken.COMMIT.action_coord == 5
        assert ForceToken.EVAL.action_coord == 6
        assert ForceToken.META.action_coord == 7

    def test_wire_values_are_camel_case(self) -> None:
        for token in ForceToken:
            value = token.value
            assert value[0].isupper(), f"{token.name}.value should start uppercase"
            assert value.isalpha(), f"{token.name}.value should be alphabetic"


# ===================================================================
# CORE_OBJECTS registry
# ===================================================================


class TestCoreObjects:
    """Tests for the CORE_OBJECTS registry."""

    def test_at_least_30_objects_registered(self) -> None:
        assert len(CORE_OBJECTS) >= 30

    EXPECTED_OBJECTS: list[str] = [
        "Plan",
        "Task",
        "Review",
        "State",
        "Change",
        "Error",
        "Result",
        "Status",
        "Complete",
        "Blocked",
        "Progress",
        "Clarify",
        "Permission",
        "Resource",
        "Help",
        "Cancel",
        "Priority",
        "Alternative",
        "Rollback",
        "Deadline",
        "Approve",
        "NeedsWork",
        "Ack",
        "Sync",
        "Handoff",
        "Escalate",
        "Abort",
        "Condition",
        "Defer",
        "Timeout",
        "Validation",
        "Generic",
    ]

    @pytest.mark.parametrize("mnemonic", EXPECTED_OBJECTS)
    def test_specific_object_exists(self, mnemonic: str) -> None:
        assert mnemonic in CORE_OBJECTS, f"{mnemonic!r} not in CORE_OBJECTS"

    @pytest.mark.parametrize("mnemonic", EXPECTED_OBJECTS)
    def test_object_is_object_token(self, mnemonic: str) -> None:
        obj = CORE_OBJECTS[mnemonic]
        assert isinstance(obj, ObjectToken)

    def test_all_objects_are_core(self) -> None:
        for mnemonic, obj in CORE_OBJECTS.items():
            assert obj.is_core is True, f"{mnemonic} should be core"


# ===================================================================
# ObjectToken dataclass
# ===================================================================


class TestObjectToken:
    """Tests for the ObjectToken frozen dataclass."""

    def test_frozen(self) -> None:
        obj = CORE_OBJECTS["Task"]
        with pytest.raises(AttributeError):
            obj.mnemonic = "Other"  # type: ignore[misc]

    def test_fields(self) -> None:
        obj = CORE_OBJECTS["Plan"]
        assert isinstance(obj.mnemonic, str)
        assert isinstance(obj.canonical, str)
        assert isinstance(obj.domain_coord, int)
        assert isinstance(obj.polarity_coord, int)
        assert isinstance(obj.urgency_coord, int)
        assert isinstance(obj.is_core, bool)

    @pytest.mark.parametrize("mnemonic", list(CORE_OBJECTS.keys()))
    def test_coords_in_valid_range(self, mnemonic: str) -> None:
        obj = CORE_OBJECTS[mnemonic]
        assert 0 <= obj.domain_coord <= 7, f"{mnemonic} domain_coord out of range"
        assert 0 <= obj.polarity_coord <= 7, f"{mnemonic} polarity_coord out of range"
        assert 0 <= obj.urgency_coord <= 7, f"{mnemonic} urgency_coord out of range"


# ===================================================================
# Intent dataclass
# ===================================================================


class TestIntent:
    """Tests for the Intent dataclass."""

    def test_force_plus_obj_combine(self) -> None:
        force = ForceToken.REQUEST
        obj = CORE_OBJECTS["Plan"]
        intent = Intent(force=force, obj=obj)
        assert intent.force is ForceToken.REQUEST
        assert intent.obj is obj

    def test_mnemonic_property(self) -> None:
        intent = Intent(force=ForceToken.REQUEST, obj=CORE_OBJECTS["Review"])
        assert intent.mnemonic == "RequestReview"

    def test_mnemonic_property_all_forces(self) -> None:
        obj = CORE_OBJECTS["Task"]
        for token in ForceToken:
            intent = Intent(force=token, obj=obj)
            assert intent.mnemonic == f"{token.value}Task"

    def test_coords_property(self) -> None:
        intent = Intent(force=ForceToken.REQUEST, obj=CORE_OBJECTS["Plan"])
        coords = intent.coords
        assert isinstance(coords, tuple)
        assert len(coords) == 4
        action, polarity, domain, urgency = coords
        # action comes from ForceToken.REQUEST.action_coord == 3
        assert action == 3
        # polarity, domain, urgency come from Plan object
        plan = CORE_OBJECTS["Plan"]
        assert polarity == plan.polarity_coord
        assert domain == plan.domain_coord
        assert urgency == plan.urgency_coord

    def test_coords_all_in_range(self) -> None:
        intent = Intent(force=ForceToken.OBSERVE, obj=CORE_OBJECTS["State"])
        for c in intent.coords:
            assert 0 <= c <= 7

    def test_wire_tokens_property(self) -> None:
        intent = Intent(force=ForceToken.EVAL, obj=CORE_OBJECTS["Approve"])
        force_str, obj_str = intent.wire_tokens
        assert force_str == "Eval"
        assert obj_str == "Approve"

    def test_wire_tokens_for_all_forces(self) -> None:
        obj = CORE_OBJECTS["Generic"]
        for token in ForceToken:
            intent = Intent(force=token, obj=obj)
            assert intent.wire_tokens == (token.value, "Generic")

    def test_frozen(self) -> None:
        intent = Intent(force=ForceToken.REQUEST, obj=CORE_OBJECTS["Task"])
        with pytest.raises(AttributeError):
            intent.force = ForceToken.INFORM  # type: ignore[misc]


# ===================================================================
# resolve_intent
# ===================================================================


class TestResolveIntent:
    """Tests for resolve_intent()."""

    VALID_PAIRS: list[tuple[str, str]] = [
        ("Request", "Plan"),
        ("Observe", "State"),
        ("Inform", "Result"),
        ("Ask", "Clarify"),
        ("Propose", "Alternative"),
        ("Commit", "Task"),
        ("Eval", "Approve"),
        ("Meta", "Ack"),
        ("Accept", "Generic"),
        ("Reject", "Generic"),
        ("Error", "Timeout"),
        ("Fallback", "Generic"),
    ]

    @pytest.mark.parametrize("force,obj", VALID_PAIRS)
    def test_valid_pairs(self, force: str, obj: str) -> None:
        intent = resolve_intent(force, obj)
        assert isinstance(intent, Intent)
        assert intent.force.value == force
        assert intent.obj.mnemonic == obj

    def test_invalid_force_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown force"):
            resolve_intent("Bogus", "Plan")

    def test_invalid_obj_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown object"):
            resolve_intent("Request", "NonExistent")

    def test_both_invalid_raises_on_force_first(self) -> None:
        with pytest.raises(ValueError, match="Unknown force"):
            resolve_intent("Bogus", "NonExistent")

    def test_empty_strings_raise(self) -> None:
        with pytest.raises(ValueError):
            resolve_intent("", "")


# ===================================================================
# V2_TO_V3 migration mapping
# ===================================================================


class TestV2ToV3:
    """Tests for the V2_TO_V3 migration dictionary."""

    def test_all_46_v2_mnemonics(self) -> None:
        assert len(V2_TO_V3) == 46

    def test_all_values_are_string_pairs(self) -> None:
        for mnemonic, pair in V2_TO_V3.items():
            assert isinstance(pair, tuple), f"{mnemonic}: expected tuple"
            assert len(pair) == 2, f"{mnemonic}: expected 2-tuple"
            force, obj = pair
            assert isinstance(force, str)
            assert isinstance(obj, str)

    SPECIFIC_MAPPINGS: list[tuple[str, str, str]] = [
        ("ObserveState", "Observe", "State"),
        ("InformResult", "Inform", "Result"),
        ("InformComplete", "Inform", "Complete"),
        ("AskClarify", "Ask", "Clarify"),
        ("RequestTask", "Request", "Task"),
        ("RequestPlan", "Request", "Plan"),
        ("RequestReview", "Request", "Review"),
        ("RequestHelp", "Request", "Help"),
        ("ProposePlan", "Propose", "Plan"),
        ("CommitTask", "Commit", "Task"),
        ("EvalApprove", "Eval", "Approve"),
        ("EvalNeedsWork", "Eval", "NeedsWork"),
        ("MetaAck", "Meta", "Ack"),
        ("MetaSync", "Meta", "Sync"),
        ("MetaEscalate", "Meta", "Escalate"),
        ("MetaAbort", "Meta", "Abort"),
        ("Accept", "Accept", "Generic"),
        ("Reject", "Reject", "Generic"),
        ("AcceptWithCondition", "Accept", "Condition"),
        ("Defer", "Meta", "Defer"),
        ("ErrorGeneric", "Error", "Generic"),
        ("ErrorTimeout", "Error", "Timeout"),
        ("Fallback", "Fallback", "Generic"),
    ]

    @pytest.mark.parametrize("v2,expected_force,expected_obj", SPECIFIC_MAPPINGS)
    def test_specific_mapping(
        self, v2: str, expected_force: str, expected_obj: str
    ) -> None:
        force, obj = V2_TO_V3[v2]
        assert force == expected_force
        assert obj == expected_obj

    def test_all_mapped_forces_are_valid(self) -> None:
        from slipcore.intent import FORCE_VALUES

        for mnemonic, (force, _obj) in V2_TO_V3.items():
            assert force in FORCE_VALUES, f"{mnemonic} maps to invalid force {force!r}"

    def test_all_mapped_objects_are_registered(self) -> None:
        for mnemonic, (_force, obj) in V2_TO_V3.items():
            assert obj in CORE_OBJECTS, (
                f"{mnemonic} maps to unregistered object {obj!r}"
            )


# ===================================================================
# from_v2_mnemonic
# ===================================================================


class TestFromV2Mnemonic:
    """Tests for from_v2_mnemonic()."""

    KNOWN_V2: list[tuple[str, str, str]] = [
        ("ObserveState", "Observe", "State"),
        ("RequestReview", "Request", "Review"),
        ("EvalApprove", "Eval", "Approve"),
        ("MetaAbort", "Meta", "Abort"),
        ("Fallback", "Fallback", "Generic"),
        ("AcceptWithCondition", "Accept", "Condition"),
        ("ErrorTimeout", "Error", "Timeout"),
    ]

    @pytest.mark.parametrize("v2,expected_force,expected_obj", KNOWN_V2)
    def test_known_mnemonics_resolve(
        self, v2: str, expected_force: str, expected_obj: str
    ) -> None:
        intent = from_v2_mnemonic(v2)
        assert isinstance(intent, Intent)
        assert intent.force.value == expected_force
        assert intent.obj.mnemonic == expected_obj

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown v2 mnemonic"):
            from_v2_mnemonic("TotallyFake")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown v2 mnemonic"):
            from_v2_mnemonic("")

    def test_all_v2_mnemonics_produce_valid_intents(self) -> None:
        for mnemonic in V2_TO_V3:
            intent = from_v2_mnemonic(mnemonic)
            assert isinstance(intent, Intent)
            # The returned intent should have valid wire tokens
            force_str, obj_str = intent.wire_tokens
            assert force_str == V2_TO_V3[mnemonic][0]
            assert obj_str == V2_TO_V3[mnemonic][1]
