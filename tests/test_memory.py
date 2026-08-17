"""Tests for the deterministic memory-recall trigger and result compaction.

The trigger exists because the fast controller model measurably fails to call
memory on personal questions. If these tests regress, the agent starts looking
like it has amnesia -- so the failing cases from that benchmark are pinned
here as explicit test cases.
"""

import json

from voice_agent.memory import compact_recall, looks_personal


class TestLooksPersonal:
    def test_measured_model_failures_are_caught(self):
        """The exact utterances the controller failed to recall on."""
        assert looks_personal("What's my sister's name again?")
        assert looks_personal("What did we decide about the memory architecture?")

    def test_continuity_references(self):
        for text in [
            "Remind me what we decided about the pipeline.",
            "Do you remember the name of that restaurant?",
            "What did I say about the deadline?",
            "Last week we talked about the latency budget.",
            "We agreed on Kokoro, right?",
            "What's my calendar look like?",
        ]:
            assert looks_personal(text), f"should trigger recall: {text!r}"

    def test_general_knowledge_does_not_trigger(self):
        for text in [
            "What's the capital of France?",
            "Explain recursion to a kid.",
            "Who won the race this weekend?",
            "Set a timer for five minutes.",
            "Thanks, that's helpful.",
        ]:
            assert not looks_personal(text), f"should not trigger recall: {text!r}"

    def test_empty_and_noise_are_safe(self):
        assert not looks_personal("")
        assert not looks_personal("   ")
        assert not looks_personal("um")


class TestCompactRecall:
    def _payload(self, *texts):
        return json.dumps(
            {
                "results": [
                    {
                        "id": f"id-{i}",
                        "text": t,
                        "scores": {"final": 0.31, "semantic": 0.73, "keyword": 0.2},
                        "metadata": {},
                        "tags": [],
                    }
                    for i, t in enumerate(texts)
                ]
            }
        )

    def test_strips_scoring_metadata(self):
        out = compact_recall(self._payload("Memory uses Hindsight."), 1000)
        assert out == "- Memory uses Hindsight."
        assert "scores" not in out and "0.31" not in out

    def test_strips_provenance_suffix(self):
        raw = self._payload("Uses the default bank | When: Monday | Involving: user")
        assert compact_recall(raw, 1000) == "- Uses the default bank"

    def test_deduplicates_repeated_facts(self):
        # Hindsight returns the same fact as both an observation and a world
        # fact, which would otherwise be spoken twice.
        raw = self._payload("Same fact.", "Same fact.", "Different fact.")
        assert compact_recall(raw, 1000) == "- Same fact.\n- Different fact."

    def test_deduplicates_near_identical_facts(self):
        """Observed live: the same fact with and without a trailing period."""
        raw = self._payload(
            "Memory architecture uses Hindsight MCP with default bank.",
            "Memory architecture uses Hindsight MCP with default bank",
            "Unmute rejected because it is CUDA only",
        )
        out = compact_recall(raw, 1000)
        assert out.count("Hindsight MCP") == 1
        assert "Unmute rejected" in out

    def test_respects_character_budget(self):
        raw = self._payload("a" * 60, "b" * 60, "c" * 60)
        out = compact_recall(raw, 100)
        assert len(out) < 150
        assert "c" * 60 not in out

    def test_malformed_payload_does_not_raise(self):
        assert compact_recall("not json at all", 50) == "not json at all"
        assert compact_recall(json.dumps({"unexpected": 1}), 50) == ""

    def test_skips_empty_facts(self):
        assert compact_recall(self._payload("", "  ", "Real."), 1000) == "- Real."
