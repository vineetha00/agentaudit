import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from agentaudit.judge import OpenAICompatJudge, get_judge, parse_judge_json
from agentaudit.trace import Step
from generate_traces import generate


def test_parse_judge_json_tolerates_fences_and_chatter():
    assert parse_judge_json('```json\n{"defective": true, "rationale": "x"}\n```') == {
        "defective": True,
        "rationale": "x",
    }
    assert parse_judge_json('Sure! Here is the JSON: {"score": 4, "rationale": "ok"} hope that helps')["score"] == 4


def test_generator_is_deterministic(tmp_path):
    a = generate(tmp_path / "a", seed=7, n_variants=2)
    b = generate(tmp_path / "b", seed=7, n_variants=2)
    assert a == b
    assert len(a) == 30  # 3 scenarios x 5 families x 2 variants
    for tid, lab in a.items():
        trace = json.loads((tmp_path / "a" / "traces" / f"{tid}.json").read_text())
        step_ids = [s["id"] for s in trace["steps"]]
        assert lab["origin_step_id"] in step_ids
        # defect labels are monotone: clean ... clean, defective ... defective
        flags = [lab["step_labels"][sid] for sid in step_ids]
        assert flags == sorted(flags)
        assert any(flags)


def test_openai_compat_judge_check_defect(monkeypatch):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"defective": true, "rationale": "asserts the wrong total"}'}}]}

    import requests

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)
    judge = OpenAICompatJudge(model="llama3.1")
    verdict = judge.check_defect(Step(id="s1", agent="analyst", input="", output="total is 12.2"), "total wrong")
    assert verdict.defective is True
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["body"]["model"] == "llama3.1"
    assert captured["body"]["temperature"] == 0


def test_get_judge_openai_compat():
    judge = get_judge("openai-compat", model="llama3.1", base_url="http://example.com/v1")
    assert isinstance(judge, OpenAICompatJudge)
    assert judge.base_url == "http://example.com/v1"
    with pytest.raises(ValueError):
        get_judge("nope")
