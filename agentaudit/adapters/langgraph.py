"""Convert a LangGraph run into an AgentAudit Trace.

Works on the events produced by ``graph.stream(..., stream_mode="updates")``:
a list of ``{node_name: state_update}`` dicts. Accepts both live LangChain
message objects and their serialized dict form (``message.model_dump()`` /
recorded JSON), so a recorded run replays identically to a live one.
"""

from __future__ import annotations

from typing import Any, Optional

from ..trace import Step, Trace

_SYNTHETIC_NODES = {"__start__", "__end__", "__interrupt__"}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _messages_of(update: Any) -> list:
    messages = _get(update, "messages")
    if messages is None:
        return []
    if not isinstance(messages, list):
        messages = [messages]
    return messages


def from_langgraph(
    events: list[dict],
    workflow: str = "langgraph_run",
    task: str = "",
    final_output: Optional[str] = None,
    expected_output: Optional[str] = None,
) -> Trace:
    """Build a Trace from LangGraph ``stream_mode="updates"`` events.

    Each graph node becomes an agent; each node update becomes a step whose
    output is the text of the messages the node emitted. A step's input is the
    previous step's output (the task for the first step), matching how state
    flows through the graph.
    """
    steps: list[Step] = []
    prev_output = task
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"expected update dict, got {type(event).__name__}")
        for node, update in event.items():
            if node in _SYNTHETIC_NODES:
                continue
            tokens_in = tokens_out = 0
            model = "unknown"
            texts = []
            for msg in _messages_of(update):
                text = _content_text(_get(msg, "content"))
                if text:
                    texts.append(text)
                usage = _get(msg, "usage_metadata") or {}
                tokens_in += int(_get(usage, "input_tokens", 0) or 0)
                tokens_out += int(_get(usage, "output_tokens", 0) or 0)
                meta = _get(msg, "response_metadata") or {}
                model = _get(meta, "model_name") or _get(meta, "model") or model
            if texts:
                output = "\n".join(texts)
            else:
                # Node with a custom (non-messages) state schema: record the
                # raw update so the step still carries auditable content.
                output = str(update) if update else ""
            steps.append(
                Step(
                    id=f"s{len(steps) + 1}",
                    agent=node,
                    input=prev_output,
                    output=output,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            )
            prev_output = output
    return Trace(
        workflow=workflow,
        steps=steps,
        final_output=final_output if final_output is not None else (steps[-1].output if steps else ""),
        expected_output=expected_output,
        task=task,
    )
