"""Wrap Agent.run main-loop block in try/finally for PLODDER_BACKBONE_FOR_RUN."""
from __future__ import annotations

from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "mini_devin" / "orchestrator" / "agent.py"


def main() -> None:
    raw = PATH.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)

    start = None
    for i, line in enumerate(lines):
        if line == "        # Main agent loop\n" or line.rstrip("\n") == "        # Main agent loop":
            start = i
            break
    if start is None:
        raise SystemExit("start marker not found")

    end = None
    for i in range(start, len(lines)):
        if lines[i].rstrip("\n") == "        return task":
            end = i
            break
    if end is None:
        raise SystemExit("return task not found")

    head = lines[:start]
    # Ensure inject block is not duplicated
    tail_join = "".join(head[-12:])
    if "PLODDER_BACKBONE_FOR_RUN" in tail_join:
        print("already wrapped; skip")
        return

    inject = (
        "        _bb_for_run = os.environ.get(\"PLODDER_BACKBONE_FOR_RUN\", \"\").strip().lower() in (\n"
        '            "1", "true", "yes", "on",\n'
        "        )\n"
        "        if _bb_for_run:\n"
        "            await self._async_setup_backbone()\n"
        "            self._use_backbone_dispatch = True\n"
        "\n"
        "        try:\n"
    )

    body = []
    for line in lines[start : end + 1]:
        if line.endswith("\n"):
            core = line[:-1]
        else:
            core = line
        body.append("    " + core + "\n")

    footer = (
        "        finally:\n"
        "            if locals().get(\"_bb_for_run\"):\n"
        "                await self._async_teardown_backbone()\n"
    )

    new_lines = head + [inject] + body + [footer] + lines[end + 1 :]
    PATH.write_text("".join(new_lines), encoding="utf-8")
    print(f"wrapped lines {start+1}-{end+1}")


if __name__ == "__main__":
    main()
