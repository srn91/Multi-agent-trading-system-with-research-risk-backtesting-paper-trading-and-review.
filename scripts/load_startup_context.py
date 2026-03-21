#!/usr/bin/env python3
"""
Startup context loader.
Reads primer.md, lessons.md, and session_state.json at session start.
Prints a summary so you know where you left off.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_primer():
    path = ROOT / "primer.md"
    if path.exists():
        print("=" * 60)
        print("  PRIMER — Where We Left Off")
        print("=" * 60)
        print(path.read_text())
    else:
        print("No primer.md found.")


def load_lessons():
    path = ROOT / "tasks" / "lessons.md"
    if path.exists():
        print("\n" + "=" * 60)
        print("  LESSONS — Rules to Follow")
        print("=" * 60)
        content = path.read_text()
        # Print just the section headers for quick scan
        for line in content.split("\n"):
            if line.startswith("## "):
                print(f"  {line}")
        print(f"\n  (Full lessons at tasks/lessons.md)")
    else:
        print("No tasks/lessons.md found.")


def load_state():
    path = ROOT / "state" / "session_state.json"
    if path.exists():
        print("\n" + "=" * 60)
        print("  SESSION STATE")
        print("=" * 60)
        state = json.loads(path.read_text())
        print(f"  Phase: {state.get('project_phase', '?')}")
        print(f"  Focus: {state.get('current_focus', '?')}")
        print(f"  Next tasks:")
        for task in state.get("next_tasks", [])[:5]:
            print(f"    → {task}")
        print(f"  Last updated: {state.get('last_updated', '?')}")
    else:
        print("No session_state.json found.")


if __name__ == "__main__":
    load_primer()
    load_lessons()
    load_state()
    print("\n" + "=" * 60)
    print("  Ready to work.")
    print("=" * 60)
