#!/usr/bin/env python3
"""Run two agents against each other N times and report win counts."""

import json
import os
import subprocess
import sys
import glob

NUM_GAMES = 30
MATCH_DIR = "3600-agents/matches"

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <agent_a> <agent_b>")
    sys.exit(1)

agent_a = sys.argv[1]
agent_b = sys.argv[2]

print(f"Running {agent_a} vs {agent_b} for {NUM_GAMES} games...\n")

# Clean old match files for these agents so we only pick up new ones
for f in glob.glob(os.path.join(MATCH_DIR, "*.json")):
    try:
        with open(f) as fh:
            data = json.load(fh)
        if data.get("player_a") == agent_a and data.get("player_b") == agent_b:
            os.remove(f)
    except Exception:
        pass

wins_a = 0
wins_b = 0
ties = 0

for i in range(NUM_GAMES):
    result = subprocess.run(
        ["python3", "engine/run_local_agents.py", agent_a, agent_b],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Find the newest match JSON
    files = glob.glob(os.path.join(MATCH_DIR, "*.json"))
    if not files:
        print(f"  Game {i + 1}: no match file found")
        continue

    latest = max(files, key=os.path.getmtime)
    try:
        with open(latest) as fh:
            data = json.load(fh)
    except Exception:
        print(f"  Game {i + 1}: failed to parse {latest}")
        continue

    r = data.get("result", -1)
    reason = data.get("reason", "")
    a_pts = data.get("a_points", [])[-1] if data.get("a_points") else 0
    b_pts = data.get("b_points", [])[-1] if data.get("b_points") else 0

    if r == 0:
        wins_a += 1
    elif r == 1:
        wins_b += 1
    else:
        ties += 1

    os.remove(latest)

    if (i + 1) % 5 == 0:
        print(
            f"  Games {i + 1}/{NUM_GAMES} done  |  {agent_a}: {wins_a}  {agent_b}: {wins_b}  Ties: {ties}"
        )

print(f"\n{'=' * 50}")
print(f"Final: {agent_a}: {wins_a} wins  |  {agent_b}: {wins_b} wins  |  Ties: {ties}")
if wins_a > wins_b:
    print(f"Winner: {agent_a}")
elif wins_b > wins_a:
    print(f"Winner: {agent_b}")
else:
    print("Result: Tie")
print(f"{'=' * 50}")
