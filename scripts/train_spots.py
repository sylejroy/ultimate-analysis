#!/usr/bin/env python3
"""Train teams that choose where to go and where to throw, from nothing (simulation/spots.py),
by reinforcement (PPO).

Each iteration the present team plays points against the scripted team (without set
plays: the base game first) or itself, a
random wind each point. Every choice its players make is kept: what was around each place
they could have chosen, which they chose, how likely that was, and the value (on the game
as it is) at that moment. A choice's reward: what the team and the player got
(ppo.rewards_of and ppo.player_rewards: completions and the yards each gained,
turnovers, scores, the stall counted; a throw caught, a catch, a score, a block, a disc
lost) from then until their next choice, discounted as time goes; its advantage, how
much better it went than the value said (GAE over the player's choices). The choices
then move towards what went better, never far from what played (clipped, PPO).

Printed per iteration: points scored and conceded, completions, the reward per point,
the time. Every --report iterations the team plays
16 points against the scripted team and is kept in a new folder:

    <output>/<run>/iter_0010/spots.pt ... and latest/

    python scripts/train_spots.py --iterations 300 --workers 12
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.simulation import game  # noqa: E402
from ultimate_analysis.simulation.match import play_point  # noqa: E402
from ultimate_analysis.simulation.ppo import (  # noqa: E402
    STALL_REWARD,
    player_rewards,
    rewards_of,
    stall_watch,
)
from ultimate_analysis.simulation.scripted import ScriptedTeam, ScriptedTeam2  # noqa: E402
from ultimate_analysis.simulation.spots import (  # noqa: E402
    SpotBrains,
    SpotTeam,
    load_spot_brains,
    save_spot_brains,
)

# Per tenth of a second: rewards count for some 50 s on (at 0.99, 10 s, a stall 7 s off
# counted half a turnover now, and holding on looked better than throwing)
GAMMA, LAMBDA = 0.998, 0.95
CLIP = 0.2
EPOCHS, BATCH = 3, 1024
ENTROPY = 0.01  # Keeps them trying other choices: near how they choose at first

# ---------------------------------------------------------------------- playing (workers)


def _brains(state) -> SpotBrains:
    torch.set_num_threads(1)
    brains = SpotBrains()
    brains.load(state)
    return brains


def play(job) -> list:
    """Play points with the present team (its state) against the scripted team, itself,
    or another team of spots (its state, a dict: not learned from). Per point: the
    choices made, the rewards, and what came of it."""
    state, opponent, seeds, limit, stall_reward, stalled, *more = job
    record = more[0] if more else True  # False: only to see how it plays
    brains = _brains(state)
    other = _brains(opponent) if isinstance(opponent, dict) else None
    found = []
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        side = k % 2
        ours = SpotTeam(brains, side, rng, record=record)
        if other is not None:
            theirs = SpotTeam(other, 1 - side, rng, record=False)
        elif opponent == "self":
            theirs = SpotTeam(brains, 1 - side, rng, record=record)
        else:
            # The base game first: no set plays
            kind = ScriptedTeam2 if opponent == "scripted2" else ScriptedTeam
            theirs = kind(seed, plays=False)
        teams = (ours, theirs) if side == 0 else (theirs, ours)
        start = game.pull_start(rng, int(rng.integers(2)), game.a_wind(rng))
        counted, watch = stall_watch()
        point = play_point(teams, rng, start, watch=watch, limit=limit)
        steps = max(ours.step, 1)
        kinds = [e for e in point.events if e.team == side]
        found.append(
            {
                "choices": ours.choices + (theirs.choices if isinstance(theirs, SpotTeam) else []),
                "rewards": rewards_of(point, steps, game.STEP, counted, stall_reward, stalled),
                "own": player_rewards(point, steps, game.STEP),
                "side": side,
                "scored": int(point.scored == side),
                "conceded": int(point.scored == 1 - side),
                "throws": sum(e.kind == "throw" for e in kinds),
                "catches": sum(e.kind == "catch" for e in kinds),
                # The other team's passes: how well this one defends
                "their_throws": sum(e.kind == "throw" and e.team != side for e in point.events),
                "their_catches": sum(e.kind == "catch" and e.team != side for e in point.events),
            }
        )
    return found


# ---------------------------------------------------------------------- learning


def advantages(points: list, credit: float, gamma: float = GAMMA) -> list:
    """Each choice with its advantage, its GAE return and what happened after it (the
    rewards to the end of the point, discounted)."""
    kept = []
    for point in points:
        by_player = defaultdict(list)
        for choice in point["choices"]:
            by_player[(choice.team, choice.player)].append(choice)
        for (team, player), choices in by_player.items():
            choices.sort(key=lambda c: c.step)
            rewards = point["rewards"][team] + credit * point["own"][player + team * game.PLAYERS]
            discounted = np.zeros(len(rewards) + 1)  # From each step to the end
            for t in range(len(rewards) - 1, -1, -1):
                discounted[t] = rewards[t] + gamma * discounted[t + 1]
            running = 0.0
            for i in range(len(choices) - 1, -1, -1):
                c = choices[i]
                start = min(c.step, len(rewards))
                end = (
                    min(choices[i + 1].step, len(rewards)) if i + 1 < len(choices) else len(rewards)
                )
                gap = end - start
                between = discounted[start] - gamma**gap * discounted[end]
                after = choices[i + 1].value if i + 1 < len(choices) else 0.0
                delta = between + gamma**gap * after - c.value
                # Faded once per choice, not per step: choices far apart keep their credit
                running = delta + gamma**gap * LAMBDA * running
                c.advantage, c.returned, c.happened = running, running + c.value, discounted[start]
                kept.append(c)
    return kept


def learn(brains, choices, optimisers, value_optimisers, values_only=False, device="cpu") -> None:
    """PPO steps on the choices of each kind, and the values towards what happened; on
    `device` (the networks are there)."""
    for side in ("offence", "defence", "throw"):
        mine = [c for c in choices if c.side == side]
        if not mine:
            continue
        network = brains.throws if side == "throw" else brains.spots[side]
        value_side = "offence" if side == "throw" else side
        features = torch.from_numpy(np.stack([c.features for c in mine])).to(device)  # float16
        # Added to the scores: the throws' lean at first, the spots that may not be chosen
        prior = torch.from_numpy(np.stack([c.prior for c in mine])).to(device)
        chosen = torch.tensor([c.chosen for c in mine], device=device)
        old = torch.tensor([c.log_chance for c in mine], dtype=torch.float32, device=device)
        advantage = torch.tensor([c.advantage for c in mine], dtype=torch.float32, device=device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        seen = torch.from_numpy(np.stack([c.value_inputs for c in mine])).to(device)
        happened = torch.tensor([c.happened for c in mine], dtype=torch.float32, device=device)
        for _ in range(EPOCHS):
            order = torch.randperm(len(mine), device=device)
            for start in range(0, len(order), BATCH):
                batch = order[start : start + BATCH]
                value_loss = (
                    (brains.values[value_side](seen[batch])[:, 0] - happened[batch]) ** 2
                ).mean()
                value_optimisers[value_side].zero_grad()
                value_loss.backward()
                value_optimisers[value_side].step()
                if values_only:
                    continue
                scores = network(features[batch].float())[..., 0]
                scores = scores + prior[batch]
                log_chances = torch.log_softmax(scores, dim=1)
                new = log_chances[torch.arange(len(batch), device=device), chosen[batch]]
                ratio = torch.exp(new - old[batch])
                a = advantage[batch]
                loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
                # Kept trying other choices as it does at first (the leans), not all alike
                at_first = torch.log_softmax(prior[batch], dim=1)
                apart = (log_chances.exp() * (log_chances - at_first)).sum(1).mean()
                loss = loss + ENTROPY * apart
                optimisers[side].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimisers[side].step()


# ---------------------------------------------------------------------- the loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=Path, default=None, help="A run's folder to go on from")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--points", type=int, default=36, help="Per iteration")
    parser.add_argument("--point-seconds", type=float, default=70.0)
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--value-rate", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=0, help="Iterations of values only")
    parser.add_argument("--credit", type=float, default=1.0)
    parser.add_argument(
        "--gamma", type=float, default=GAMMA, help="How rewards fade, per tenth of a second"
    )
    parser.add_argument("--stall-reward", type=float, default=STALL_REWARD)
    parser.add_argument(
        "--stalled",
        type=float,
        default=3.0,
        help="What a stall costs the team besides the turnover (else holding on to the end "
        "beats throwing for a team that cannot yet complete a throw)",
    )
    parser.add_argument("--self-share", type=float, default=0.0)
    parser.add_argument("--report", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/models/movement"))
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Where the networks learn (the points are played on the processor)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    brains = (load_spot_brains(args.start) if args.start else SpotBrains()).to(args.device)
    optimisers = {
        "offence": torch.optim.Adam(brains.spots["offence"].parameters(), lr=args.rate),
        "defence": torch.optim.Adam(brains.spots["defence"].parameters(), lr=args.rate),
        "throw": torch.optim.Adam(brains.throws.parameters(), lr=args.rate),
    }
    value_optimisers = {
        side: torch.optim.Adam(brains.values[side].parameters(), lr=args.value_rate)
        for side in ("offence", "defence")
    }
    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_spots"
    run.mkdir(parents=True)
    settings = {**vars(args), "start": str(args.start), "output": str(args.output)}
    (run / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"Written to {run}", flush=True)
    with Pool(args.workers) as pool:
        for iteration in range(1, args.iterations + 1):
            began = time.perf_counter()
            state = brains.state()
            share = max(1, args.points // args.workers)
            jobs = [
                (
                    state,
                    "self" if rng.random() < args.self_share else "scripted",
                    [int(s) for s in rng.integers(0, 2**31, share)],
                    args.point_seconds,
                    args.stall_reward,
                    args.stalled,
                )
                for _ in range(args.workers)
            ]
            points = [p for found in pool.map(play, jobs) for p in found]
            played = time.perf_counter() - began
            choices = advantages(points, args.credit, args.gamma)
            learn(
                brains,
                choices,
                optimisers,
                value_optimisers,
                iteration <= args.warmup,
                args.device,
            )
            throws = sum(p["throws"] for p in points)
            print(
                f"iteration {iteration}: {len(points)} points, scored "
                f"{sum(p['scored'] for p in points)}, conceded {sum(p['conceded'] for p in points)}, "
                f"completed {sum(p['catches'] for p in points)}/{throws}, "
                f"{len(choices)} choices, "
                f"{played:.0f} s playing, {time.perf_counter() - began - played:.0f} s learning"
                + (" (values only)" if iteration <= args.warmup else ""),
                flush=True,
            )
            if iteration % args.report == 0:
                folder = run / f"iter_{iteration:04d}"
                save_spot_brains(brains, folder, iteration=iteration)
                save_spot_brains(brains, run / "latest", iteration=iteration)
                seeds = [int(s) for s in rng.integers(0, 2**31, 16)]
                found = pool.apply(play, ((brains.state(), "scripted", seeds, 120.0, 0.0, 0.0),))
                scored = sum(p["scored"] for p in found)
                conceded = sum(p["conceded"] for p in found)
                (folder / "against_scripted.json").write_text(
                    json.dumps({"scored": scored, "conceded": conceded})
                )
                print(
                    f"  against scripted: scored {scored}, conceded {conceded} of 16 points; "
                    f"completed {sum(p['catches'] for p in found)}/{sum(p['throws'] for p in found)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
