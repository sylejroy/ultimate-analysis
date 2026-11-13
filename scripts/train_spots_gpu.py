#!/usr/bin/env python3
"""Train teams that choose where to go and where to throw (simulation/spots.py) from
nothing, playing themselves thousands of points at once on the graphics card
(simulation/batch.py), by reinforcement (PPO).

Each iteration every point plays on for --steps tenths of a second (a point that ends
starts again with a new pull). Every choice is kept: the view it was made from, which
spot or throw, how likely it was, and the value (on the game as it is). Its reward:
what the team and the player got (as scripts/train_spots.py: completions and the yards
each gained, turnovers, scores, the stall counted and a stall's extra cost; a throw
caught, a catch, a score, a block, a disc lost) until the player's next choice; its
advantage, how much better that went than the value said (GAE). The choices then move
towards what went better (clipped, PPO), what is around each spot worked out again from
the views kept.

With --marking, in that share of the points one team defends by rules, man to man
(batch.BatchPlayers): the offence learns to beat a defender before self-play. With
--league, in that share one team is an earlier team: the run's start and its
--league-size latest kept teams, not learned from, so that it does not only learn to
beat itself as it is. With --spread, in that share one team attacks by rules, for the
defence; with --only, only one of the three kinds of choice learns (--only defence: the
defenders learn to stop the attackers as they are).

Printed per iteration: points ended and scored, throws and completions, the time. Every
--report iterations the team is kept
(<run>/iter_0010/spots.pt ... and latest/) and plays --check points against the scripted
team on the processor (as scripts/train_spots.py), and the benchmark one (scripted 2,
set as the recorded teams play), while learning goes on; and, as a
ladder, as many points against the team of --ladder iterations before (each) and the
team the run started from: whether self-play makes it better than it was. The best team
against the benchmark so far (over three checks: one alone is too noisy) is named in
<run>/best.json; with --revert, a team that does that many points worse against it than
the best goes back to the best (its learning afresh), so that a bad stretch does not
lose a good team.

    python scripts/train_spots_gpu.py --points 2048 --iterations 500 --marking 1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_spots  # noqa: E402

from ultimate_analysis.simulation import batch  # noqa: E402
from ultimate_analysis.simulation import spot_features as sf  # noqa: E402
from ultimate_analysis.simulation.ppo import STALL_REWARD, THEIRS_CAUGHT  # noqa: E402
from ultimate_analysis.simulation.spots import (  # noqa: E402
    SpotBrains,
    load_spot_brains,
    save_spot_brains,
)

GAMMA, LAMBDA = train_spots.GAMMA, train_spots.LAMBDA
CLIP, ENTROPY = train_spots.CLIP, train_spots.ENTROPY
CHECKS_WEIGHED = 3  # Checks against scripted 2 whose mean says how good a team is
P, PLAYERS = batch.P, batch.PLAYERS


# ---------------------------------------------------------------------- playing


def rollout(game, players, steps):
    """Play every point on for `steps` steps; what was chosen, and the rewards."""
    game.begin_rollout(steps)
    players.kept = []
    with torch.no_grad():
        for _ in range(steps):
            game.step(*players.decide())
            players.forget(game.done[game.step_count - 1])
        # The value at the end, for each player, by whether their team has the disc
        final_value = torch.zeros(game.B, P, device=game.device)
        for team in (0, 1):
            rows = slice(team * PLAYERS, (team + 1) * PLAYERS)
            seen = batch.critic_inputs(game, team)
            attacking = (game.offence == team)[:, None]
            off = players.brains.values["offence"](seen)[..., 0]
            dfn = players.brains.values["defence"](seen)[..., 0]
            final_value[:, rows] = torch.where(attacking, off, dfn)
    kept = {
        key: torch.cat([k[key] for k in players.kept]) for key in players.kept[0] if key != "view"
    }
    kept["view"] = sf.stack_views([k["view"] for k in players.kept])
    return kept, final_value


def advantages(game, kept, final_value, credit, gamma=GAMMA, lam=LAMBDA):
    """GAE over each player's choices, and what happened after each (the rewards to the
    end of the point, discounted, or to the end of the rollout and then the value)."""
    T, B = game.rewards.shape[:2]
    dev = game.device
    team_of = torch.arange(P, device=dev) // PLAYERS
    reward = game.rewards[:, :, team_of] + credit * game.own  # (T, B, P)
    done = game.done.float()[..., None]  # (T, B, 1)
    G = torch.zeros(T + 1, B, P, device=dev)
    boot = torch.zeros(T + 1, B, P, device=dev)
    boot[T] = final_value
    for t in range(T - 1, -1, -1):
        G[t] = reward[t] + gamma * G[t + 1] * (1 - done[t])
        boot[t] = gamma * boot[t + 1] * (1 - done[t])
    ended_before = torch.cat(
        [torch.zeros(1, B, device=dev), game.done.float().cumsum(0)]
    )  # (T+1, B)
    g, p, s, value = kept["game"], kept["player"], kept["step"], kept["value"]
    order = torch.argsort((g * P + p) * (T + 1) + s)
    g, p, s, value = g[order], p[order], s[order], value[order]
    n = len(g)
    same_player = torch.zeros(n, dtype=torch.bool, device=dev)
    same_player[:-1] = (g[1:] == g[:-1]) & (p[1:] == p[:-1])
    nxt = torch.arange(1, n + 1, device=dev).clamp(max=n - 1)
    s_next = s[nxt]
    same_point = same_player & (ended_before[s_next, g] == ended_before[s, g])
    gap = (s_next - s).float()
    here = G[s, g, p]
    happened = here + boot[s, g, p]
    later = torch.where(same_point, gamma**gap * G[s_next, g, p], torch.zeros_like(here))
    after = torch.where(same_point, gamma**gap * value[nxt], torch.zeros_like(here))
    last = ~same_player
    delta = here - later + after - value
    delta = torch.where(last, here + boot[s, g, p] - value, delta)
    # Backwards along each player's choices: by how many choices come after
    count_after = torch.zeros(n, dtype=torch.long, device=dev)
    group_end = torch.nonzero(~same_player)[:, 0]
    group_start = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), group_end[:-1] + 1])
    lengths = group_end - group_start + 1
    position = torch.arange(n, device=dev) - torch.repeat_interleave(group_start, lengths)
    count_after = torch.repeat_interleave(lengths, lengths) - 1 - position
    advantage = torch.zeros(n, device=dev)
    for level in range(int(count_after.max()) + 1):
        at = count_after == level
        # Faded once per choice, not per step: choices far apart keep their credit
        carry = torch.where(same_point[at], gamma ** gap[at] * lam * advantage[nxt[at]], 0.0)
        advantage[at] = delta[at] + carry
    back = torch.empty_like(order)
    back[order] = torch.arange(n, device=dev)
    return advantage[back], happened[back]


# ---------------------------------------------------------------------- learning


def learn(
    brains,
    kept,
    advantage,
    happened,
    optimisers,
    value_optimisers,
    epochs,
    batch_size,
    values_only,
    share=1.0,
    only=None,
):
    dev = advantage.device
    prior = batch.throw_prior(dev)
    for kind, side in ((0, "offence"), (1, "defence"), (2, "throw")):
        mine = torch.nonzero(kept["kind"] == kind)[:, 0]
        if len(mine) == 0:
            continue
        network = brains.throws if kind == 2 else brains.spots[side]
        value_side = "defence" if kind == 1 else "offence"
        normalised = torch.zeros_like(advantage)
        a = advantage[mine]
        normalised[mine] = (a - a.mean()) / (a.std() + 1e-8)
        for _ in range(epochs):
            # Each pass over a share of them, drawn afresh
            shuffled = mine[torch.randperm(len(mine), device=dev)][: max(1, int(share * len(mine)))]
            for start in range(0, len(shuffled), batch_size):
                rows = shuffled[start : start + batch_size]
                value_loss = (
                    (brains.values[value_side](kept["seen"][rows])[:, 0] - happened[rows]) ** 2
                ).mean()
                value_optimisers[value_side].zero_grad()
                value_loss.backward()
                value_optimisers[value_side].step()
                if values_only or (only is not None and only != ("throw" if kind == 2 else side)):
                    continue  # The values learn; this side's choices stay as they are
                view = {name: kept["view"][name][rows] for name in sf.VIEW_FIELDS}
                if kind == 2:
                    leans = prior.expand(len(rows), -1)
                    scores = network(sf.throw_choices(view))[..., 0] + leans
                else:
                    parts = sf.spot_parts(view)
                    leans = sf.spot_leans(view, parts["allowed"])
                    scores = sf.spot_scores(network, parts) + leans
                log_chances = torch.log_softmax(scores, -1)
                new = log_chances.gather(1, kept["chosen"][rows][:, None])[:, 0]
                ratio = torch.exp(new - kept["log_chance"][rows])
                adv = normalised[rows]
                loss = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
                # Kept trying other choices as it does at first (the leans), not all alike:
                # drawn towards all alike, it unlearns going on and marking for nothing
                at_first = torch.log_softmax(leans, -1)
                apart = (log_chances.exp() * (log_chances - at_first)).sum(-1).mean()
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
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--points", type=int, default=2048, help="Played at once")
    parser.add_argument("--steps", type=int, default=200, help="Tenths of a second an iteration")
    parser.add_argument("--point-seconds", type=float, default=70.0)
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--value-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4096, help="Choices per learning step")
    parser.add_argument(
        "--learn-share",
        type=float,
        default=1.0,
        help="Of the choices of an iteration, the share each pass learns from",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Iterations of values only")
    parser.add_argument(
        "--marking",
        type=float,
        default=0.0,
        help="Share of points where one team defends by rules, man to man (0: self-play)",
    )
    parser.add_argument(
        "--only",
        choices=("offence", "defence", "throw"),
        default=None,
        help="Only these choices learn (where attackers run, where defenders run, or the "
        "throws); the rest play as they are",
    )
    parser.add_argument(
        "--fresh",
        choices=("offence", "defence"),
        default=None,
        help="With --start: this side's network for where to run begins anew",
    )
    parser.add_argument("--credit", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--stall-reward", type=float, default=STALL_REWARD)
    parser.add_argument("--stalled", type=float, default=3.0)
    parser.add_argument(
        "--theirs-caught",
        type=float,
        default=THEIRS_CAUGHT,
        help="To a defender when the attacker they guard catches a pass (0: nothing of their "
        "own, so that leaving them to cover a more dangerous place may pay)",
    )
    parser.add_argument("--report", type=int, default=20)
    parser.add_argument("--check", type=int, default=192, help="Points against the scripted team")
    parser.add_argument("--workers", type=int, default=12, help="For the checks, on the processor")
    parser.add_argument(
        "--spread",
        type=float,
        default=0.0,
        help="Share of points where one team attacks by rules, spread out, for the defence",
    )
    parser.add_argument(
        "--league",
        type=float,
        default=0.0,
        help="Share of points where one team is an earlier team of the run (or its start)",
    )
    parser.add_argument("--league-size", type=int, default=5, help="Latest teams kept to play")
    parser.add_argument(
        "--ladder",
        default="10,50",
        help="Iterations back of the teams each kept team also plays (and the run's start)",
    )
    parser.add_argument(
        "--revert",
        type=int,
        default=24,
        help="Back to the best team when the last checks are this many points worse (of the "
        "margin, their mean) against scripted 2 than its were (0: never)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/models/movement"))
    args = parser.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    brains = load_spot_brains(args.start) if args.start else SpotBrains()
    if args.fresh:
        brains.spots[args.fresh].load_state_dict(SpotBrains().spots[args.fresh].state_dict())
    brains = brains.to(dev)
    optimisers = {
        "offence": torch.optim.Adam(brains.spots["offence"].parameters(), lr=args.rate),
        "defence": torch.optim.Adam(brains.spots["defence"].parameters(), lr=args.rate),
        "throw": torch.optim.Adam(brains.throws.parameters(), lr=args.rate),
    }
    value_optimisers = {
        side: torch.optim.Adam(brains.values[side].parameters(), lr=args.value_rate)
        for side in ("offence", "defence")
    }
    game = batch.BatchGame(
        args.points,
        dev,
        args.seed,
        args.point_seconds,
        args.stalled,
        args.stall_reward,
        args.theirs_caught,
    )
    players = batch.BatchPlayers(game, brains, args.marking, args.league, args.spread)
    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_spots_gpu"
    run.mkdir(parents=True)
    settings = {**vars(args), "start": str(args.start), "output": str(args.output)}
    (run / "settings.json").write_text(json.dumps(settings, indent=2))

    def say(text: str, flush: bool = True) -> None:
        """Printed, and kept in the run's log.txt (scripts/watch_spots.py shows it)."""
        print(text, flush=flush)
        with open(run / "log.txt", "a", encoding="utf-8") as log:
            log.write(text + "\n")

    say(f"Written to {run}")
    checking = None
    best = None  # (margin against scripted 2 over the last checks, iteration)
    margins = []  # Of the last checks: one check alone is too noisy to go back on
    ladder = [int(n) for n in args.ladder.split(",") if n.strip()]
    start_state = brains.state() if args.start else None  # The team the run started from

    def rival(state):
        found = SpotBrains()
        found.load(state)
        return found.to(dev)

    if args.league > 0 and start_state is not None:
        players.set_rivals([rival(start_state)])
    with Pool(args.workers) as pool:
        for iteration in range(1, args.iterations + 1):
            began = time.perf_counter()
            before = dict(game.tally)
            kept, final_value = rollout(game, players, args.steps)
            torch.cuda.synchronize()
            played = time.perf_counter() - began
            advantage, happened = advantages(game, kept, final_value, args.credit, args.gamma)
            learn(
                brains,
                kept,
                advantage,
                happened,
                optimisers,
                value_optimisers,
                args.epochs,
                args.batch,
                iteration <= args.warmup,
                args.learn_share,
                args.only,
            )
            torch.cuda.synchronize()
            tally = {k: v - before.get(k, 0) for k, v in game.tally.items()}
            say(
                f"iteration {iteration}: {int(game.done.sum())} points ended, "
                f"{tally.get('score', 0)} scored; completed {tally.get('catch', 0)}/"
                f"{tally.get('throw', 0)}; {len(advantage)} choices; "
                f"{played:.0f} s playing, {time.perf_counter() - began - played:.0f} s learning"
                + (" (values only)" if iteration <= args.warmup else ""),
                flush=True,
            )
            if checking is not None and checking[1].ready():
                at, result, names = checking
                parts = result.get()
                results = {}
                for name in dict.fromkeys(names):
                    found = [p for part, n in zip(parts, names) if n == name for p in part]
                    scored = sum(p["scored"] for p in found)
                    conceded = sum(p["conceded"] for p in found)
                    results[name] = {"scored": scored, "conceded": conceded, "points": len(found)}
                    say(
                        f"  against {name} (iteration {at}): scored {scored}, conceded "
                        f"{conceded} of {len(found)}; completed "
                        f"{sum(p['catches'] for p in found)}/{sum(p['throws'] for p in found)}"
                        f"; they completed {sum(p['their_catches'] for p in found)}/"
                        f"{sum(p['their_throws'] for p in found)}",
                        flush=True,
                    )
                (run / f"iter_{at:04d}" / "against_scripted.json").write_text(
                    json.dumps(results["scripted"])
                )
                (run / f"iter_{at:04d}" / "checks.json").write_text(json.dumps(results, indent=2))
                checking = None
                bench = results["scripted 2"]
                margin = bench["scored"] - bench["conceded"]
                margins = [*margins, margin][-CHECKS_WEIGHED:]
                margin = round(sum(margins) / len(margins))
                if len(margins) < CHECKS_WEIGHED:
                    pass  # Too few checks yet to say how good this team is
                elif best is None or margin > best[0]:
                    best = (margin, at)
                    (run / "best.json").write_text(
                        json.dumps(
                            {
                                "iteration": at,
                                "margin against scripted 2": margin,
                                "over checks": CHECKS_WEIGHED,
                            }
                        )
                    )
                elif args.revert and margin <= best[0] - args.revert:
                    margins = []  # Weighed afresh from the team gone back to
                    kept_best = run / f"iter_{best[1]:04d}" / "spots.pt"
                    brains.load(torch.load(kept_best, map_location="cpu", weights_only=True))
                    for optimiser in [*optimisers.values(), *value_optimisers.values()]:
                        optimiser.state.clear()
                    say(
                        f"  {margin:+d} against scripted 2 over {CHECKS_WEIGHED} checks, the best "
                        f"{best[0]:+d} (iteration "
                        f"{best[1]}): back to that team",
                        flush=True,
                    )
            if iteration % args.report == 0:
                folder = run / f"iter_{iteration:04d}"
                save_spot_brains(brains, folder, iteration=iteration)
                save_spot_brains(brains, run / "latest", iteration=iteration)
                if args.league > 0:
                    # The league: the run's start and its latest kept teams
                    kept_teams = sorted(run.glob("iter_*/spots.pt"))[-args.league_size :]
                    states = [start_state] if start_state is not None else []
                    states += [
                        torch.load(p, map_location="cpu", weights_only=True) for p in kept_teams
                    ]
                    players.set_rivals([rival(s) for s in states])
                if checking is None:
                    state = brains.state()
                    share = max(1, args.check // args.workers)
                    # The scripted team, the teams of this run some iterations back, and
                    # the one it started from
                    opponents = [("scripted", "scripted"), ("scripted 2", "scripted2")]
                    for back in ladder:
                        earlier = run / f"iter_{iteration - back:04d}" / "spots.pt"
                        if iteration - back > 0 and earlier.exists():
                            kept = torch.load(earlier, map_location="cpu", weights_only=True)
                            opponents.append((f"iteration {iteration - back}", kept))
                    if start_state is not None:
                        opponents.append(("the start", start_state))
                    jobs, names = [], []
                    for name, opponent in opponents:
                        # The benchmark twice as many points: 48 are too few to tell
                        times = 2 if name == "scripted 2" else 1
                        for _ in range(args.workers * times):
                            jobs.append(
                                (
                                    state,
                                    opponent,
                                    [int(s) for s in rng.integers(0, 2**31, share)],
                                    120.0,
                                    0.0,
                                    0.0,
                                    False,  # Played to see, not kept
                                )
                            )
                            names.append(name)
                    checking = (iteration, pool.map_async(train_spots.play, jobs), names)


if __name__ == "__main__":
    main()
