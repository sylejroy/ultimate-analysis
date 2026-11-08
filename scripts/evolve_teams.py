#!/usr/bin/env python3
"""Improve simulated teams by playing: evolution of their minds (simulation/learned.py).

A team's minds are one long list of numbers: the weights of its offence and defence
networks, the routes of its set plays, how much its holders lean to each play, and how
sharply they pick where to throw, and how they throw (simulation/scripted.py STYLE: curve,
loft, flatness, wind, pace). Each generation, copies of the present team with
small changes, each matched by its opposite change, play points: half starting with the
disc, half without, against the present team itself, the scripted team, and teams of
earlier generations (a league, so that beating the latest fashion is not enough). The
present team moves towards the changes that did better (evolution strategies, the
fitness ranked).

Fitness per point: +1 for a point scored, -1 for one conceded, a little for completing
throws and for making the other team miss theirs; less the further the networks have
moved from those learned from the recorded players (--keep, 0: free to play however
wins). Points are stopped unscored after --point-seconds.

Progress is printed per generation, and every --report generations the present team
plays the scripted team and the team it started from; it is kept then, in a new folder:

    <output>/<run>/gen_0010/minds.pt ... and latest/minds.pt

    python scripts/evolve_teams.py data/models/movement/20261107_232115_imitation \\
        --generations 500 --workers 12
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.simulation import game  # noqa: E402
from ultimate_analysis.simulation.learned import (  # noqa: E402
    LearnedTeam,
    Minds,
    load_minds,
    rewire,
    save_minds,
)
from ultimate_analysis.simulation.match import play_point  # noqa: E402
from ultimate_analysis.simulation.scripted import ScriptedTeam  # noqa: E402

# How far each kind of number is changed by, as a typical change
NETWORK_STEP, ROUTE_STEP, LEANING_STEP, TEMPERATURE_STEP = 0.01, 1.0, 0.3, 0.1
STYLE_STEPS = np.array([0.03, 0.3, 0.1, 0.1, 0.1])  # Curve, loft, flatness, wind, pace
COMPLETION_CREDIT = 0.02  # Fitness per throw completed, and per throw the other team missed


# ---------------------------------------------------------------------- numbers <-> minds


def _network_numbers(network) -> np.ndarray:
    return torch.nn.utils.parameters_to_vector(network.parameters()).detach().numpy().copy()


def numbers_of(minds: Minds) -> np.ndarray:
    """A team's minds as one list of numbers."""
    parts = [_network_numbers(minds.offence), _network_numbers(minds.defence)]
    parts += [route.ravel() for play in minds.plays for _, route in sorted(play.routes.items())]
    parts += [minds.play_leaning, [np.log(max(minds.throw_temperature, 1e-3))], minds.style]
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])


def steps_of(minds: Minds) -> np.ndarray:
    """How far each number is changed by, as a typical change."""
    networks = len(_network_numbers(minds.offence)) + len(_network_numbers(minds.defence))
    parts = [np.full(networks, NETWORK_STEP)]
    for play in minds.plays:
        for _, route in sorted(play.routes.items()):
            step = np.full(route.shape, ROUTE_STEP)
            step[:, 0] = 0.1 * ROUTE_STEP  # Seconds of a route change by less
            step[0, 0] = 0.0  # A route starts at the call
            parts.append(step.ravel())
    parts += [np.full(len(minds.play_leaning), LEANING_STEP), [TEMPERATURE_STEP], STYLE_STEPS]
    return np.concatenate(parts)


def wired_numbers(minds: Minds) -> np.ndarray:
    """1 for every number that may change, 0 for a cut connection's weight."""
    return (numbers_of(minds) != 0) | ~_is_weight(minds)


def _is_weight(minds: Minds) -> np.ndarray:
    """Which of the numbers are connection weights (not biases or the rest)."""
    found = []
    for network in (minds.offence, minds.defence):
        for name, parameter in network.named_parameters():
            found.append(np.full(parameter.numel(), name.endswith("weight")))
    rest = len(numbers_of(minds)) - sum(len(f) for f in found)
    return np.concatenate(found + [np.zeros(rest, dtype=bool)])


def minds_from(numbers: np.ndarray, like: Minds) -> Minds:
    """The minds a list of numbers stands for, laid out as `like`."""
    minds = copy.deepcopy(like)
    at = 0
    for network in (minds.offence, minds.defence):
        n = len(_network_numbers(network))
        torch.nn.utils.vector_to_parameters(
            torch.from_numpy(numbers[at : at + n].astype(np.float32)), network.parameters()
        )
        at += n
    for play in minds.plays:
        for role in sorted(play.routes):
            route = play.routes[role]
            values = numbers[at : at + route.size].reshape(route.shape).copy()
            values[:, 0] = np.maximum.accumulate(np.maximum(values[:, 0], 0.0))  # In order
            play.routes[role] = values
            at += route.size
    minds.play_leaning = numbers[at : at + len(minds.play_leaning)].copy()
    at += len(minds.play_leaning)
    minds.throw_temperature = float(np.exp(numbers[at]))
    at += 1
    minds.style = numbers[at : at + len(minds.style)].copy()
    return minds


# ---------------------------------------------------------------------- playing (workers)

_like = None
_throw_net = None


def _start_worker(like_folder: str, throw_weights: str) -> None:
    global _like, _throw_net
    from ultimate_analysis.processing.throw_targets import load_throw_targets

    torch.set_num_threads(1)
    _like = load_minds(Path(like_folder))
    _throw_net = load_throw_targets(throw_weights)


def _team(spec, seed):
    if spec is None:
        return ScriptedTeam(seed)
    return LearnedTeam(minds_from(spec, _like), _throw_net, seed)


def play(job) -> list:
    """Play points between a team (numbers) and an opponent (numbers, or None for the
    scripted team): (scored for, scored against, own completions, other's misses)."""
    numbers, opponent, seeds, limit = job
    results = []
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        ours, theirs = _team(numbers, seed), _team(opponent, seed + 1)
        # Half the points ours receives the pull, half the other team does; ours is
        # team 0 or team 1 by turns, so that neither end is favoured. Teams learn to play
        # in any wind
        side = k % 2
        teams = (ours, theirs) if side == 0 else (theirs, ours)
        receiving = side if (k // 2) % 2 == 0 else 1 - side
        start = game.pull_start(rng, receiving, game.a_wind(rng))
        state = play_point(teams, rng, start, limit=limit)
        caught = {0: 0, 1: 0}
        missed = {0: 0, 1: 0}
        for event in state.events:
            if event.kind == "catch":
                caught[event.team] += 1
            elif event.kind in ("block", "mark block", "drop", "out", "stall"):
                missed[event.team] += 1
        results.append(
            (
                int(state.scored == side),
                int(state.scored == 1 - side),
                caught[side],
                missed[1 - side],
            )
        )
    return results


def fitness_of(results: list) -> float:
    points = sum(r[0] - r[1] for r in results)
    credit = COMPLETION_CREDIT * sum(r[2] + r[3] for r in results)
    return (points + credit) / len(results)


# ---------------------------------------------------------------------- evolving


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("start", type=Path, help="Folder of the minds to start from")
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--population", type=int, default=24, help="Even: in opposite pairs")
    parser.add_argument("--points", type=int, default=4, help="Per team and generation")
    parser.add_argument("--point-seconds", type=float, default=60.0)
    parser.add_argument("--keep", type=float, default=0.5, help="Pull to the recorded players")
    parser.add_argument("--rate", type=float, default=0.03, help="Step towards what did better")
    parser.add_argument("--report", type=int, default=10)
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Rewiring thins the networks to this share of their connections",
    )
    parser.add_argument(
        "--rewire-every", type=int, default=0, help="Generations between rewirings; 0: never"
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/models/movement"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start_minds = load_minds(args.start)
    centre = numbers_of(start_minds)
    origin = centre.copy()
    steps = steps_of(start_minds)
    networks = len(_network_numbers(start_minds.offence)) + len(
        _network_numbers(start_minds.defence)
    )
    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_evolved"
    run.mkdir(parents=True)
    (run / "settings.json").write_text(
        json.dumps({**vars(args), "start": str(args.start), "output": str(args.output)}, indent=2)
    )
    throw_weights = str(get_setting("models.throw_targets.weights"))
    league = [origin.copy()]
    # The connections of the team started from: any it has cut stay cut until rewired
    wired = wired_numbers(start_minds) if args.rewire_every else np.ones(len(centre), dtype=bool)
    moment = np.zeros_like(centre)
    second = np.zeros_like(centre)
    print(
        f"{len(centre)} numbers, {networks} of them network weights; written to {run}", flush=True
    )

    with Pool(args.workers, _start_worker, (str(args.start), throw_weights)) as pool:
        for generation in range(1, args.generations + 1):
            began = time.perf_counter()
            pairs = args.population // 2
            if args.rewire_every and generation % args.rewire_every == 0:
                # The weakest connections are cut and new ones grown
                minds = minds_from(centre, start_minds)
                for network in (minds.offence, minds.defence):
                    rewire(network, rng, density=args.density)
                centre = numbers_of(minds)
                wired = wired_numbers(minds)
            noise = rng.standard_normal((pairs, len(centre))) * wired
            jobs = []
            for i in range(pairs):
                # Each pair meets the same opponent with the same luck
                choice = rng.random()
                opponent = (
                    centre.copy()
                    if choice < 0.4
                    else None
                    if choice < 0.7
                    else league[int(rng.integers(len(league)))]
                )
                seeds = [int(s) for s in rng.integers(0, 2**31, args.points)]
                for sign in (1, -1):
                    jobs.append(
                        (centre + sign * steps * noise[i], opponent, seeds, args.point_seconds)
                    )
            outcomes = pool.map(play, jobs)
            fitness = np.array([fitness_of(o) for o in outcomes])
            # Staying near the recorded players: the networks' distance from the start,
            # in typical changes
            drift = np.array(
                [
                    np.mean(((job[0][:networks] - origin[:networks]) / NETWORK_STEP) ** 2)
                    for job in jobs
                ]
            )
            score = fitness - args.keep * 0.01 * drift
            ranks = np.empty(len(score))
            ranks[np.argsort(score)] = np.linspace(-0.5, 0.5, len(score))
            gradient = ((ranks[0::2] - ranks[1::2])[:, None] * noise).sum(0) / len(noise)
            # Adam, in units of the typical change of each number
            moment = 0.9 * moment + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            corrected = moment / (1 - 0.9**generation)
            scale = np.sqrt(second / (1 - 0.999**generation)) + 1e-8
            # Cut connections stay cut (what moved them before they were cut is spent)
            centre = np.where(wired, centre + args.rate * steps * corrected / scale, 0.0)
            results = [r for o in outcomes for r in o]
            print(
                f"generation {generation}: fitness {fitness.mean():+.3f} (best {fitness.max():+.3f}), "
                f"scored {sum(r[0] for r in results)}, conceded {sum(r[1] for r in results)}, "
                f"drift {drift.mean():.1f}, wired {wired[_is_weight(start_minds)].mean():.0%}, "
                f"{time.perf_counter() - began:.0f} s",
                flush=True,
            )
            if generation % args.report == 0:
                folder = run / f"gen_{generation:04d}"
                save_minds(minds_from(centre, start_minds), folder, generation=generation)
                save_minds(minds_from(centre, start_minds), run / "latest", generation=generation)
                league.append(centre.copy())
                seeds = [int(s) for s in rng.integers(0, 2**31, 16)]
                against = {
                    "scripted": pool.apply(play, ((centre, None, seeds, 120.0),)),
                    "start": pool.apply(play, ((centre, origin, seeds, 120.0),)),
                }
                said = []
                for name, found in against.items():
                    said.append(
                        f"against {name}: scored {sum(r[0] for r in found)}, conceded "
                        f"{sum(r[1] for r in found)} of {len(found)} points"
                    )
                    (folder / f"against_{name}.json").write_text(json.dumps(found))
                print("  " + "; ".join(said), flush=True)


if __name__ == "__main__":
    main()
