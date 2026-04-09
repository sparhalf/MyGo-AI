"""
批量对弈统计脚本：支持 random / mcts_std / mcts_enh / minimax 任意组合（默认 5×5）。

示例：
    python bulk_play.py --games 200 --size 5 --black mcts --white random --out result.txt
    python bulk_play.py --games 200 --size 5 --black minimax --white mcts --swap
"""

from __future__ import annotations

import argparse
import datetime
from typing import Callable, Optional, Any, Dict

from dlgo import Player
from play import play_game


def _make_agent_factory(kind: str, args, side: str) -> Callable[[], object]:
    kind = kind.lower()
    if kind == "random":
        from agents.random_agent import RandomAgent

        return lambda: RandomAgent()
    if kind in ("mcts_std", "mcts_enh"):
        from agents.mcts_agent import MCTSAgent

        use_enhancements = kind == "mcts_enh"
        rounds = args.mcts_rounds
        tlim = args.mcts_time
        depth = args.rollout_depth

        if side == "black":
            if args.black_mcts_rounds is not None:
                rounds = args.black_mcts_rounds
            if args.black_mcts_time is not None:
                tlim = args.black_mcts_time
            if args.black_rollout_depth is not None:
                depth = args.black_rollout_depth
        else:
            if args.white_mcts_rounds is not None:
                rounds = args.white_mcts_rounds
            if args.white_mcts_time is not None:
                tlim = args.white_mcts_time
            if args.white_rollout_depth is not None:
                depth = args.white_rollout_depth

        return lambda: MCTSAgent(
            num_rounds=rounds,
            time_limit_s=tlim,
            rollout_depth=depth,
            use_enhancements=use_enhancements,
        )
    if kind == "minimax":
        from agents.minimax_agent import MinimaxAgent

        return lambda: MinimaxAgent(max_depth=args.minimax_depth)
    raise ValueError(f"未知 agent 类型: {kind}（可选：random/mcts_std/mcts_enh/minimax）")


def _wrap_select(agent_obj):
    return lambda s, a=agent_obj: a.select_move(s)


def _winner_name(winner: Optional[Player]) -> str:
    if winner is None:
        return "draw"
    return winner.name


def main():
    parser = argparse.ArgumentParser(description="批量运行对局并统计结果")
    parser.add_argument("--games", type=int, default=100, help="对局数")
    parser.add_argument("--size", type=int, default=5, help="棋盘大小（默认 5）")

    parser.add_argument(
        "--black",
        choices=["random", "mcts_std", "mcts_enh", "minimax"],
        default="mcts_std",
        help="黑方 agent 类型",
    )
    parser.add_argument(
        "--white",
        choices=["random", "mcts_std", "mcts_enh", "minimax"],
        default="random",
        help="白方 agent 类型",
    )

    parser.add_argument("--mcts_rounds", type=int, default=1000, help="MCTS 每步模拟轮数上限")
    parser.add_argument("--mcts_time", type=float, default=10.0, help="MCTS 每步时间上限（秒）")
    parser.add_argument("--rollout_depth", type=int, default=30, help="增强版 rollout 深度上限")
    parser.add_argument("--black_mcts_rounds", type=int, default=None, help="黑方 MCTS rounds（覆盖全局）")
    parser.add_argument("--white_mcts_rounds", type=int, default=None, help="白方 MCTS rounds（覆盖全局）")
    parser.add_argument("--black_mcts_time", type=float, default=None, help="黑方 MCTS time（覆盖全局）")
    parser.add_argument("--white_mcts_time", type=float, default=None, help="白方 MCTS time（覆盖全局）")
    parser.add_argument("--black_rollout_depth", type=int, default=None, help="黑方 rollout_depth（覆盖全局）")
    parser.add_argument("--white_rollout_depth", type=int, default=None, help="白方 rollout_depth（覆盖全局）")
    # 兼容旧参数：以前用 --enhanced 切换 mcts 行为；现在用 mcts_std/mcts_enh 区分双方
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="（兼容旧版本参数）忽略。请改用 --black mcts_enh/--white mcts_enh",
    )
    parser.add_argument("--minimax_depth", type=int, default=3, help="Minimax 最大深度")

    parser.add_argument(
        "--swap",
        action="store_true",
        help="每局交换先后手（减少先手偏置）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="将每局结果与汇总写入该 txt 文件（如 result.txt）",
    )
    args = parser.parse_args()

    black_factory = _make_agent_factory(args.black, args, side="black")
    white_factory = _make_agent_factory(args.white, args, side="white")

    results = {Player.black: 0, Player.white: 0, None: 0}
    total_moves = 0
    total_time = 0.0

    algo_wins = {args.black: 0, args.white: 0, "draw": 0}

    log_fp = None
    if args.out:
        log_fp = open(args.out, "w", encoding="utf-8")
        log_fp.write(f"bulk_play started_at={datetime.datetime.now().isoformat(timespec='seconds')}\n")
        log_fp.write(f"board={args.size}x{args.size} games={args.games} swap={args.swap}\n")
        log_fp.write(
            f"black={args.black} white={args.white} | "
            f"mcts_rounds={args.mcts_rounds} mcts_time={args.mcts_time} rollout_depth={args.rollout_depth} | "
            f"black_mcts_rounds={args.black_mcts_rounds} black_mcts_time={args.black_mcts_time} black_rollout_depth={args.black_rollout_depth} | "
            f"white_mcts_rounds={args.white_mcts_rounds} white_mcts_time={args.white_mcts_time} white_rollout_depth={args.white_rollout_depth} | "
            f"minimax_depth={args.minimax_depth}\n"
        )
        log_fp.write("\n")

    for i in range(args.games):
        swap_sides = bool(args.swap and (i % 2 == 1))
        if not swap_sides:
            black_agent = black_factory()
            white_agent = white_factory()
            agent1 = _wrap_select(black_agent)
            agent2 = _wrap_select(white_agent)
            black_kind = args.black
            white_kind = args.white
        else:
            black_agent = white_factory()
            white_agent = black_factory()
            agent1 = _wrap_select(black_agent)
            agent2 = _wrap_select(white_agent)
            black_kind = args.white
            white_kind = args.black

        winner, moves, duration = play_game(
            agent1, agent2, board_size=args.size, verbose=False
        )

        results[winner] += 1
        total_moves += moves
        total_time += duration

        if winner is None:
            algo_wins["draw"] += 1
        elif winner == Player.black:
            algo_wins[black_kind] += 1
        else:
            algo_wins[white_kind] += 1

        if log_fp is not None:
            log_fp.write(
                f"game={i+1:05d} swap={int(swap_sides)} black={black_kind} white={white_kind} "
                f"winner={_winner_name(winner)} moves={moves} time={duration:.3f}s\n"
            )

    print("\n========== 批量统计 ==========")
    print(f"对局数: {args.games}")
    print(f"棋盘: {args.size}×{args.size}")
    print(f"黑方: {args.black} | 白方: {args.white}")
    print(f"MCTS: rounds<={args.mcts_rounds}, time<={args.mcts_time}s, rollout_depth={args.rollout_depth}")
    print(f"Minimax: depth={args.minimax_depth}")
    print(f"是否交换先后手: {'是' if args.swap else '否'}")

    print("\n-- 按黑白胜负 --")
    print(f"黑胜: {results[Player.black]}")
    print(f"白胜: {results[Player.white]}")
    print(f"平局: {results[None]}")

    print("\n-- 按算法胜负 --")
    for k in [args.black, args.white, "draw"]:
        print(f"{k}: {algo_wins.get(k, 0)}")

    if args.games > 0:
        print(f"\n平均步数: {total_moves / args.games:.1f}")
        print(f"平均用时: {total_time / args.games:.2f}s")
        decisive = args.games - results[None]
        if decisive > 0:
            # 这里按“命令行指定的 black/white 组合”给出胜率（交换先后手也仍统计到各算法名下）
            a = algo_wins.get(args.black, 0)
            b = algo_wins.get(args.white, 0)
            print(f"{args.black} 胜率(不含平局): {a / max(1, (a + b)):.3f}")

    if log_fp is not None:
        log_fp.write("\n========== summary ==========\n")
        log_fp.write(f"black={args.black} white={args.white} swap={args.swap}\n")
        log_fp.write(f"black_win={results[Player.black]} white_win={results[Player.white]} draw={results[None]}\n")
        log_fp.write(f"avg_moves={total_moves / max(1, args.games):.1f} avg_time={total_time / max(1, args.games):.3f}s\n")
        for k in [args.black, args.white, 'draw']:
            log_fp.write(f"{k}={algo_wins.get(k, 0)}\n")
        log_fp.close()


if __name__ == "__main__":
    main()

