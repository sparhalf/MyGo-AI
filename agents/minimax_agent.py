"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

from __future__ import annotations

import math

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=3, evaluator=None):
        self.max_depth = max_depth
        # 默认评估函数（TODO：学生可替换为神经网络）
        self.evaluator = evaluator or self._default_evaluator
        self._cache = GameResultCache()
        self._root_player: Player | None = None

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        self._root_player = game_state.next_player

        best_move = None
        best_value = -math.inf

        moves = self._get_ordered_moves(game_state)
        non_resign = [m for m in moves if not m.is_resign]
        if non_resign:
            moves = non_resign

        alpha = -math.inf
        beta = math.inf
        for move in moves:
            next_state = game_state.apply_move(move)
            value = self.alphabeta(
                next_state, self.max_depth - 1, alpha, beta, maximizing_player=False
            )
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move if best_move is not None else Move.pass_turn()

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        if game_state.is_over() or depth == 0:
            return self._terminal_or_eval(game_state)

        moves = self._get_ordered_moves(game_state)
        non_resign = [m for m in moves if not m.is_resign]
        if non_resign:
            moves = non_resign

        if maximizing_player:
            value = -math.inf
            for move in moves:
                value = max(
                    value,
                    self.minimax(
                        game_state.apply_move(move), depth - 1, maximizing_player=False
                    ),
                )
            return value
        else:
            value = math.inf
            for move in moves:
                value = min(
                    value,
                    self.minimax(
                        game_state.apply_move(move), depth - 1, maximizing_player=True
                    ),
                )
            return value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        zobrist_hash = game_state.board.zobrist_hash()
        cached = self._cache.get((zobrist_hash, depth, maximizing_player, self._root_player))
        if cached is not None:
            return cached[1]

        if game_state.is_over() or depth == 0:
            value = self._terminal_or_eval(game_state)
            self._cache.put((zobrist_hash, depth, maximizing_player, self._root_player), depth, value)
            return value

        moves = self._get_ordered_moves(game_state)
        non_resign = [m for m in moves if not m.is_resign]
        if non_resign:
            moves = non_resign

        if maximizing_player:
            value = -math.inf
            for move in moves:
                value = max(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        maximizing_player=False,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for move in moves:
                value = min(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        maximizing_player=True,
                    ),
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break

        self._cache.put((zobrist_hash, depth, maximizing_player, self._root_player), depth, value)
        return value

    def _terminal_or_eval(self, game_state: GameState) -> float:
        if game_state.is_over():
            winner = game_state.winner()
            if winner is None:
                return 0.0
            return 1e6 if winner == self._root_player else -1e6
        return float(self.evaluator(game_state))

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        assert self._root_player is not None
        me = self._root_player
        opp = me.other

        board = game_state.board

        me_stones = 0
        opp_stones = 0
        me_libs = 0
        opp_libs = 0

        seen_strings = set()
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                pt = Point(r, c)
                stone = board.get(pt)
                if stone is None:
                    continue
                if stone == me:
                    me_stones += 1
                else:
                    opp_stones += 1

                s = board.get_go_string(pt)
                if s is None:
                    continue
                key = (s.color, s.stones)
                if key in seen_strings:
                    continue
                seen_strings.add(key)
                if s.color == me:
                    me_libs += s.num_liberties
                else:
                    opp_libs += s.num_liberties

        # 子数差 + 气数差（气权重略小，避免过度偏好“虚气”）
        return (me_stones - opp_stones) + 0.2 * (me_libs - opp_libs)

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        moves = game_state.legal_moves()
        non_resign = [m for m in moves if not m.is_resign]
        if non_resign:
            moves = non_resign

        player = game_state.next_player
        opponent = player.other

        board = game_state.board
        before_opp = 0
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                if board.get(Point(r, c)) == opponent:
                    before_opp += 1

        scored = []
        for m in moves:
            if not m.is_play:
                # pass 放后面（仍保留）
                scored.append(((10, 0, 0), m))
                continue
            ns = game_state.apply_move(m)
            nb = ns.board
            after_opp = 0
            for r in range(1, nb.num_rows + 1):
                for c in range(1, nb.num_cols + 1):
                    if nb.get(Point(r, c)) == opponent:
                        after_opp += 1
            captured = max(0, before_opp - after_opp)
            s = ns.board.get_go_string(m.point)
            libs = s.num_liberties if s is not None else 0
            # key 越小越优先：先吃子(越多越好)，再避免自紧(气<=1)，再偏好气更多
            key = (0, -captured, libs <= 1, -libs)
            scored.append((key, m))

        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored]



class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        old = self.cache.get(zobrist_hash)
        if old is None:
            self.cache[zobrist_hash] = (depth, value, flag)
            return
        old_depth, _, _ = old
        if depth >= old_depth:
            self.cache[zobrist_hash] = (depth, value, flag)
