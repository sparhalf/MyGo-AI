"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

from __future__ import annotations

import math
import random
import time

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MCTSAgent"]


def _count_stones(game_state: GameState, color: Player) -> int:
    board = game_state.board
    cnt = 0
    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            if board.get(Point(r, c)) == color:
                cnt += 1
    return cnt



class MCTSNode:
    """
    MCTS 树节点。


    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
    """

    def __init__(self, game_state, parent=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.move = None  # 从父节点走到当前节点的棋步
        self.player_just_played = None  # 谁走了 self.move（根节点为 None）
        moves = list(game_state.legal_moves())
        # MCTS 里认输一般没有决策价值；若仍有其它合法走法，则不展开 resign
        non_resign = [m for m in moves if not m.is_resign]
        self.untried_moves = non_resign if non_resign else moves

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        if not self.children:
            return None

        parent_visits = max(1, self.visit_count)
        best = None
        best_score = -1e30
        for child in self.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                score = child.value + c * math.sqrt(math.log(parent_visits) / child.visit_count)
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        if not self.untried_moves:
            return None
        move = self.untried_moves.pop()
        next_state = self.game_state.apply_move(move)
        child = MCTSNode(next_state, parent=self)
        child.move = move
        # child 是“我方走了 move 之后”到达的局面
        child.player_just_played = self.game_state.next_player
        self.children.append(child)
        return child

    def backup(self, winner: Player | None):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            winner: 模拟/评估得到的赢家（None 表示和棋）
        """
        node = self
        while node is not None:
            node.visit_count += 1
            if node.player_just_played is None:
                # 根节点没有“上一手落子方”，只累计访问次数即可
                pass
            elif winner is None:
                node.value_sum += 0.5
            elif winner == node.player_just_played:
                node.value_sum += 1.0
            else:
                node.value_sum += 0.0
            node = node.parent


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(
        self,
        num_rounds=1000,
        temperature=1.0,
        time_limit_s=10.0,
        rollout_depth=30,
        use_enhancements=True,
    ):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.time_limit_s = float(time_limit_s)
        self.rollout_depth = int(rollout_depth)
        self.use_enhancements = bool(use_enhancements)

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            3. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        root = MCTSNode(game_state)

        start = time.time()
        rounds = 0
        while rounds < self.num_rounds and (time.time() - start) < self.time_limit_s:
            node = root

            # 1) Selection：用 UCT 下钻到可扩展节点
            while (not node.is_terminal()) and (not node.untried_moves) and node.children:
                node = node.best_child()
                if node is None:
                    break
            if node is None:
                break

            # 2) Expansion：扩展一个子节点
            if (not node.is_terminal()) and node.untried_moves:
                node = node.expand() or node

            # 3) Simulation：随机模拟（带优化）
            winner = self._simulate(node.game_state)

            # 4) Backup：反向传播
            node.backup(winner)

            rounds += 1

        best_move = self._select_best_move(root)
        return best_move if best_move is not None else Move.pass_turn()

    def _simulate(self, game_state: GameState) -> Player | None:
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜, 0=负, 0.5=和）
        """
        # 标准版：纯随机 rollout 直到终局（不做深度截断/启发式）
        if not self.use_enhancements:
            state = game_state
            while not state.is_over():
                moves = state.legal_moves()
                non_resign = [m for m in moves if not m.is_resign]
                state = state.apply_move(random.choice(non_resign) if non_resign else random.choice(moves))
            return state.winner()

        # 增强版：启发式 rollout + 限制深度
        state = game_state
        for _ in range(self.rollout_depth):
            if state.is_over():
                break
            move = self._select_rollout_move(state)
            state = state.apply_move(move)

        if state.is_over():
            return state.winner()

        # 深度截断后的快速评估：简单子数差（5×5 下足够快）
        black = _count_stones(state, Player.black)
        white = _count_stones(state, Player.white)
        score = black - white
        if score > 0:
            return Player.black
        if score < 0:
            return Player.white
        return None

    def _select_rollout_move(self, game_state: GameState) -> Move:
        """
        启发式 rollout 策略：
        - 优先选能提子的走法
        - 若不能提子，优先选“落子后气更多/避免自紧”的走法
        - 尽量避免无意义认输
        - 其次随机选落子（必要时才 pass）
        """
        moves = game_state.legal_moves()
        non_resign = [m for m in moves if not m.is_resign]
        if non_resign:
            moves = non_resign

        player = game_state.next_player
        opponent = player.other
        before = _count_stones(game_state, opponent)

        capture_moves = []
        for m in moves:
            if not m.is_play:
                continue
            next_state = game_state.apply_move(m)
            after = _count_stones(next_state, opponent)
            if after < before:
                capture_moves.append(m)
        if capture_moves:
            return random.choice(capture_moves)

        play_moves = [m for m in moves if m.is_play]
        if play_moves:
            scored = []
            for m in play_moves:
                ns = game_state.apply_move(m)
                s = ns.board.get_go_string(m.point)
                libs = s.num_liberties if s is not None else 0
                # 先避免自紧（气<=1），再偏好气更多
                scored.append(((libs <= 1, -libs), m))

            scored.sort(key=lambda item: item[0])
            best_key = scored[0][0]
            best_moves = [m for k, m in scored if k == best_key]
            return random.choice(best_moves)
        return random.choice(moves)

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        if not root.children:
            moves = root.game_state.legal_moves()
            non_resign = [m for m in moves if not m.is_resign]
            return random.choice(non_resign) if non_resign else random.choice(moves)

        candidates = [c for c in root.children if c.move is not None and not c.move.is_resign]
        if not candidates:
            candidates = [c for c in root.children if c.move is not None]

        max_visits = max(c.visit_count for c in candidates)
        top = [c for c in candidates if c.visit_count == max_visits]

        # 平手时按价值再筛一次；仍平手则随机打破，避免总是选列表第一个
        best_value = max(c.value for c in top)
        top2 = [c for c in top if c.value == best_value]
        return random.choice(top2).move
