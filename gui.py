"""
围棋 GUI（Tkinter）。

需求要点：
- 进入先选择“人机对弈 / AI 对弈”，再进入棋盘界面
- 人机对弈：用户选择 1 种 AI（random/mcts/minimax），默认玩家执黑
- AI 对弈：用户选择黑方/白方 agent（3 选 1），点击“下一步”才走一步
- 显示当前回合、提子数，支持新游戏、悔棋
"""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
import time

from dlgo import GameState, Player, Point
from dlgo.goboard import IllegalMoveError, Move


AGENT_TYPES = ("random", "mcts_std", "mcts_enh", "minimax")


def _make_agent(kind: str):
    kind = kind.lower()
    if kind == "random":
        from agents.random_agent import RandomAgent

        return RandomAgent()
    if kind in ("mcts_std", "mcts_enh"):
        from agents.mcts_agent import MCTSAgent

        use_enhancements = kind == "mcts_enh"
        return MCTSAgent(
            num_rounds=200,
            time_limit_s=2.0,
            rollout_depth=15,
            use_enhancements=use_enhancements,
        )
    if kind == "minimax":
        from agents.minimax_agent import MinimaxAgent

        return MinimaxAgent(max_depth=3)
    raise ValueError(f"未知 agent: {kind}")


def _wrap_select(agent_obj):
    return lambda s, a=agent_obj: a.select_move(s)


class StartFrame(tk.Frame):
    def __init__(self, master: "GoApp"):
        super().__init__(master)
        self.app = master

        self.mode_var = tk.StringVar(value="human_vs_ai")
        self.human_ai_var = tk.StringVar(value="mcts_enh")
        self.ai_black_var = tk.StringVar(value="mcts_enh")
        self.ai_white_var = tk.StringVar(value="random")

        tk.Label(self, text="请选择对弈模式", font=("Segoe UI", 12)).pack(pady=(10, 8))

        mode_box = tk.Frame(self)
        mode_box.pack(pady=(0, 8))
        tk.Radiobutton(mode_box, text="人机对弈", variable=self.mode_var, value="human_vs_ai", command=self._refresh).pack(
            side="left", padx=10
        )
        tk.Radiobutton(mode_box, text="AI 对弈", variable=self.mode_var, value="ai_vs_ai", command=self._refresh).pack(
            side="left", padx=10
        )

        self.config_box = tk.Frame(self)
        self.config_box.pack(padx=10, pady=(0, 10), fill="x")

        btns = tk.Frame(self)
        btns.pack(pady=(0, 10))
        tk.Button(btns, text="开始", width=10, command=self._start).pack(side="left", padx=8)
        tk.Button(btns, text="退出", width=10, command=self.app.destroy).pack(side="left", padx=8)

        self._refresh()

    def _clear_box(self):
        for w in self.config_box.winfo_children():
            w.destroy()

    def _refresh(self):
        self._clear_box()
        mode = self.mode_var.get()
        if mode == "human_vs_ai":
            tk.Label(self.config_box, text="请选择 AI 类型（你执黑先行）").pack(anchor="w", pady=(2, 6))
            row = tk.Frame(self.config_box)
            row.pack(anchor="w")
            tk.Label(row, text="AI：").pack(side="left")
            tk.OptionMenu(row, self.human_ai_var, *AGENT_TYPES).pack(side="left", padx=(6, 0))
        else:
            tk.Label(self.config_box, text="请选择黑白双方 Agent 类型").pack(anchor="w", pady=(2, 6))
            row = tk.Frame(self.config_box)
            row.pack(anchor="w")
            tk.Label(row, text="黑方：").pack(side="left")
            tk.OptionMenu(row, self.ai_black_var, *AGENT_TYPES).pack(side="left", padx=(6, 12))
            tk.Label(row, text="白方：").pack(side="left")
            tk.OptionMenu(row, self.ai_white_var, *AGENT_TYPES).pack(side="left", padx=(6, 0))

    def _start(self):
        mode = self.mode_var.get()
        if mode == "human_vs_ai":
            config = {
                "mode": "human_vs_ai",
                "black": "human",
                "white": self.human_ai_var.get(),
            }
        else:
            config = {
                "mode": "ai_vs_ai",
                "black": self.ai_black_var.get(),
                "white": self.ai_white_var.get(),
            }
        self.app.start_game(config)


class BoardFrame(tk.Frame):
    def __init__(self, master: "GoApp", board_size: int):
        super().__init__(master)
        self.app = master
        self.board_size = board_size

        self.mode = "human_vs_ai"
        self.player_kind = {Player.black: "human", Player.white: "random"}
        self.agents = {Player.black: None, Player.white: None}

        self.game = GameState.new_game(board_size)
        self.captures = {Player.black: 0, Player.white: 0}
        self.history: list[tuple[GameState, dict[Player, int]]] = []
        self._no_play_hint_hash: int | None = None  # 防止同一局面重复提示
        self._ai_busy = False
        self._game_start_ts = time.time()

        # UI 尺寸参数（默认放大一些，便于观察）
        self.cell_size = 80
        self.margin = 26
        self.stone_radius = 30
        canvas_size = self.margin * 2 + self.cell_size * (self.board_size - 1)

        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        self.turn_var = tk.StringVar()
        self.captures_var = tk.StringVar()
        self.extra_var = tk.StringVar()
        info = tk.Frame(top)
        info.pack(side="left", fill="x", expand=True)
        tk.Label(info, textvariable=self.turn_var, anchor="w").pack(fill="x")
        tk.Label(info, textvariable=self.captures_var, anchor="w").pack(fill="x")
        tk.Label(info, textvariable=self.extra_var, anchor="w").pack(fill="x")

        controls = tk.Frame(top)
        controls.pack(side="right")
        tk.Button(controls, text="返回", command=self.app.back_to_start).pack(side="right", padx=(6, 0))
        tk.Button(controls, text="新游戏", command=self.new_game).pack(side="right", padx=(6, 0))
        self.undo_btn = tk.Button(controls, text="悔棋", command=self.undo)
        self.undo_btn.pack(side="right", padx=(6, 0))
        self.pass_btn = tk.Button(controls, text="Pass", command=self.pass_turn)
        self.pass_btn.pack(side="right", padx=(6, 0))
        self.resign_btn = tk.Button(controls, text="认输", command=self.resign)
        self.resign_btn.pack(side="right", padx=(6, 0))
        self.step_btn = tk.Button(controls, text="下一步", command=self.next_step)
        self.step_btn.pack(side="right", padx=(6, 0))

        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size, bg="#DEB887", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 10))
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw()
        self._update_status()

    def configure_game(self, config: dict):
        self.mode = config["mode"]
        black = config["black"]
        white = config["white"]
        self.player_kind = {Player.black: black, Player.white: white}
        self.agents = {
            Player.black: (_make_agent(black) if black != "human" else None),
            Player.white: (_make_agent(white) if white != "human" else None),
        }
        self.new_game()

        # AI 对弈只允许“下一步”驱动；人机对弈隐藏其意义，但保留可用（若轮到 AI 可点）
        if self.mode == "ai_vs_ai":
            self.step_btn.configure(state="normal")
            # AI 对弈：去除 pass/悔棋 功能（禁用即可）
            self.pass_btn.configure(state="disabled")
            self.undo_btn.configure(state="disabled")
            self.resign_btn.configure(state="disabled")
        else:
            self.step_btn.configure(state="normal")
            self.pass_btn.configure(state="normal")
            self.undo_btn.configure(state="normal")
            self.resign_btn.configure(state="normal")

    def new_game(self):
        self.game = GameState.new_game(self.board_size)
        self.captures = {Player.black: 0, Player.white: 0}
        self.history.clear()
        self._no_play_hint_hash = None
        self._game_start_ts = time.time()
        self.draw()
        self._update_status()

    def undo(self):
        if self._ai_busy:
            return
        if not self.history:
            return

        any_human = "human" in self.player_kind.values()
        steps = 0
        max_steps = 2 if any_human else 1
        while self.history and steps < max_steps:
            self.game, cap = self.history.pop()
            self.captures = {Player.black: cap[Player.black], Player.white: cap[Player.white]}
            steps += 1
            if any_human and self._is_human_turn():
                break

        self.draw()
        self._update_status()

    def pass_turn(self):
        if self.game.is_over():
            return
        if self.mode == "ai_vs_ai":
            return
        if self._ai_busy:
            return
        if not self._is_human_turn():
            return
        self._apply_move(Move.pass_turn(), allow_ai_reply=True)

    def resign(self):
        if self.game.is_over():
            return
        if self.mode == "ai_vs_ai":
            return
        if self._ai_busy:
            return
        if not self._is_human_turn():
            return
        self._apply_move(Move.resign(), allow_ai_reply=False)

    def next_step(self):
        if self.game.is_over():
            return
        if self._ai_busy:
            return
        if self._is_human_turn():
            return
        self._play_ai_move()

    def on_click(self, event):
        if self.game.is_over():
            return
        if self.mode == "ai_vs_ai":
            return
        if self._ai_busy:
            return
        if not self._is_human_turn():
            return
        point = self._pixel_to_point(event.x, event.y)
        if point is None:
            return
        move = Move.play(point)
        if not self.game.is_valid_move(move):
            messagebox.showwarning("非法落子", "该位置不能下子，请重新选择位置。")
            return
        self._apply_move(move, allow_ai_reply=True)

    def _is_human_turn(self) -> bool:
        return self.player_kind.get(self.game.next_player) == "human"

    def _apply_move(self, move: Move, allow_ai_reply: bool):
        try:
            if not self.game.is_valid_move(move):
                return
            self._push_history()
            self.game = self.game.apply_move(move)
            self._update_captures_from_last_move()
        except (IllegalMoveError, AssertionError):
            self._pop_history_if_same()
            return

        self.draw()
        self._update_status()
        # 立刻刷新 UI，避免被 AI 计算阻塞导致“棋子一起出现”
        self.update_idletasks()
        if self._maybe_game_over():
            return

        if allow_ai_reply and self.mode == "human_vs_ai" and not self._is_human_turn():
            # 让 AI 的计算放到下一次事件循环执行，确保先显示玩家落子
            self._ai_busy = True
            self._update_status()
            self.after(10, self._play_ai_move)

    def _play_ai_move(self):
        if self._ai_busy is False and self.mode == "human_vs_ai":
            # 人机模式下该方法可能被按钮触发，此处不强制 busy
            pass
        agent = self.agents.get(self.game.next_player)
        if agent is None:
            self._ai_busy = False
            return
        ai_move = agent.select_move(self.game)
        if not self.game.is_valid_move(ai_move):
            ai_move = Move.pass_turn()
        self._apply_move(ai_move, allow_ai_reply=False)
        self._ai_busy = False
        self._update_status()

    def _maybe_game_over(self) -> bool:
        if not self.game.is_over():
            return False
        winner = self.game.winner()
        if winner is None:
            messagebox.showinfo("终局", "平局")
        else:
            messagebox.showinfo("终局", f"胜者：{winner.name}")
        return True

    def _push_history(self):
        self.history.append((self.game, dict(self.captures)))

    def _pop_history_if_same(self):
        if self.history and self.history[-1][0] == self.game:
            self.history.pop()

    def draw(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_stones()

    def _draw_grid(self):
        n = self.board_size
        for i in range(n):
            x0, y0 = self._grid_to_pixel(1, i + 1)
            x1, y1 = self._grid_to_pixel(n, i + 1)
            self.canvas.create_line(x0, y0, x1, y1, fill="black")

            x0, y0 = self._grid_to_pixel(i + 1, 1)
            x1, y1 = self._grid_to_pixel(i + 1, n)
            self.canvas.create_line(x0, y0, x1, y1, fill="black")

        for r in range(1, n + 1):
            x, y = self._grid_to_pixel(r, 1)
            self.canvas.create_text(x - 12, y, text=str(r), fill="black")
        for c in range(1, n + 1):
            x, y = self._grid_to_pixel(1, c)
            self.canvas.create_text(x, y - 12, text=str(c), fill="black")

    def _draw_stones(self):
        board = self.game.board
        for r in range(1, self.board_size + 1):
            for c in range(1, self.board_size + 1):
                stone = board.get(Point(r, c))
                if stone is None:
                    continue
                x, y = self._grid_to_pixel(r, c)
                fill = "black" if stone == Player.black else "white"
                self.canvas.create_oval(
                    x - self.stone_radius,
                    y - self.stone_radius,
                    x + self.stone_radius,
                    y + self.stone_radius,
                    fill=fill,
                    outline="black",
                )

    def _grid_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        x = self.margin + (col - 1) * self.cell_size
        y = self.margin + (row - 1) * self.cell_size
        return x, y

    def _pixel_to_point(self, x: int, y: int) -> Point | None:
        col_f = (x - self.margin) / self.cell_size + 1
        row_f = (y - self.margin) / self.cell_size + 1
        col = int(round(col_f))
        row = int(round(row_f))
        if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
            return None

        gx, gy = self._grid_to_pixel(row, col)
        if abs(x - gx) > self.cell_size * 0.35 or abs(y - gy) > self.cell_size * 0.35:
            return None
        return Point(row, col)

    def _update_status(self):
        # 回合数：已下步数 = len(history)；当前手数 = len(history)+1
        move_count = len(self.history)
        elapsed = int(time.time() - self._game_start_ts)
        mm = elapsed // 60
        ss = elapsed % 60

        if self.game.is_over():
            self.turn_var.set("终局")
        else:
            color = "黑" if self.game.next_player == Player.black else "白"
            who = "人类" if self._is_human_turn() else "AI"
            self.turn_var.set(f"当前回合：{color}（{who}）")
        self.captures_var.set(f"提子数  黑:{self.captures[Player.black]}  白:{self.captures[Player.white]}")
        self.extra_var.set(f"回合数: {move_count + 1}（已下 {move_count} 手）  用时: {mm:02d}:{ss:02d}")

        # 人类回合若没有任何合法落子点，提示用户需要 Pass（避免“点哪都没反应”）
        if (
            (not self.game.is_over())
            and self.mode == "human_vs_ai"
            and self._is_human_turn()
        ):
            moves = self.game.legal_moves()
            has_play = any(m.is_play for m in moves)
            if not has_play:
                h = self.game.board.zobrist_hash()
                if self._no_play_hint_hash != h:
                    self._no_play_hint_hash = h
                    messagebox.showinfo("提示", "当前无合法落子点，请点击 Pass（停一手）。")

    def _update_captures_from_last_move(self):
        if not self.history:
            return
        prev_state, prev_caps = self.history[-1]
        last_move = self.game.last_move
        if last_move is None or not last_move.is_play:
            return

        player = prev_state.next_player
        opponent = player.other

        def count_in(state: GameState, color: Player) -> int:
            board = state.board
            cnt = 0
            for r in range(1, self.board_size + 1):
                for c in range(1, self.board_size + 1):
                    if board.get(Point(r, c)) == color:
                        cnt += 1
            return cnt

        before_opponent = count_in(prev_state, opponent)
        after_opponent = count_in(self.game, opponent)
        captured = max(0, before_opponent - after_opponent)
        if captured:
            self.captures[player] = prev_caps[player] + captured


class GoApp(tk.Tk):
    def __init__(self, board_size: int = 5):
        super().__init__()
        self.title(f"围棋 AI（{board_size}×{board_size}）")
        # 允许缩放窗口（棋盘为固定像素绘制，但不会被裁切）
        self.resizable(True, True)
        # 初始窗口放大一些，避免界面过于紧凑
        self.geometry("600x600")
        self.minsize(500, 500)

        self.board_size = board_size
        self.start_frame = StartFrame(self)
        self.board_frame = BoardFrame(self, board_size=board_size)

        self.start_frame.pack(fill="both", expand=True)

    def start_game(self, config: dict):
        self.start_frame.pack_forget()
        self.board_frame.pack(fill="both", expand=True)
        self.board_frame.configure_game(config)

    def back_to_start(self):
        self.board_frame.pack_forget()
        self.start_frame.pack(fill="both", expand=True)


def main():
    parser = argparse.ArgumentParser(description="围棋 GUI（Tkinter）")
    parser.add_argument("--size", type=int, default=5, help="棋盘大小（默认 5）")
    args = parser.parse_args()

    app = GoApp(board_size=args.size)
    app.mainloop()


if __name__ == "__main__":
    main()

