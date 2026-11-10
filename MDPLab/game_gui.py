"""
game_gui.py — Pygame view/controller for Treasure Hunt
=========================================================

This file has NO MDP math in it. It only:
  - renders the grid and side panel,
  - handles keyboard/mouse input,
  - asks an injected PolicyEngine to solve the CURRENT random layout
    when you switch into policy mode or hit reset.

That injection point is the whole "engine outside the game" idea: swap
`engine=PolicyIterationEngine()` for `engine=ValueIterationEngine()` (or
pass both and toggle) without touching a single line in this file.

Controls
--------
Arrow keys / WASD : move (human mode)
R                 : reset — regenerates a brand-new random layout
P                 : toggle policy mode (solves the CURRENT layout)
SPACE             : advance one policy step
A                 : toggle auto-play of the policy
ESC               : quit
"""

import pygame

from env import TreasureHuntEnvironment
from mdp_builder import build_mdp_tensors


class TreasureHuntGUI:

    def __init__(self, engine=None, discount=0.95, rows=4, cols=4,
                 num_walls=2, num_danger=1):
        pygame.init()

        self.engine = engine            # a PolicyEngine, or None
        self.discount = discount
        self.rows, self.cols = rows, cols
        self.num_walls, self.num_danger = num_walls, num_danger

        self.environment = TreasureHuntEnvironment(
            layout=TreasureHuntEnvironment.generate_random_layout(
                rows=rows, cols=cols, num_walls=num_walls, num_danger=num_danger,
            )
        )

        self.policy = None               # np.ndarray, filled in when solved
        self.state_to_index = None
        self.solve_info = None           # last PolicyEngineResult, for display

        self.cell_size = 115
        self.grid_width = self.environment.cols * self.cell_size
        self.grid_height = self.environment.rows * self.cell_size
        self.side_panel_width = 400
        self.window_width = self.grid_width + self.side_panel_width
        self.window_height = max(self.grid_height + 120, 620)

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Treasure Hunt: An MDP Game")

        self.title_font = pygame.font.SysFont("sans", 32, bold=True)
        self.heading_font = pygame.font.SysFont("sans", 22, bold=True)
        self.normal_font = pygame.font.SysFont("sans", 19)
        self.small_font = pygame.font.SysFont("sans", 15)
        self.cell_font = pygame.font.SysFont("sans", 42, bold=True)

        self.background = (24, 28, 38)
        self.panel_background = (34, 40, 54)
        self.grid_background = (225, 230, 235)
        self.grid_line = (65, 72, 84)
        self.wall_color = (70, 74, 82)
        self.start_color = (185, 220, 245)
        self.treasure_color = (245, 210, 95)
        self.goal_color = (125, 215, 150)
        self.danger_color = (235, 125, 120)
        self.player_color = (80, 145, 240)
        self.text_color = (240, 242, 245)
        self.secondary_text = (180, 188, 202)
        self.button_color = (72, 100, 155)
        self.button_hover = (92, 125, 185)
        self.button_text = (250, 250, 250)

        self.running = True
        self.policy_mode = False
        self.auto_policy = False
        self.policy_delay = 550
        self.last_policy_time = 0

        self.reset_button = pygame.Rect(self.grid_width + 30, 480, 150, 46)
        self.policy_button = pygame.Rect(self.grid_width + 200, 480, 165, 46)
        self.auto_button = pygame.Rect(self.grid_width + 30, 535, 335, 44)

        self.clock = pygame.time.Clock()

    # --------------------------------------------------------
    # Policy solving — the only place this file talks to the engine
    # --------------------------------------------------------

    def _solve_current_layout(self):
        if self.engine is None:
            self.environment.message = "No policy engine attached."
            return

        P, R, states, state_to_index = build_mdp_tensors(self.environment)
        result = self.engine.solve(P, R, discount=self.discount)

        self.policy = result.policy
        self.state_to_index = state_to_index
        self.solve_info = result
        self.environment.message = (
            f"{result.engine_name} solved this layout "
            f"({len(states)} states) in {result.iterations} iterations."
        )

    def get_policy_action(self):
        state = self.environment.get_state()
        index = self.state_to_index[state]
        return int(self.policy[index])

    # --------------------------------------------------------
    # Drawing helpers
    # --------------------------------------------------------

    def draw_text(self, text, font, color, x, y):
        self.screen.blit(font.render(text, True, color), (x, y))

    def draw_centered_text(self, text, font, color, rect):
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def draw_grid(self):
        env = self.environment
        for row in range(env.rows):
            for col in range(env.cols):
                position = (row, col)
                x, y = col * self.cell_size, row * self.cell_size
                cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                color = self.grid_background
                if position == env.start:
                    color = self.start_color
                if position == env.treasure and not env.has_treasure:
                    color = self.treasure_color
                if position == env.goal:
                    color = self.goal_color
                if position in env.danger:
                    color = self.danger_color
                if position in env.walls:
                    color = self.wall_color

                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, self.grid_line, cell_rect, 2)

                label = ""
                if position == env.start:
                    label = "S"
                if position == env.treasure and not env.has_treasure:
                    label = "T"
                if position == env.goal:
                    label = "G"
                if position in env.danger:
                    label = "!"
                if position in env.walls:
                    label = "X"
                if label:
                    self.draw_centered_text(label, self.cell_font, (30, 35, 42), cell_rect)

                if position == env.position:
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    radius = self.cell_size // 4
                    pygame.draw.circle(self.screen, self.player_color, center, radius)
                    pygame.draw.circle(self.screen, (20, 35, 65), center, radius, 3)
                    self.draw_centered_text("P", self.heading_font, (255, 255, 255), cell_rect)

        pygame.draw.rect(self.screen, self.text_color,
                          pygame.Rect(0, 0, self.grid_width, self.grid_height), 3)

    def draw_panel(self):
        env = self.environment
        panel_rect = pygame.Rect(self.grid_width, 0, self.side_panel_width, self.window_height)
        pygame.draw.rect(self.screen, self.panel_background, panel_rect)

        x = self.grid_width + 25
        self.draw_text("TREASURE HUNT", self.title_font, self.text_color, x, 22)
        self.draw_text("Random layout each reset", self.normal_font, self.secondary_text, x, 65)

        self.draw_text("GAME STATE", self.heading_font, self.text_color, x, 118)
        self.draw_text(f"Treasure: {'YES' if env.has_treasure else 'NO'}",
                        self.normal_font, self.text_color, x, 155)
        self.draw_text(f"Total reward: {env.total_reward}", self.normal_font, self.text_color, x, 186)
        self.draw_text(f"Steps: {env.steps}", self.normal_font, self.text_color, x, 217)

        self.draw_text("LAST TRANSITION", self.heading_font, self.text_color, x, 262)
        if env.last_chosen_action is None:
            chosen_text, actual_text = "Chosen: --", "Actual: --"
        else:
            from env import ACTIONS, ACTION_SYMBOLS
            c, a = env.last_chosen_action, env.last_actual_action
            chosen_text = f"Chosen: {ACTIONS[c]} {ACTION_SYMBOLS[c]}"
            actual_text = f"Actual: {ACTIONS[a]} {ACTION_SYMBOLS[a]}"
        self.draw_text(chosen_text, self.normal_font, self.text_color, x, 298)
        self.draw_text(actual_text, self.normal_font, self.text_color, x, 328)
        self.draw_text(f"Reward: {env.last_reward}", self.normal_font, self.text_color, x, 358)

        mode_text = "MODE: POLICY" if self.policy_mode else "MODE: HUMAN"
        mode_color = (120, 220, 170) if self.policy_mode else (130, 185, 245)
        self.draw_text(mode_text, self.heading_font, mode_color, x, 400)

        if self.solve_info is not None:
            self.draw_text(
                f"Engine: {self.solve_info.engine_name}",
                self.small_font, self.secondary_text, x, 428,
            )
            self.draw_text(
                f"Converged in {self.solve_info.iterations} iterations",
                self.small_font, self.secondary_text, x, 448,
            )

        mouse_position = pygame.mouse.get_pos()
        self.draw_button(self.reset_button, "NEW LAYOUT", mouse_position)
        self.draw_button(self.policy_button,
                          "HUMAN MODE" if self.policy_mode else "POLICY MODE", mouse_position)
        self.draw_button(self.auto_button,
                          "STOP AUTO PLAY" if self.auto_policy else "AUTO PLAY POLICY", mouse_position)

        self.draw_text("CONTROLS", self.heading_font, self.text_color, x, 600)
        for i, line in enumerate([
            "WASD / Arrows: move",
            "P: policy mode   Space: one policy step",
            "A: auto-play policy   R: new random layout",
        ]):
            self.draw_text(line, self.small_font, self.secondary_text, x, 632 + i * 24)

        message_rect = pygame.Rect(15, self.grid_height + 18, self.grid_width - 30, 75)
        pygame.draw.rect(self.screen, (38, 45, 58), message_rect, border_radius=10)
        self.draw_centered_text(env.message, self.normal_font, self.text_color, message_rect)

    def draw_button(self, rect, label, mouse_position):
        color = self.button_hover if rect.collidepoint(mouse_position) else self.button_color
        pygame.draw.rect(self.screen, color, rect, border_radius=9)
        self.draw_centered_text(label, self.small_font, self.button_text, rect)

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    def new_random_layout(self):
        layout = TreasureHuntEnvironment.generate_random_layout(
            rows=self.rows, cols=self.cols,
            num_walls=self.num_walls, num_danger=self.num_danger,
        )
        self.environment = TreasureHuntEnvironment(layout=layout)
        self.auto_policy = False
        self.policy = None
        self.state_to_index = None
        self.solve_info = None
        if self.policy_mode:
            self._solve_current_layout()

    def human_action(self, action):
        if self.policy_mode or self.environment.done:
            return
        self.environment.step(action)

    def policy_step(self):
        if not self.policy_mode or self.environment.done:
            self.auto_policy = False
            return
        self.environment.step(self.get_policy_action())

    def toggle_policy_mode(self):
        self.policy_mode = not self.policy_mode
        self.auto_policy = False
        if self.policy_mode:
            self._solve_current_layout()
        else:
            self.environment.message = "Human control enabled."

    # --------------------------------------------------------
    # Event handling / main loop
    # --------------------------------------------------------

    def handle_key(self, key):
        # In human mode, WASD/arrows move — 'A' is LEFT.
        # In policy mode, movement keys do nothing (human_action() no-ops),
        # so 'A' is free to mean "toggle auto-play" instead.
        if key == pygame.K_a and self.policy_mode:
            self.auto_policy = not self.auto_policy
            return

        action_keys = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_DOWN: 1, pygame.K_s: 1,
            pygame.K_LEFT: 2, pygame.K_a: 2,
            pygame.K_RIGHT: 3, pygame.K_d: 3,
        }
        if key in action_keys:
            self.human_action(action_keys[key])
        elif key == pygame.K_r:
            self.new_random_layout()
        elif key == pygame.K_p:
            self.toggle_policy_mode()
        elif key == pygame.K_SPACE:
            self.policy_step()
        elif key == pygame.K_ESCAPE:
            self.running = False

    def handle_mouse(self, position):
        if self.reset_button.collidepoint(position):
            self.new_random_layout()
        elif self.policy_button.collidepoint(position):
            self.toggle_policy_mode()
        elif self.auto_button.collidepoint(position) and self.policy_mode:
            self.auto_policy = not self.auto_policy

    def run(self):
        while self.running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse(event.pos)

            if self.policy_mode and self.auto_policy and not self.environment.done:
                if current_time - self.last_policy_time >= self.policy_delay:
                    self.policy_step()
                    self.last_policy_time = current_time

            if self.environment.done:
                self.auto_policy = False

            self.screen.fill(self.background)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
