"""
frontend.py -- Pygame dashboard for the library bot.

New in this version:
  - A grid floor plan with wall obstacles, so the robot's route is a real
    pathfinding problem, not a straight line.
  - pathfinding.find_path (A*) computes the robot's route each trip.
  - More shelves (6 total; 3 hold the seeded books, 3 are unused shelving
    for visual complexity).
  - The dashboard shows each shelf's live bandit success_score/attempts
    (from shelves.get_shelf_reliability), and flags when a check was
    skipped because that shelf's catalog has proven reliable enough.

Run `python data.py` once first to seed digital_database.db / shelves.db.
Type a book id (book-1, book-2, or book-3) into the input box, press Enter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame

import digital_database
import shelves
from integrate import check_shelf, compare_db
from a_star import find_path

# ---------------------------------------------------------------------------
# Grid / floor plan
# ---------------------------------------------------------------------------

CELL = 36
# '#' = wall / shelf block (robot cannot stand here), '.' = open floor
GRID_TEMPLATE = [
    "................",
    ".###..###..###..",
    "................",
    ".###..###..###..",
    "................",
    "................",
    ".###..###..###..",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
]
GRID = [[1 if ch == "#" else 0 for ch in row] for row in GRID_TEMPLATE]
ROWS, COLS = len(GRID), len(GRID[0])
MAP_W = COLS * CELL
MAP_H = ROWS * CELL

PANEL_W = 380
WIDTH, HEIGHT = MAP_W + PANEL_W, max(MAP_H, 640)
FPS = 60
STEP_FRAMES = 12   # frames to glide between adjacent grid cells

COLOR_BG = (245, 244, 238)
COLOR_MAP_BG = (255, 255, 255)
COLOR_WALL = (120, 150, 200)
COLOR_WALL_TEXT = (20, 30, 60)
COLOR_GRID_LINE = (230, 230, 225)
COLOR_HOME = (180, 180, 180)
COLOR_ROBOT = (210, 90, 50)
COLOR_PATH = (240, 190, 170)
COLOR_PANEL_BG = (250, 250, 247)
COLOR_TEXT = (40, 40, 38)
COLOR_MUTED = (120, 120, 115)
COLOR_CONFLICT = (170, 40, 40)
COLOR_OK = (30, 110, 60)
COLOR_SKIP = (150, 110, 20)
COLOR_INPUT_BG = (255, 255, 255)
COLOR_INPUT_BORDER = (100, 100, 95)

Cell = Tuple[int, int]

# Each labeled shelf's APPROACH cell (open floor directly in front of its
# wall block) is what the robot paths to and what's shown on the panel.
# book-1/2/3 map onto the first three blocks; the other three are unused
# shelving, present for a more complex floor plan.
SHELF_APPROACH: dict = {
    "shelf-12B": (2, 2),
    "shelf-14A": (7, 2),
    "shelf-09C": (12, 2),
    "shelf-03A": (2, 4),
    "shelf-07D": (7, 4),
    "shelf-11F": (12, 4),
}
# Where to draw each shelf's wall block + label (top-left cell of the block)
SHELF_BLOCK_ORIGIN: dict = {
    "shelf-12B": (1, 1),
    "shelf-14A": (6, 1),
    "shelf-09C": (11, 1),
    "shelf-03A": (1, 3),
    "shelf-07D": (6, 3),
    "shelf-11F": (11, 3),
}
HOME_CELL: Cell = (1, 14)
ROBOT_ID = "robot-1"


def cell_to_px(cell: Cell) -> Tuple[float, float]:
    col, row = cell
    return (col * CELL + CELL / 2, row * CELL + CELL / 2)


# ---------------------------------------------------------------------------
# Pipeline state machine
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    stage: str = "idle"           # idle | moving_to_shelf | checking | moving_home | done
    book_id: Optional[str] = None
    shelf_id: Optional[str] = None
    db_record: Optional[dict] = None
    reconciled: Optional[dict] = None
    last_check_skipped: bool = False
    path: List[Cell] = field(default_factory=list)
    path_index: int = 0
    step_timer: int = 0
    check_timer: int = 0


@dataclass
class AppState:
    robot_cell: Cell = HOME_CELL
    robot_px: List[float] = field(default_factory=lambda: list(cell_to_px(HOME_CELL)))
    trail: List[Tuple[int, int]] = field(default_factory=list)
    log: List[Tuple[str, Tuple[int, int, int]]] = field(default_factory=list)
    input_text: str = ""
    pipeline: PipelineState = field(default_factory=PipelineState)

    def push_log(self, text: str, color: Tuple[int, int, int] = COLOR_TEXT) -> None:
        self.log.append((text, color))
        if len(self.log) > 10:
            self.log.pop(0)


def handle_query(app: AppState, book_id: str) -> None:
    book_id = book_id.strip()
    if not book_id:
        return

    if app.pipeline.stage not in ("idle", "done"):
      app.push_log("Robot is busy -- wait until it's back at the dock.", COLOR_MUTED)
      return

    record = digital_database.get_book_record(book_id)
    if record is None:
        app.push_log(f"'{book_id}' not found. Try book-1, book-2 or book-3.", COLOR_CONFLICT)
        return

    app.push_log(f"Catalog lookup: '{book_id}' -> {record['status']} at {record['shelf_id']}")

    goal = SHELF_APPROACH.get(record["shelf_id"])
    if goal is None:
        app.push_log(f"No floor-plan position for shelf '{record['shelf_id']}'.", COLOR_CONFLICT)
        return

    path = find_path(app.robot_cell, goal, GRID)
    if path is None:
        app.push_log(f"No route to {record['shelf_id']} -- blocked.", COLOR_CONFLICT)
        return

    app.pipeline = PipelineState(
        stage="moving_to_shelf",
        book_id=book_id,
        shelf_id=record["shelf_id"],
        db_record=record,
        path=path,
        path_index=0,
    )
    app.push_log(f"Route planned: {len(path)} cells to {record['shelf_id']}.")


def step_pipeline(app: AppState) -> None:
    p = app.pipeline
    if p.stage in ("idle", "done"):
        return

    if p.stage in ("moving_to_shelf", "moving_home"):
        _advance_along_path(app)
        if p.path_index >= len(p.path) - 1 and p.step_timer == 0:
            if p.stage == "moving_to_shelf":
                p.stage = "checking"
                p.check_timer = FPS
                app.push_log(f"Arrived at {p.shelf_id}. Checking shelf...")
            else:
                p.stage = "done"
                app.push_log("Robot back at charging dock.", COLOR_MUTED)
        return

    if p.stage == "checking":
        p.check_timer -= 1
        if p.check_timer <= 0:
            assert p.book_id is not None and p.shelf_id is not None and p.db_record is not None
            check = check_shelf(p.book_id, p.shelf_id)
            reconciled = compare_db(p.db_record["status"], check)
            p.reconciled = reconciled
            p.last_check_skipped = check.skipped

            if check.skipped:
                app.push_log(
                    "Shelf trusted (bandit) -- skipped physical check, used catalog directly.",
                    COLOR_SKIP,
                )
            elif reconciled["difference"]:
                app.push_log(
                    f"Check disagrees (correct_prob {check.correct_prob:.2f}): {reconciled['content']}",
                    COLOR_CONFLICT,
                )
                if reconciled["check_status"] is not None:
                    digital_database.update_book_status(p.book_id, reconciled["check_status"])
                    app.push_log(f"Catalog corrected to '{reconciled['check_status']}'.", COLOR_CONFLICT)
            else:
                app.push_log(f"Check confirms catalog: '{reconciled['check_status']}'.", COLOR_OK)

            home_path = find_path(app.robot_cell, HOME_CELL, GRID)
            p.path = home_path or [app.robot_cell]
            p.path_index = 0
            p.stage = "moving_home"


def _advance_along_path(app: AppState) -> None:
    p = app.pipeline
    if not p.path or p.path_index >= len(p.path) - 1:
        p.step_timer = 0
        return

    current_cell = p.path[p.path_index]
    next_cell = p.path[p.path_index + 1]
    p.step_timer += 1

    t = min(1.0, p.step_timer / STEP_FRAMES)
    cx, cy = cell_to_px(current_cell)
    nx, ny = cell_to_px(next_cell)
    app.robot_px[0] = cx + (nx - cx) * t
    app.robot_px[1] = cy + (ny - cy) * t

    app.trail.append((int(app.robot_px[0]), int(app.robot_px[1])))
    if len(app.trail) > 60:
        app.trail.pop(0)

    if t >= 1.0:
        p.path_index += 1
        p.step_timer = 0
        app.robot_cell = next_cell


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_map(screen: "pygame.Surface", app: AppState, font: "pygame.font.Font") -> None:
    pygame.draw.rect(screen, COLOR_MAP_BG, (0, 0, MAP_W, MAP_H))

    for c in range(COLS + 1):
        pygame.draw.line(screen, COLOR_GRID_LINE, (c * CELL, 0), (c * CELL, MAP_H))
    for r in range(ROWS + 1):
        pygame.draw.line(screen, COLOR_GRID_LINE, (0, r * CELL), (MAP_W, r * CELL))

    for row in range(ROWS):
        for col in range(COLS):
            if GRID[row][col] == 1:
                pygame.draw.rect(screen, COLOR_WALL, (col * CELL, row * CELL, CELL, CELL))

    for shelf_id, (ox, oy) in SHELF_BLOCK_ORIGIN.items():
        label = font.render(shelf_id, True, COLOR_WALL_TEXT)
        lx = ox * CELL + (3 * CELL - label.get_width()) // 2
        ly = oy * CELL - 16
        screen.blit(label, (lx, max(0, ly)))

    hx, hy = HOME_CELL
    pygame.draw.rect(screen, COLOR_HOME, (hx * CELL + 2, hy * CELL + 2, CELL - 4, CELL - 4), border_radius=6)

    for px, py in app.trail:
        pygame.draw.circle(screen, COLOR_PATH, (px, py), 3)

    rx, ry = int(app.robot_px[0]), int(app.robot_px[1])
    pygame.draw.circle(screen, COLOR_ROBOT, (rx, ry), 11)
    pygame.draw.circle(screen, (255, 255, 255), (rx, ry), 11, width=2)

    pygame.draw.line(screen, (210, 208, 200), (MAP_W, 0), (MAP_W, HEIGHT), 2)


def draw_panel(screen: "pygame.Surface", app: AppState, font: "pygame.font.Font",
               font_bold: "pygame.font.Font") -> None:
    pygame.draw.rect(screen, COLOR_PANEL_BG, (MAP_W, 0, PANEL_W, HEIGHT))
    x0 = MAP_W + 18
    y = 16

    title = font_bold.render("Stackbot dashboard", True, COLOR_TEXT)
    screen.blit(title, (x0, y))
    y += 32

    robot_state = digital_database.get_robot_state(ROBOT_ID) or {}
    battery = robot_state.get("battery_pct", "?")
    status_line = font.render(f"Battery: {battery}%   Stage: {app.pipeline.stage}", True, COLOR_MUTED)
    screen.blit(status_line, (x0, y))
    y += 26

    p = app.pipeline
    if p.db_record:
        line = font.render(f"Book: {p.book_id} (catalog: {p.db_record['status']})", True, COLOR_TEXT)
        screen.blit(line, (x0, y))
        y += 22
    if p.reconciled:
        if p.last_check_skipped:
            color, resolved = COLOR_SKIP, (p.reconciled["check_status"] or "unresolved") + " (skipped check)"
        else:
            color = COLOR_CONFLICT if p.reconciled["difference"] else COLOR_OK
            resolved = p.reconciled["check_status"] or "unresolved"
        line = font.render(f"Resolved: {resolved}", True, color)
        screen.blit(line, (x0, y))
        y += 26

    y += 6
    sect = font_bold.render("Shelf reliability (bandit)", True, COLOR_TEXT)
    screen.blit(sect, (x0, y))
    y += 24
    for shelf_id in SHELF_APPROACH:
        rel = shelves.get_shelf_reliability(shelf_id)
        bar_color = COLOR_OK if rel["success_score"] >= 0.9 else COLOR_MUTED
        line = font.render(
            f"{shelf_id}: {rel['success_score']:.2f} (n={rel['attempts']})", True, bar_color
        )
        screen.blit(line, (x0, y))
        y += 18

    y += 12
    log_title = font_bold.render("AI reasoning log", True, COLOR_TEXT)
    screen.blit(log_title, (x0, y))
    y += 24
    for text, color in app.log:
        for line in _wrap_text(text, font, PANEL_W - 36):
            screen.blit(font.render(line, True, color), (x0, y))
            y += 17

    box_y = HEIGHT - 50
    pygame.draw.rect(screen, COLOR_INPUT_BG, (x0, box_y, PANEL_W - 36, 32), border_radius=4)
    hint = app.input_text if app.input_text else "Type book-1/2/3, press Enter..."
    hint_color = COLOR_TEXT if app.input_text else COLOR_MUTED
    screen.blit(font.render(hint, True, hint_color), (x0 + 6, box_y + 7))


def _wrap_text(text: str, font: "pygame.font.Font", max_width: int) -> List[str]:
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    digital_database.init_db()
    shelves.init_db()
    if digital_database.get_book_record("book-1") is None:
        print("No data found -- run `python data.py` first to seed the databases.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Stackbot -- library bot dashboard")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 15)
    font_bold = pygame.font.SysFont("arial", 19, bold=True)

    app = AppState()
    app.push_log("Ready. Try 'book-1', 'book-2', or 'book-3'.", COLOR_MUTED)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    handle_query(app, app.input_text)
                    app.input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    app.input_text = app.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    app.input_text = ""
                else:
                    if event.unicode.isprintable():
                        app.input_text += event.unicode

        step_pipeline(app)

        screen.fill(COLOR_BG)
        draw_map(screen, app, font)
        draw_panel(screen, app, font, font_bold)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
