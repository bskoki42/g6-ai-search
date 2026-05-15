import random
import streamlit as st

st.set_page_config(page_title="パズドラ風パズル", page_icon="🧩", layout="centered")

ROWS = 5
COLS = 6
EMPTY_EMOJI = "⬜"
ORB_INFO = {
    "fire": {"emoji": "🔴", "name": "火"},
    "water": {"emoji": "🔵", "name": "水"},
    "wood": {"emoji": "🟢", "name": "木"},
    "light": {"emoji": "🟡", "name": "光"},
    "dark": {"emoji": "🟣", "name": "闇"},
    "heart": {"emoji": "🩷", "name": "回復"},
}
ORB_TYPES = list(ORB_INFO.keys())


def generate_board():
    return [[random.choice(ORB_TYPES) for _ in range(COLS)] for _ in range(ROWS)]


def init_state():
    if "board" not in st.session_state:
        st.session_state.board = generate_board()
    if "selected" not in st.session_state:
        st.session_state.selected = None
    if "path" not in st.session_state:
        st.session_state.path = []
    if "result" not in st.session_state:
        st.session_state.result = None


def reset_selection():
    st.session_state.selected = None
    st.session_state.path = []


def new_board():
    st.session_state.board = generate_board()
    reset_selection()
    st.session_state.result = None


def shuffle_board():
    board = st.session_state.board
    flat = [orb for row in board for orb in row]
    random.shuffle(flat)
    for r in range(ROWS):
        for c in range(COLS):
            board[r][c] = flat[r * COLS + c]
    reset_selection()
    st.session_state.result = None


def move_selected(dr: int, dc: int):
    selected = st.session_state.selected
    if selected is None:
        return
    sr, sc = selected
    nr, nc = sr + dr, sc + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return
    board = st.session_state.board
    board[sr][sc], board[nr][nc] = board[nr][nc], board[sr][sc]
    st.session_state.selected = (nr, nc)
    st.session_state.path.append((nr, nc))
    st.session_state.result = None


def find_matches(board):
    rows = len(board)
    cols = len(board[0])
    mask = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        run_color = None
        run_start = 0
        run_length = 0
        for c in range(cols):
            color = board[r][c]
            if color == run_color:
                run_length += 1
            else:
                if run_color is not None and run_length >= 3:
                    for k in range(run_start, run_start + run_length):
                        mask[r][k] = True
                run_color = color
                run_start = c
                run_length = 1
        if run_color is not None and run_length >= 3:
            for k in range(run_start, run_start + run_length):
                mask[r][k] = True

    for c in range(cols):
        run_color = None
        run_start = 0
        run_length = 0
        for r in range(rows):
            color = board[r][c]
            if color == run_color:
                run_length += 1
            else:
                if run_color is not None and run_length >= 3:
                    for k in range(run_start, run_start + run_length):
                        mask[k][c] = True
                run_color = color
                run_start = r
                run_length = 1
        if run_color is not None and run_length >= 3:
            for k in range(run_start, run_start + run_length):
                mask[k][c] = True

    return mask


def get_combos(board, mask):
    rows = len(board)
    cols = len(board[0])
    visited = [[False] * cols for _ in range(rows)]
    combos = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r][c] or visited[r][c]:
                continue
            color = board[r][c]
            if color is None:
                continue
            stack = [(r, c)]
            visited[r][c] = True
            count = 0
            while stack:
                x, y = stack.pop()
                count += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if mask[nx][ny] and not visited[nx][ny] and board[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
            combos.append({"color": color, "count": count})
    return combos


def clear_matches(board, mask):
    cleared = 0
    for r in range(len(board)):
        for c in range(len(board[0])):
            if mask[r][c]:
                board[r][c] = None
                cleared += 1
    return cleared


def drop_new_orbs(board):
    for c in range(len(board[0])):
        stack = [board[r][c] for r in range(len(board)) if board[r][c] is not None]
        for r in range(len(board) - 1, -1, -1):
            if stack:
                board[r][c] = stack.pop()
            else:
                board[r][c] = random.choice(ORB_TYPES)


def resolve_board(board):
    cascades = []
    while True:
        mask = find_matches(board)
        combos = get_combos(board, mask)
        if not combos:
            break
        cascades.append(combos)
        clear_matches(board, mask)
        drop_new_orbs(board)
    return cascades


def format_result(cascades):
    if not cascades:
        return {"total": 0, "lines": ["コンボはありませんでした。"]}
    total_combos = sum(len(combo_list) for combo_list in cascades)
    lines = [f"{total_combos}コンボ！"]
    for idx, combos in enumerate(cascades, start=1):
        pieces = []
        for combo in combos:
            info = ORB_INFO[combo["color"]]
            pieces.append(f"{info['emoji']} {info['name']} ×{combo['count']}")
        lines.append(f"{idx}連鎖: " + " / ".join(pieces))
    return {"total": total_combos, "lines": lines}


def render_result(result):
    if not result:
        return
    text = "\n".join(result["lines"])
    if result["total"]:
        st.success(text)
    else:
        st.info(text)


def render_board():
    clicked = None
    path_index = {
        coord: idx for idx, coord in enumerate(st.session_state.path)
    } if st.session_state.path else {}
    for r in range(ROWS):
        row_cols = st.columns(COLS)
        for c in range(COLS):
            orb_key = st.session_state.board[r][c]
            emoji = ORB_INFO[orb_key]["emoji"] if orb_key is not None else EMPTY_EMOJI
            label = emoji
            if (r, c) == st.session_state.selected:
                label = f"👉{emoji}"
            elif (r, c) in path_index:
                label = f"{emoji}{path_index[(r, c)] + 1}"
            if row_cols[c].button(label, key=f"cell_{r}_{c}"):
                clicked = (r, c)
    return clicked


def finish_move():
    cascades = resolve_board(st.session_state.board)
    st.session_state.result = format_result(cascades)
    reset_selection()


init_state()

st.title("パズドラをあそぼう！")
st.caption("ドロップをタップして矢印ボタンで動かし、コンボを狙おう！")

with st.sidebar:
    st.header("あそびかた")
    st.markdown(
        """
1. 盤面のドロップをタップして持ち上げます。
2. 矢印ボタンで移動させてルートを作ります。
3. 「移動完了」ボタンを押すとコンボ判定＆落ちコンが発生します。
        """
    )
    st.markdown(
        """
- 「新しい盤面」で盤面を入れ替えられます。
- 「選択を解除」で持ち上げたドロップを戻せます。
- ルートに表示される数字は経路の順番を示しています。
        """
    )

controls = st.columns(3)
if controls[0].button("新しい盤面", type="primary"):
    new_board()
if controls[1].button("盤面をシャッフル"):
    shuffle_board()
if controls[2].button("選択を解除"):
    reset_selection()

st.divider()
render_result(st.session_state.result)

clicked_cell = render_board()
if clicked_cell is not None:
    st.session_state.selected = clicked_cell
    st.session_state.path = [clicked_cell]
    st.session_state.result = None

if st.session_state.path:
    coords = [f"({r + 1}, {c + 1})" for r, c in st.session_state.path]
    st.caption("移動ルート: " + " → ".join(coords))

if st.session_state.selected is not None:
    st.markdown("**ドロップ移動中**")
    up_row = st.columns(3)
    if up_row[1].button("⬆️", key="move_up"):
        move_selected(-1, 0)

    middle_row = st.columns(3)
    if middle_row[0].button("⬅️", key="move_left"):
        move_selected(0, -1)
    if middle_row[1].button("移動完了", key="finish_move"):
        finish_move()
    if middle_row[2].button("➡️", key="move_right"):
        move_selected(0, 1)

    down_row = st.columns(3)
    if down_row[1].button("⬇️", key="move_down"):
        move_selected(1, 0)

st.write("盤面のドロップを動かして最大コンボを目指そう！")
