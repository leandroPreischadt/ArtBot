import threading
from pathlib import Path

import arcade
from artbot.face.config.settings import HTTP_PORT
import uvicorn
from arcade.sprite.animated import TextureAnimation, TextureAnimationSprite, TextureKeyframe
from fastapi import FastAPI, HTTPException
from PIL import Image

from artbot.face.state import FaceState

WINDOW_TITLE = "Robot Face"

EYES_SCALE = 0.5
MOUTH_SCALE = 0.5
EYES_OFFSET_Y = 80
MOUTH_OFFSET_Y = -90
BACKGROUND_COLOR = (1, 61, 115)

BACKGROUND_IMAGE_PATH = Path(__file__).parent / "assets/bg.png"
SPRITES_DIR = Path(__file__).parent / "assets"

current_state = FaceState.IDLE

STATE_GIFS = {
    FaceState.IDLE:     {"eyes": "eyes-bored.gif", "mouth": "mouth-idle.gif"},
    FaceState.SPEAKING: {"eyes": "eyes-lookin-arround.gif", "mouth": "mouth-speaking.gif"},
    FaceState.THINKING: {"eyes": "eyes-thinking.gif",       "mouth": "mouth-thinking.gif"},
    FaceState.BORED:    {"eyes": "eyes-bored.gif",          "mouth": "mouth-idle.gif"},
}
DEFAULT_GIF = {"eyes": "eyes-lookin-arround.gif", "mouth": "mouth-idle.gif"}

_animation_cache: dict[Path, TextureAnimation] = {}


def load_gif_animation(path: Path) -> TextureAnimation:
    if path in _animation_cache:
        return _animation_cache[path]

    image = Image.open(path)
    logical_size = image.size

    keyframes: list[TextureKeyframe] = []
    canvas = Image.new("RGBA", logical_size, (0, 0, 0, 0))
    previous = canvas.copy()

    for i in range(image.n_frames): # type: ignore
        image.seek(i)
        duration = image.info.get("duration", 100) or 100
        disposal = getattr(image, "disposal_method", 1)
        extents = image.tile[0][1] if image.tile else (0, 0, *logical_size)

        if disposal == 3:
            previous = canvas.copy()

        frame = image.convert("RGBA")
        # Full-size frames must paste at (0, 0); offsetting smears pixels.
        if frame.size == logical_size:
            canvas.paste(frame, (0, 0), frame)
        else:
            canvas.paste(frame, extents[:2], frame) # type: ignore

        keyframes.append(TextureKeyframe(arcade.Texture(canvas.copy()), duration))

        if disposal == 2:
            w, h = extents[2] - extents[0], extents[3] - extents[1] # type: ignore
            canvas.paste(Image.new("RGBA", (w, h), (0, 0, 0, 0)), extents[:2]) # type: ignore
        elif disposal == 3:
            canvas = previous.copy()

    animation = TextureAnimation(keyframes)
    _animation_cache[path] = animation
    return animation


class FacePart(TextureAnimationSprite):
    def __init__(self, part_name: str, center_x: float, center_y: float, scale: float):
        folder = SPRITES_DIR / part_name
        self.state_animations = {
            state: load_gif_animation(folder / gifs.get(part_name, DEFAULT_GIF[part_name]))
            for state, gifs in STATE_GIFS.items()
        }
        super().__init__(
            center_x=center_x,
            center_y=center_y,
            scale=scale,
            animation=self.state_animations[FaceState.IDLE],
        )
        self.current_state = FaceState.IDLE

    def set_state(self, state: FaceState) -> None:
        if state == self.current_state:
            return
        self.current_state = state
        self.animation = self.state_animations[state]
        self.time = 0.0
        self._current_keyframe_index = 0

class GameView(arcade.View):
    def __init__(self) -> None:
        super().__init__()
        self.draw_face()

    def draw_face(self):
        width = self.window.width
        height = self.window.height
        center_x, center_y = width / 2, height / 2

        self.mouth = FacePart("mouth", center_x, center_y + MOUTH_OFFSET_Y, MOUTH_SCALE)
        self.eyes = FacePart("eyes", center_x, center_y + EYES_OFFSET_Y, EYES_SCALE)
        self.background_image = arcade.load_texture(file_path=BACKGROUND_IMAGE_PATH)
        self.background_image.height = height;
        self.background_image.width = width;
        self.background_sprite = arcade.Sprite(self.background_image, center_x=center_x, center_y=center_y)

        self.mouth_list = arcade.SpriteList()
        self.mouth_list.append(self.mouth)
        self.eyes_list = arcade.SpriteList()
        self.eyes_list.append(self.eyes)
        self.others_list = arcade.SpriteList()
        self.others_list.append(self.background_sprite);

    def on_draw(self) -> None:
        self.clear()
        self.others_list.draw()
        self.mouth_list.draw()
        self.eyes_list.draw()

    def on_update(self, delta_time: float) -> None:
        self.eyes.set_state(current_state)
        self.mouth.set_state(current_state)
        self.mouth_list.update_animation(delta_time)
        self.eyes_list.update_animation(delta_time)

    def on_resize(self, width: int, height: int) -> bool | None:
        self.draw_face()
        return super().on_resize(width, height)

    def on_key_press(self, key: int, modifiers: int) -> None:
        global current_state

        if key == arcade.key.F11:
            isFullScreen = not self.window.fullscreen;
            self.window.set_fullscreen(isFullScreen);

        if key == arcade.key.KEY_1:
            current_state = FaceState.IDLE
        elif key == arcade.key.KEY_2:
            current_state = FaceState.SPEAKING
        elif key == arcade.key.KEY_3:
            current_state = FaceState.THINKING
        elif key == arcade.key.KEY_4:
            current_state = FaceState.BORED

app = FastAPI()


@app.get("/state")
def get_state() -> dict[str, str]:
    return {"state": current_state.name}


@app.put("/state/{state}")
def set_state(state: str) -> dict[str, str]:
    global current_state
    key = state.upper()
    by_name = {s.name: s for s in FaceState}
    by_value = {str(s.value): s for s in FaceState}
    found = by_name.get(key) or by_value.get(key)
    if found is None:
        raise HTTPException(status_code=400, detail="Use 1-4 or IDLE, SPEAKING, THINKING, BORED")
    current_state = found
    return {"state": current_state.name}


def main():
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=HTTP_PORT, log_level="info"),
        daemon=True,
    ).start()

    window = arcade.Window(title=WINDOW_TITLE, resizable=True, center_window=True)
    window.background_color = BACKGROUND_COLOR
    game = GameView()
    window.show_view(game)
    arcade.run()


if __name__ == "__main__":
    main()
