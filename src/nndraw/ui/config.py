from dataclasses import dataclass

@dataclass(frozen=True)
class CanvasConfig:
    width: int = 1600
    height: int = 900
    fps: int = 60
    left_btn: int = 1
    right_btn: int = 3
    purple: tuple = (180, 130, 220)
    green: tuple = (130, 210, 160)
    grid_size: int = 25
    learning_rate: float = 0.05
    hidden_size: int = 6