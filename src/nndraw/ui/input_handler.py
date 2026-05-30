import pygame

from nndraw.ui.config import CanvasConfig
from nndraw.ui.point_request import PointRequest
from nndraw.ui.label import Label

_config = CanvasConfig()

def parse_event(event: pygame.Event) -> PointRequest | None:
    if event.type != pygame.MOUSEBUTTONDOWN:
        return None
    x, y = event.pos
    if event.button == _config.left_btn:
        return PointRequest(x, y, Label.PURPLE)
    if event.button == _config.right_btn:
        return PointRequest(x, y, Label.GREEN)
    return None