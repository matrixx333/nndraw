from typing import Any

import pygame
import threading
import time

from nndraw.linalg.vector import Vector
from nndraw.nn.activation import Activation
from nndraw.nn.network import Network
from nndraw.nn.activations import (
    sigmoid, 
    sigmoid_derivative
)
from nndraw.ui.config import CanvasConfig
from nndraw.db.point_store import PointStore
from nndraw.ui.input_handler import parse_event
from nndraw.ui.coords import normalize, denormalize
from nndraw.ui.label import Label

_config = CanvasConfig()

LABEL_COLORS = {
    Label.PURPLE: _config.purple,
    Label.GREEN: _config.green
}

SIGMOID = Activation(sigmoid, sigmoid_derivative)

class Canvas:
    def __init__(self):
        self._point_store = PointStore()
        self._points = []
        self._network = Network(
            [2, _config.hidden_size, 1],
            SIGMOID
        )
        self._lock = threading.Lock()

    def run(self):
        pygame.init()
        pygame.display.set_caption("nndraw")
        screen = pygame.display.set_mode(
            (_config.width, _config.height)
        )
        clock = pygame.time.Clock()
        thread = threading.Thread(target=self._training_loop, daemon=True)
        thread.start()
        threading.Thread(target=self._load_points, daemon=True).start()
        running = True

        while running:
            for event in pygame.event.get():
                self._add_vector(event)

                if event.type == pygame.QUIT:
                    running = False

            screen.fill("#181818ff")

            # RENDER GAME HERE
            self._draw_background(screen)
            with self._lock:
                for p in self._points:
                    self._draw_circle(screen, p)
            pygame.display.flip()

            clock.tick(_config.fps)

        pygame.quit()

    def _draw_background(self, screen: pygame.Surface) -> None:
        for x in range(0, _config.width, _config.grid_size):
            for y in range(0, _config.height, _config.grid_size):
                normalized_input = normalize(x, y)
                output = self._network.predict(Vector(normalized_input))
                color = self._lerp_color(output)
                pygame.draw.rect(
                    screen, 
                    color, 
                    (x, y, _config.grid_size, _config.grid_size)
                )

    def _add_vector(self, event: pygame.Event) -> None:
        request = parse_event(event)
        if request is not None:
            normalized_input = normalize(request.x, request.y)
            v = Vector(normalized_input);
            self._point_store.add_point(v, request.label)
            with self._lock:
                self._points.append((v, request.label));

    def _draw_circle(
            self, 
            surface: pygame.Surface,
            point: tuple[Vector, Label]
        ) -> None:
        v, label = point
        x, y = v
        color = LABEL_COLORS[label]
        x, y = denormalize(v)
        pygame.draw.circle(surface, color, (x, y), 6)
        pygame.draw.circle(surface, (255, 255, 255), (x, y), 6, 2)

    def _lerp_color(self, output: Vector) -> tuple[int, int, int]:
        r = _config.purple[0] + (_config.green[0] - _config.purple[0]) * output[0]
        g = _config.purple[1] + (_config.green[1] - _config.purple[1]) * output[0]
        b = _config.purple[2] + (_config.green[2] - _config.purple[2]) * output[0]
        return (int(r), int(g), int(b))

    def _train(self) -> None:
        with self._lock:
            points_snapshot = list(self._points)
        for v, label in points_snapshot:
            target = Vector([float(label)])
            self._network.train(v, target, learning_rate=_config.learning_rate)

    def _training_loop(self) -> None:
        while True:
            self._train()
            time.sleep(0.001)

    def _load_points(self) -> None:
        points = self._point_store.get_all()
        points = [(v, Label(label)) for v, label in points]
        with self._lock:
            self._points.extend(points)