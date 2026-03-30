from typing import Any

import pygame

from nndraw.linalg.vector import Vector
from nndraw.nn.network import Network
from nndraw.nn.activations import sigmoid, sigmoid_derivative

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FPS = 60
LEFT_BTN = 1
RIGHT_BTN = 3
PURPLE = (180, 130, 220)
GREEN = (130, 210, 160)
GRID_SIZE = 10

class Canvas:
    def __init__(self):
        self._points = []
        self._network = Network([2, 6, 1], sigmoid, sigmoid_derivative)

    def run(self):
        pygame.init()
        pygame.display.set_caption("nndraw")
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                self._add_vector(event)

                if event.type == pygame.QUIT:
                    running = False

            screen.fill("#181818ff")

            # RENDER GAME HERE
            self._train()
            self._draw_background(screen)
            for p in self._points:
                self._draw_circle(screen, p)
            pygame.display.flip()

            clock.tick(FPS)

        pygame.quit()

    def _draw_background(self, screen: pygame.Surface) -> None:
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
                normalized_input = self._normalize_input(x, y)
                output = self._network.predict(Vector(normalized_input))
                color = self._lerp_color(output)
                pygame.draw.rect(screen, color, (x, y, GRID_SIZE, GRID_SIZE))

    def _add_vector(self, event: pygame.Event) -> None:
        label = None
        is_mouse_btn_down = event.type == pygame.MOUSEBUTTONDOWN
        is_left_click = is_mouse_btn_down and event.button == LEFT_BTN
        is_right_click = is_mouse_btn_down and event.button == RIGHT_BTN
        if is_left_click:                    
            print(event.pos)
            x, y = event.pos
            label = 0            
            normalized_input = self._normalize_input(x, y)
            v = Vector(normalized_input);
            self._points.append((v, label));
        elif is_right_click:
            print(event.pos)
            x, y = event.pos
            label = 1
            normalized_input = self._normalize_input(x, y)
            v = Vector(normalized_input);
            self._points.append((v, label));

    def _draw_circle(
            self, 
            surface: pygame.Surface, 
            point: tuple[Vector, int]
        ) -> None:
        v, label = point
        x, y = v
        color = ()
        if label == 0:
            color = PURPLE
        elif label == 1:
            color = GREEN
        x, y = self._denormalize_input(v)   
        pygame.draw.circle(surface, color, (x, y), 6)

    def _lerp_color(self, output: Vector) -> tuple[int, int, int]:
        r = PURPLE[0] + (GREEN[0] - PURPLE[0]) * output[0]
        g = PURPLE[1] + (GREEN[1] - PURPLE[1]) * output[0]
        b = PURPLE[2] + (GREEN[2] - PURPLE[2]) * output[0]
        return (int(r), int(g), int(b))
    
    def _normalize_input(self, x: int, y: int) -> list[float]:
        nx = x / WINDOW_WIDTH
        ny = y / WINDOW_HEIGHT
        return [nx, ny]
    
    def _denormalize_input(self, input: Vector) -> tuple[float, float]:
        x = input[0] * WINDOW_WIDTH
        y = input[1] * WINDOW_HEIGHT
        return (x, y)

    def _train(self) -> None:
        for v, label in self._points:
            target = Vector([float(label)])
            self._network.train(v, target, learning_rate=0.1)
