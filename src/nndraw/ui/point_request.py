from dataclasses import dataclass

from nndraw.ui.label import Label


@dataclass(frozen=True)
class PointRequest:
    x: int
    y: int
    label: Label