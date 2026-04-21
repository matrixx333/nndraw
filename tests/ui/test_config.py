from nndraw.ui.config import CanvasConfig

def test_canvas_config_grid_size():
    config = CanvasConfig()
    assert(config.grid_size == 25)
