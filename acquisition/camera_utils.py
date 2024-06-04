from typing import Any

import gphoto2
import time

class Camera(gphoto2.Camera):
  shutterspeed_s: float | None

  def __init__(self):
    super().__init__()
    self.shutterspeed_s = None

  def set_widgets(self, widget_values: dict[str, Any]):
    config = self.get_config()
    for name, value in widget_values.items():
      config.get_child_by_name(name).set_value(value)
    self.set_config(config)

  def set_widget(self, name: str, value: Any):
    self.set_widgets({name: value})

  def read_widget(self, name: str) -> Any:
    config = self.get_config()
    return config.get_child_by_name(name).get_value()

  def set_shutterspeed(self, shutterspeed_s: float):
    if shutterspeed_s == self.shutterspeed_s:
      return
    self.shutterspeed_s = shutterspeed_s
    if shutterspeed_s < 0.29:
      shutterspeed_str = f'1/{int(round(1/shutterspeed_s)):d}'
    else:
      shutterspeed_10x = int(round(10 * shutterspeed_s))
      shutterspeed_int = shutterspeed_10x // 10
      shutterspeed_decimal = shutterspeed_10x % 10
      shutterspeed_str = f'{shutterspeed_int:d}'
      if shutterspeed_decimal != 0:
        shutterspeed_str += f'.{shutterspeed_decimal:d}'
    self.set_widget('shutterspeed', shutterspeed_str)

  def take_photo(self, shutterspeed_s: float):
    self.set_shutterspeed(shutterspeed_s)

    # Press camera trigger once to take the photo.
    self.set_widget('eosremoterelease', 'Press Full')
    self.set_widget('eosremoterelease', 'Release Full')
    time.sleep(0.6 + 1.1 *shutterspeed_s)

  def take_photo_with_mirrorlock(self, shutterspeed_s: float):
    self.set_shutterspeed(shutterspeed_s)

    # Press camera trigger once to trigger mirror lift.
    self.set_widget('eosremoterelease', 'Press Full')
    self.set_widget('eosremoterelease', 'Release Full')
    time.sleep(0.2)

    # After camera settles, press camera trigger again to take the photo.
    self.set_widget('eosremoterelease', 'Press Full')
    self.set_widget('eosremoterelease', 'Release Full')
    time.sleep(0.6 + 1.1 *shutterspeed_s)

def get_camera(camera_name: str) -> Camera:
  port_info_list = gphoto2.PortInfoList()
  port_info_list.load()
  abilities_list = gphoto2.CameraAbilitiesList()
  abilities_list.load()

  camera_list = list(gphoto2.Camera.autodetect())
  for name, addr in camera_list:
    if name == camera_name:
      camera = Camera()
      idx = port_info_list.lookup_path(addr)
      camera.set_port_info(port_info_list[idx])
      idx = abilities_list.lookup_model(name)
      camera.set_abilities(abilities_list[idx])
      return camera
  raise RuntimeError('No matching cameras found')

def available_camera_names() -> list[str]:
  camera_list = list(gphoto2.Camera.autodetect())
  return [name for name, _ in camera_list]

