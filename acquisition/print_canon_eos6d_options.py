import gphoto2

import camera_utils

def get_widget_tree(widget) -> str:
  def add_indents(lines: str) -> str:
    return '\n'.join(['  ' + line for line in lines.split('\n')])

  tree_description = f'{widget.get_name()}: {widget.get_label()}'
  for child in widget.get_children():
    tree_description += '\n' + add_indents(get_widget_tree(child))
  return tree_description

def main():
  camera = camera_utils.get_camera('Canon EOS 6D')
  config = camera.get_config()

  print('=== Camera summary ===')
  print()
  print(camera.get_summary())
  print()

  print('=== Available widgets ===')
  print()
  print(get_widget_tree(config))
  print()

  print('=== Commonly used widgets ===')
  print()
  for name in ['capturetarget', 'drivemode', 'mirrorlock',
               'eosremoterelease', 'imageformat',
               'autoexposuremode', 'autoexposuremodedial',
               'highisonr', 'shutterspeed', 'aperture', 'iso']:
    child = config.get_child_by_name(name)
    print(f"{name} = '{child.get_value()}'")
    print('Available:', [choice for choice in child.get_choices()])
    print()

  camera.exit()
  return

if __name__ == '__main__':
  main()
