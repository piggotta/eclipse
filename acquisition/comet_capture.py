import datetime
import gphoto2
import time

import camera_utils

DEFAULT_SETTINGS = {
  # Set exposure mode to 'Manual' as a check only. This will throw an error
  # when we try to set the config if the camera dial is not in 'Manual' mode.
  'autoexposuremode': 'Manual',
  'capturetarget': 'Memory card',
  'imageformat': 'RAW',
  'drivemode': 'Single',
  'highisonr': 'Off',
  # F number options are 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, ...
  'aperture': '5',
}

def exposure_sweep(camera: camera_utils.Camera):
  camera.set_widget('viewfinder', 1)
  time.sleep(1)

  camera.set_widgets({'iso': '3200', 'aperture': '2.8'})
  for shutterspeed_s in [5, 0.6, 1/13, 1/100, 1/800]:
    camera.take_photo(shutterspeed_s)
    time.sleep(0.5)

  camera.set_widgets({'iso': '800', 'aperture': '5'})
  camera.take_photo(1/800)
  time.sleep(0.5)

  camera.set_widgets({'iso': '400', 'aperture': '8'})
  camera.take_photo(1/800)
  time.sleep(0.5)

  camera.set_widgets({'iso': '100', 'aperture': '8'})
  camera.take_photo(1/2000)
  time.sleep(0.5)

  camera.set_widget('viewfinder', 0)

def main():
  camera = camera_utils.get_camera('Canon EOS 700D')
  camera.set_widgets(DEFAULT_SETTINGS)
  exposure_sweep(camera)
  camera.exit()

if __name__ == '__main__':
  main()
