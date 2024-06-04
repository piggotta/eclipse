import datetime
import gphoto2
import time

import camera_utils

TOTAL_ECLIPSE_DURATION = datetime.timedelta(minutes=8)
#TOTAL_ECLIPSE_START = datetime.datetime.now() + datetime.timedelta(seconds=30)
TOTAL_ECLIPSE_START = datetime.datetime.fromisoformat('2024-04-08T13:46:36')
TOTAL_ECLIPSE_STOP = TOTAL_ECLIPSE_START + TOTAL_ECLIPSE_DURATION
PARTIAL_ECLIPSE_STOP = TOTAL_ECLIPSE_STOP + datetime.timedelta(minutes=80)

EXPOSURE_INTERVAL = datetime.timedelta(seconds=15)
POLLING_INTERVAL_S = 0.01

DEFAULT_SETTINGS = {
  # Set exposure mode to 'Manual' as a check only. This will throw an error
  # when we try to set the config if the camera dial is not in 'Manual' mode.
  'autoexposuremode': 'Manual',
  'mirrorlock': '1',
  'capturetarget': 'Memory card',
  'imageformat': 'RAW',
  'drivemode': 'Single silent',
  'highisonr': 'Off',
  # F-number options are 5.6, 6.3, 6.1, 8, 9, 10, 11, 13, 14, 16, ...
  'aperture': '8',
}

def single_exposure(camera: camera_utils.Camera):
  camera.set_widgets({'iso': '100', 'aperture': '8'})
  camera.take_photo_with_mirrorlock(1/2000)

def exposure_sweep(camera: camera_utils.Camera):
  camera.set_widgets({'iso': '2000', 'aperture': '5.6'})
  for shutterspeed_s in [5, 0.6, 1/13, 1/100, 1/800]:
    camera.take_photo_with_mirrorlock(shutterspeed_s)

  camera.set_widgets({'iso': '500', 'aperture': '8'})
  camera.take_photo_with_mirrorlock(1/800)

  camera.set_widgets({'iso': '100', 'aperture': '8'})
  camera.take_photo_with_mirrorlock(1/2000)

def run_eclipse_acquisition(camera: camera_utils.Camera):
  deadline = datetime.datetime.now()
  while deadline < PARTIAL_ECLIPSE_STOP:
    now = datetime.datetime.now()
    now_str = now.isoformat().split('T')[1]
    if now > deadline:
      deadline += EXPOSURE_INTERVAL
      if now > TOTAL_ECLIPSE_START and now < TOTAL_ECLIPSE_STOP:
        print(f'{now_str}: exposure sweep')
        exposure_sweep(camera)
      else:
        print(f'{now_str}: single exposure')
        single_exposure(camera)
    time.sleep(POLLING_INTERVAL_S)

def main():
  camera = camera_utils.get_camera('Canon EOS 6D')
  camera.set_widgets(DEFAULT_SETTINGS)
  run_eclipse_acquisition(camera)
  camera.set_widget('mirrorlock', '0')
  camera.exit()

if __name__ == '__main__':
  main()
