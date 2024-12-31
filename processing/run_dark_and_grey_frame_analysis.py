from collections.abc import Sequence
import os

import numpy as np

import constants
import eclipse_image_loader
import raw_processor


def analyze_black_frames(processor: raw_processor.RawProcessor()):
  print('== Processing black frames ==')
  black_frames = eclipse_image_loader.maybe_read_images_by_index(
      range(constants.IND_FIRST_BLACK, constants.IND_LAST_BLACK)
  )
  processor.process_black_frames(black_frames)

  print('== Black frame summary ==')
  for ind, exposure in enumerate(constants.EXPOSURES):
    background = processor.background[ind]
    stdev = processor.stdev[ind]

    print(
        f'{exposure.exposure_s:.4f} s, '
        f'f/{exposure.f_number:.2f}, ISO {exposure.iso:d}: '
    )
    print(f'  Mean: {np.mean(background):.2f} +/- {np.std(background):.2f}')
    print(f'  Std: {np.mean(stdev):.2f} +/- {np.std(stdev):.2f}')
    print()


def analyze_grey_frames(processor: raw_processor.RawProcessor()):
  print('== Processing grey frames  ==')
  grey_images = eclipse_image_loader.maybe_read_images_by_index(
      range(constants.IND_FIRST_TOTAL + len(constants.EXPOSURES),
            constants.IND_FIRST_TOTAL + 11 * len(constants.EXPOSURES))
  )
  processor.process_grey_frames(grey_images, verbose=True)

  print('== Grey frame summary ==')
  print()
  print('      Configuration            '
        'Nominal exposure (s)     '
        'Actual exposure (s)')
  for ind, exposure in enumerate(constants.EXPOSURES):
    norm_exposure_s = exposure.norm_exposure_s()
    actual_exposure_s = processor.effective_exposures_s[ind]
    print(
        f'{exposure.exposure_s:6.4f} s, '
        f'f/{exposure.f_number:5.2f}, ISO {exposure.iso:4d}        '
        f'{norm_exposure_s:.3e} s             '
        f'{actual_exposure_s:.3e} s'
    )
  print()

def main():
  processor = raw_processor.RawProcessor()
  analyze_black_frames(processor)
  analyze_grey_frames(processor)
  processor.save_calibration_to_file('calibration')
  processor.load_calibration_from_file('calibration')


if __name__ == '__main__':
  main()
