from collections.abc import Sequence
import os

import numpy as np
import matplotlib.pyplot as plt

import constants
import image_loader
import raw_processor


def create_hdr_image(processor: raw_processor.RawProcessor()):
  images = image_loader.maybe_load_images_by_index(range(1350, 1357))
  hdr_image = processor.stack_hdr_image(images)

  # Get a single-color sub-image.
  offset = image_loader.BAYER_MASK_OFFSET['red']
  bw_image = hdr_image[offset[0]::2, offset[1]::2]

  plt.figure()
  plt.imshow(bw_image, cmap='inferno')
  plt.colorbar()
  plt.title('Linear scale HDR image')

  plt.figure()
  plt.imshow(10 * np.log10(bw_image), cmap='inferno')
  plt.colorbar()
  plt.title('Log scale HDR image (dB)')


def main():
  print('== Processing black frames ==')
  processor = raw_processor.RawProcessor()
  black_frames = image_loader.maybe_load_images_by_index(
      range(constants.IND_FIRST_BLACK, constants.IND_LAST_BLACK)
  )
  processor.process_black_frames(black_frames)
  processor.save_calibration_to_file('calibration')
  processor.load_calibration_from_file('calibration')

  create_hdr_image(processor)
  plt.show()

if __name__ == '__main__':
  main()
