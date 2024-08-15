from collections.abc import Sequence
import os

import numpy as np
import matplotlib.pyplot as plt

import constants
import image_loader
import raw_processor


def analyze_black_frames(processor: raw_processor.RawProcessor()):
  print('== Black frame analysis ==')
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
  print('= Analyzing grey frames  ==')
  grey_images = image_loader.maybe_load_images_by_index(
      range(constants.IND_FIRST_GREY, constants.IND_LAST_GREY + 1)
  )

  # Perform black subtraction and hot pixel removal.
  grey_images = [
      processor.remove_hot_pixels(processor.subtract_background(image))
      for image in grey_images
  ]

  # Split images into exposure stacks.
  stacks = raw_processor.split_into_exposure_stacks(grey_images)

  print('Comparing expected vs actual ratios in pixel values for '
        'different exposures...')
  print()
  num_ratios = int(np.max([len(stack) for stack in stacks])) - 1
  actual_ratios = [[] for _ in range(num_ratios)]
  expected_ratios = [[] for _ in range(num_ratios)]
  for colour in image_loader.BAYER_MASK_OFFSET:
    offset = image_loader.BAYER_MASK_OFFSET[colour]
    print(f'Bayer mask: {colour} {offset}')

    for stack_num, stack in enumerate(stacks):
      images = [image.raw_image[offset[0]::2, offset[1]::2]
                for image in stack]

      invalid = []
      for image in images:
        num_pixels = np.prod(image.shape)
        current_invalid = False

        # Check for image saturation.
        if np.sum(image > 12e3) / num_pixels > 0.2:
          current_invalid = True

        # Check if image is essentially completely black.
        if np.sum(image < 20) / num_pixels > 0.2:
          current_invalid = True

        invalid.append(current_invalid)

      print(f'  Stack {stack_num:d}:')
      for ind in range(len(stack) - 1):
        if invalid[ind] or invalid[ind + 1]:
          continue
        actual_ratio = images[ind] / images[ind + 1]
        invalid_pixels = np.logical_or(
            np.logical_not(np.isfinite(actual_ratio)),
            np.abs(actual_ratio) > 100
        )
        actual_ratio[invalid_pixels] = np.nan
        actual_ratios[ind].append(np.nanmean(actual_ratio))

        expected_ratio = (stack[ind].get_exposure().norm_exposure_s() /
                          stack[ind + 1].get_exposure().norm_exposure_s())
        expected_ratios[ind] = expected_ratio

        print(
            f'    Image {ind:d} vs {ind + 1:d}: '
            f'{expected_ratio:5.2f} (expected), '
            f'{np.nanmean(actual_ratio):5.2f} +/- '
            f'{np.nanstd(actual_ratio):4.2f} (actual)'
        )
      print()

  print('Summary of ratios:')
  for ind in range(len(actual_ratios)):
    print(
        f'  Image {ind:d} vs {ind + 1:d}: '
        f'{expected_ratios[ind]:5.2f} (expected), '
        f'{np.mean(actual_ratios[ind]):5.2f} +/- '
        f'{np.std(actual_ratios[ind]):4.2f} (actual)'
    )
  print()


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

  analyze_black_frames(processor)
  analyze_grey_frames(processor)
  create_hdr_image(processor)
  plt.show()

if __name__ == '__main__':
  main()
