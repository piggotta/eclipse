from collections.abc import Sequence
import os

import numpy as np

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
  grey_images = image_loader.maybe_read_images_by_index(
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
  actual_ratios = {}
  expected_ratios = [[] for _ in range(num_ratios)]
  for colour in image_loader.BAYER_MASK_OFFSET:
    actual_ratios[colour] = [[] for _ in range(num_ratios)]

    offset = image_loader.BAYER_MASK_OFFSET[colour]
    print(f'Bayer mask: {colour} {offset}')

    for stack_num, stack in enumerate(stacks):
      images = [image.raw_image[offset[0]::2, offset[1]::2]
                for image in stack]
      invalid_pixels = []

      invalid = []
      for image in images:
        num_pixels = np.prod(image.shape)

        # Pixels that are too saturated or too dark are considered invalid.
        current_invalid_pixels = np.logical_or(image > 10e3, image < 40)
        invalid_pixels.append(current_invalid_pixels)

        # Do not use images that have too many invalid pixels.
        if np.sum(current_invalid_pixels) / num_pixels > 0.7:
          current_invalid = True
        else:
          current_invalid = False
        invalid.append(current_invalid)

      print(f'  Stack {stack_num:d}:')
      for ind in range(len(stack) - 1):
        if invalid[ind] or invalid[ind + 1]:
          continue
        actual_ratio = images[ind] / images[ind + 1]
        actual_ratio[invalid_pixels[ind]] = np.nan
        actual_ratio[invalid_pixels[ind + 1]] = np.nan
        actual_ratios[colour][ind].append(np.nanmean(actual_ratio))

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
  for ind in range(len(expected_ratios)):
    reports = []
    for colour in image_loader.BAYER_MASK_OFFSET:
      reports.append(
          f'{np.mean(actual_ratios[colour][ind]):5.2f} +/- '
          f'{np.std(actual_ratios[colour][ind]):4.2f} ({colour})'
      )
    report = ', '.join(reports)
    print(
        f'  Image {ind:d} vs {ind + 1:d}: '
        f'{expected_ratios[ind]:5.2f} (expected), ' + report
    )
  print()


def main():
  print('== Processing black frames ==')
  processor = raw_processor.RawProcessor()
  black_frames = image_loader.maybe_read_images_by_index(
      range(constants.IND_FIRST_BLACK, constants.IND_LAST_BLACK)
  )
  processor.process_black_frames(black_frames)
  processor.save_calibration_to_file('calibration')
  processor.load_calibration_from_file('calibration')

  analyze_black_frames(processor)
  analyze_grey_frames(processor)

if __name__ == '__main__':
  main()
