import matplotlib.pyplot as plt
import numpy as np

import constants
import eclipse_images
import eclipse_tracker
import filepaths
import image_loader
import partial_eclipse_tracker
import raw_processor


def main():
  total_ind = 10

  # Load raw image processor configuration from file.
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  print('Read total eclipse image attributes...')
  attributes = image_loader.maybe_read_images_attributes_by_index(
      range(constants.IND_FIRST_TOTAL, constants.IND_LAST_TOTAL + 1)
  )

  # Stack and HDR image.
  stack = raw_processor.split_into_exposure_stacks(attributes)[total_ind]
  metadata = stack[0]
  index = metadata.index
  unix_time_s = metadata.time.timestamp()
  npz_filepath = filepaths.hdr_total(index)
  print('Generating a single HDR total eclipse image: ' + npz_filepath)

  indices = [entry.index for entry in stack]
  images = image_loader.maybe_read_images_by_index(indices, verbose=False)
  hdr_image = processor.stack_hdr_image(images, smoothing_radius=30)
  np.savez(npz_filepath,
           index=index,
           unix_time_s=unix_time_s,
           raw=hdr_image
  )

  # Load full eclipse image sequence from file
  sequence = eclipse_images.load_sequence('full_sequence')

  # Create renderer object.
  renderer = eclipse_images.Renderer(processor, sequence)

  # Analyze HDR image.
  renderer.analyze_total(unix_time_s)

  plt.show()

if __name__ == '__main__':
  main()
