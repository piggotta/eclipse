import numpy as np
import matplotlib.pyplot as plt

import constants
import eclipse_image_loader
import eclipse_images
import filepaths
import raw_processor

def stack_single_total(processor: raw_processor.RawProcessor,
                       totals_index: int) -> float:
  print('Read total eclipse image attributes...')
  num_photos_to_check = (totals_index + 1) * len(constants.EXPOSURES)
  attributes = eclipse_image_loader.maybe_read_images_attributes_by_index(
      range(
          constants.IND_FIRST_TOTAL,
          constants.IND_FIRST_TOTAL + num_photos_to_check + 1
      )
  )

  print('Generating HDR total eclipse image...')
  stack = raw_processor.split_into_exposure_stacks(attributes)[totals_index]

  metadata = stack[-1]
  index = metadata.index
  unix_time_s = metadata.time.timestamp()
  npz_filepath = filepaths.hdr_total(index)
  print('  ' + npz_filepath)

  indices = [entry.index for entry in stack]
  images = eclipse_image_loader.maybe_read_images_by_index(indices, verbose=False)
  hdr_image = processor.stack_hdr_image(images, smoothing_radius=5)

  np.savez(npz_filepath,
           index=index,
           unix_time_s=unix_time_s,
           raw=hdr_image
  )
  return unix_time_s


def main(restack: bool = True):
  totals_index = 2

  # Load raw image processor configuration from file.
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  # Load full eclipse image sequence from file.
  sequence = eclipse_images.load_sequence('full_sequence')

  # Find unix time of the total eclipse image to render.
  totals_unix_time_s = [entry.unix_time_s for entry in sequence.image_metadata
                        if entry.image_type == eclipse_images.ImageType.HDR]
  unix_time_s = totals_unix_time_s[totals_index]

  # Create renderer object.
  options = eclipse_images.RendererOptions(
      shape=constants.RENDER_SHAPE,
      bright_corona_deg_lims=constants.RENDER_BRIGHT_CORONA_DEG_LIMS,
      dark_corona_deg_lims=constants.RENDER_DARK_CORONA_DEG_LIMS,
  )
  renderer = eclipse_images.Renderer(processor, sequence, options)

  # Re-stack a single image.
  if restack and stack_single_total(processor, totals_index) != unix_time_s:
    raise ValueError('Incorrect Unix time for the image we re-stacked.')

  # Render image.
  renderer.render_total(unix_time_s, show_plots=True)

  plt.show()

if __name__ == '__main__':
  main(restack=False)
