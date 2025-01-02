import numpy as np
import matplotlib.pyplot as plt

import constants
import eclipse_images
import filepaths
import raw_processor


def main():
  # Load raw image processor configuration from file.
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  # Load full eclipse image sequence from file
  sequence = eclipse_images.load_sequence('full_sequence')

  # Create renderer object.
  options = eclipse_images.RendererOptions(
      shape=constants.RENDER_SHAPE,
      bright_corona_deg_lims=constants.RENDER_BRIGHT_CORONA_DEG_LIMS,
      dark_corona_deg_lims=constants.RENDER_DARK_CORONA_DEG_LIMS,
  )
  renderer = eclipse_images.Renderer(processor, sequence, options)

  # Find times of all total eclipse images.
  totals_unix_time_s = [entry.unix_time_s for entry in sequence.image_metadata
                        if entry.image_type == eclipse_images.ImageType.HDR]

  # Fit corona from the middle of the total eclipse sequence.
  renderer.fit_corona(totals_unix_time_s[5:-5], show_plots=True)

  plt.show()


if __name__ == '__main__':
  main()
