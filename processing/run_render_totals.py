import matplotlib.pyplot as plt
import numpy as np

import constants
import eclipse_images
import eclipse_tracker
import filepaths
import image_loader
import partial_eclipse_tracker
import raw_processor
import util


def main(show_plots: bool = False):
  util.print_title('Render total eclipse images')

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

  # Find total eclipse images.
  totals_unix_time_s = [entry.unix_time_s for entry in sequence.image_metadata
                        if entry.image_type == eclipse_images.ImageType.HDR]

  # Render images.
  for ind in 1, 10, -2:
    renderer.render_total(totals_unix_time_s[ind], show_plots=show_plots)

  if show_plots:
    plt.show()

if __name__ == '__main__':
  main(show_plots=True)
