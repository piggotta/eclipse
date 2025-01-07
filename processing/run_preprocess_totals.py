import matplotlib.pyplot as plt

import eclipse_images
import eclipse_tracker
import partial_eclipse_tracker
import raw_processor
import util


def main(show_plots: bool = False):
  util.print_title('Preprocess total eclipse images')

  # Load raw image processor configuration from file.
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  # Load partial eclipse tracks from file.
  partials_preprocessor = partial_eclipse_tracker.PartialEclipsePreprocessor()
  partials_preprocessor.load_from_file('preprocess_partials')
  partials_track = eclipse_tracker.load_track('partial_track')

  # Stack total eclipse HDR images and align to known eclipse track.
  totals_processor = eclipse_images.TotalEclipseProcessor(
      processor, partials_preprocessor, partials_track)
  totals_processor.stack_and_track_totals()

  # Build full eclipse image sequence.
  sequence = totals_processor.build_sequence()
  eclipse_images.save_sequence('full_sequence', sequence)
  new_sequence = eclipse_images.load_sequence('full_sequence')

  if show_plots:
    totals_processor.plot_track()
    plt.show()

if __name__ == '__main__':
  main(show_plots=True)
