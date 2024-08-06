import copy

import matplotlib.pyplot as plt
import numpy as np

import partial_eclipse_tracker


def main(plot_only: bool = False):
  options = partial_eclipse_tracker.PartialEclipseTrackerOptions(
      guess_sun_radius=100,
      # Moon movement rate estimate
      #  * Radius of sun is approximately 130 pixels
      #  * Takes approximately 9000 seconds for moon to cross sun
      #  * Moon moves from bottom-left corner of sun to top-left corner of sun
      guess_moon_dr_dt=-0.7 * 4 * 130 / 9000,
      guess_moon_dc_dt=0.7 * 4 * 130 / 9000,
  )
  tracker = partial_eclipse_tracker.PartialEclipseTracker(options)
  tracker.load_preprocess_from_file('preprocess_partials')

  if plot_only:
    track = partial_eclipse_tracker.load_track('partial_track')
  else:
    track = tracker.track_sun_and_moon()
    partial_eclipse_tracker.save_track('partial_track', track)

  # Print tracking results.
  print_track = copy.copy(track)
  print_track.sun_centers = None
  print_track.unix_time_s = None
  print('Final track:')
  print(print_track)

  # Plot final aligned images.
  ind_to_plot = np.asarray(
      np.linspace(0, tracker.is_sun.shape[0] - 1, 30),
      dtype=int
  )
  tracker.plot_track(track, ind_to_plot)

  plt.show()
  return


if __name__ == '__main__':
  main(plot_only=True)
