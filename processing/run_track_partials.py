import matplotlib.pyplot as plt
import numpy as np

import partial_eclipse_tracker


def main():
  tracker = partial_eclipse_tracker.PartialEclipseTracker()
  tracker.load_preprocess_from_file('preprocess_partials')
  tracker.fit_sun_radius()


if __name__ == '__main__':
  main()
