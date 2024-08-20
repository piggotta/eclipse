import os

import constants

def metadata(filename):
  return os.path.join(constants.OUTPUTS_PATH, filename + '.cattr')


def image_stack(filename):
  return os.path.join(constants.OUTPUTS_PATH, filename + '.npz')


