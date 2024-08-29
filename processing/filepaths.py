import os
import pathlib

import constants


def _make_recursive_dir_if_not_exist(path: str):
  dirs = pathlib.Path(path).parts
  for ind in range(len(dirs)):
    current_path = os.path.join(*dirs[:ind + 1])
    if not os.path.isdir(current_path):
      os.mkdir(current_path)


def _make_folder_and_return_path(folder: str, filename: str):
  _make_recursive_dir_if_not_exist(folder)
  return os.path.join(folder, filename)


def metadata(filename: str) -> str:
  return _make_folder_and_return_path(constants.OUTPUTS_PATH,
                                      filename + '.cattr')

def calibration(filename: str) -> str:
  return _make_folder_and_return_path(constants.OUTPUTS_PATH,
                                      filename + '.npz')


def cropped_partials(filename: str) -> str:
  return _make_folder_and_return_path(constants.OUTPUTS_PATH,
                                      filename + '.npz')


def corrected_raw(index: int) -> str:
  return _make_folder_and_return_path(constants.CORRECTED_RAWS_PATH,
                                      f'corrected_raw_{index:04d}.npz')
