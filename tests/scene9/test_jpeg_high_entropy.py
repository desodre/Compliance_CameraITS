# Copyright 2024 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verifies JPEG still capture images are correct in the complex scene."""


import logging
import os.path

from mobly import test_runner
import numpy as np
import PIL

import its_base_test
import camera_properties_utils
import capture_request_utils
import image_processing_utils
import its_session_utils


_BUSY_SCENE_VARIANCE_ATOL = 0.01  # busy scenes variances > this for [0, 1] img
_JPEG_EXTENSION = '.jpg'
_JPEG_QUALITY_SETTING = 100  # set value to max
_NAME = os.path.splitext(os.path.basename(__file__))[0]
_NUM_STEPS = 8
_ZOOM_RATIO_MAX = 4  # too high zoom ratios will eventualy reduce entropy
_ZOOM_RATIO_MIN = 1  # low zoom ratios don't fill up FoV
_ZOOM_RATIO_THRESH = 2  # some zoom ratio needed to fill up FoV


def _read_files_back_from_disk(log_path):
  """Read the JPEG files written as part of test back from disk.

  Args:
    log_path: string; location to read files.

  Returns:
    list of uint8 images read with Image.read().
    jpeg_size_max: int; max size of jpeg files.
  """
  jpeg_files = []
  for file in sorted(os.listdir(log_path)):
    if _JPEG_EXTENSION in file:
      jpeg_files.append(file)
  if jpeg_files:
    logging.debug('JPEG files from directory: %s', jpeg_files)
  else:
    raise AssertionError(f'No JPEG files in {log_path}')
  for jpeg_file in jpeg_files:
    jpeg_file_with_log_path = os.path.join(log_path, jpeg_file)
    jpeg_file_size = os.stat(jpeg_file_with_log_path).st_size
    logging.debug('Opening file %s', jpeg_file)
    logging.debug('File size %d (bytes)', jpeg_file_size)
    try:
      image_processing_utils.convert_image_to_numpy_array(
          jpeg_file_with_log_path)
    except PIL.UnidentifiedImageError as e:
      raise AssertionError(f'Cannot read {jpeg_file_with_log_path}') from e
    logging.debug('Successfully read %s.', jpeg_file)


class JpegHighEntropyTest(its_base_test.ItsBaseTest):
  """Tests JPEG still capture with a complex scene.

  Steps zoom ratio to ensure the complex scene fills the camera FoV.
  """

  def test_jpeg_high_entropy(self):
    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:
      props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(props)
      log_path = self.log_path
      test_name_with_log_path = os.path.join(log_path, _NAME)

      # Load chart for scene
      its_session_utils.load_scene(
          cam, props, self.scene, self.tablet,
          its_session_utils.CHART_DISTANCE_NO_SCALING)

      # Determine test zoom range
      zoom_range = props['android.control.zoomRatioRange']
      zoom_min, zoom_max = float(zoom_range[0]), float(zoom_range[1])
      logging.debug('Zoom max value: %.2f', zoom_max)
      if zoom_min == zoom_max:
        zoom_ratios = [zoom_min]
      else:
        zoom_max = min(zoom_max, _ZOOM_RATIO_MAX)
        zoom_ratios = np.arange(
            _ZOOM_RATIO_MIN, zoom_max,
            (zoom_max - _ZOOM_RATIO_MIN) / (_NUM_STEPS - 1)
        )
        zoom_ratios = np.append(zoom_ratios, zoom_max)
      logging.debug('Testing zoom range: %s', zoom_ratios)

      # Do captures over zoom range
      req = capture_request_utils.auto_capture_request()
      req['android.jpeg.quality'] = _JPEG_QUALITY_SETTING
      out_surface = capture_request_utils.get_largest_jpeg_format(props)
      logging.debug('req W: %d, H: %d',
                    out_surface['width'], out_surface['height'])

      for zoom_ratio in zoom_ratios:
        req['android.control.zoomRatio'] = zoom_ratio
        logging.debug('zoom ratio: %.3f', zoom_ratio)
        cam.do_3a(zoom_ratio=zoom_ratio)
        cap = cam.do_capture(req, out_surface)

        # Save JPEG image
        try:
          img = image_processing_utils.convert_capture_to_rgb_image(
              cap, props=props)
        except PIL.UnidentifiedImageError as e:
          raise AssertionError(
              f'Cannot convert cap to JPEG for zoom: {zoom_ratio:.2f}') from e
        logging.debug('cap size (pixels): %d', img.shape[1]*img.shape[0])
        image_processing_utils.write_image(
            img, f'{test_name_with_log_path}_{zoom_ratio:.2f}{_JPEG_EXTENSION}')

        r_var, b_var, g_var = image_processing_utils.compute_image_variances(
            img
        )
        logging.debug('img vars: %.4f, %.4f, %.4f', r_var, g_var, b_var)
        if max(r_var, g_var, b_var) < _BUSY_SCENE_VARIANCE_ATOL:
          raise AssertionError(
              'Scene is not busy enough! Measured RGB variances: '
              f'{r_var:.4f}, {g_var:.4f}, {b_var:.4f}, '
              f'ATOL: {_BUSY_SCENE_VARIANCE_ATOL}'
          )

      # Read JPEG files back to ensure readable encoding
      _read_files_back_from_disk(log_path)

if __name__ == '__main__':
  test_runner.main()
