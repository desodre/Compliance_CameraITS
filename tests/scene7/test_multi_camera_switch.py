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
"""Verify that the switch from UW to W has similar RGB values."""


import logging
import math
import os.path
import pathlib

from mobly import test_runner
import numpy as np

import its_base_test
import camera_properties_utils
import image_processing_utils
import its_session_utils
import opencv_processing_utils
import preview_processing_utils


_AE_ATOL = 4.0
_AE_RTOL = 0.04  # 4%
_AF_ATOL = 0.02  # 2%
_ARUCO_MARKERS_COUNT = 4
_AWB_ATOL = 0.02  # 2%
_CH_FULL_SCALE = 255
_COLORS = ('r', 'g', 'b', 'gray')
_IMG_FORMAT = 'png'
_NAME = os.path.splitext(os.path.basename(__file__))[0]
_PATCH_MARGIN = 50  # pixels
_RECORDING_DURATION = 400  # milliseconds
_SENSOR_ORIENTATIONS = (90, 270)
_SKIP_INITIAL_FRAMES = 15
_TAP_COORDINATES = (500, 500)  # Location to tap tablet screen via adb
_ZOOM_RANGE_UW_W = (0.95, 2.05)  # UW/W crossover range
_ZOOM_STEP = 0.01


def _get_error_msg(failed_awb_msg, failed_ae_msg, failed_af_msg):
  """"Returns the error message string.

  Args:
    failed_awb_msg: list of awb error msgs
    failed_ae_msg: list of ae error msgs
    failed_af_msg: list of af error msgs
  Returns:
    error_msg: str; error_msg string
  """
  error_msg = ''
  if failed_awb_msg:
    error_msg = f'{error_msg}----AWB Check----\n'
    for msg in failed_awb_msg:
      error_msg = f'{error_msg}{msg}\n'
  if failed_ae_msg:
    error_msg = f'{error_msg}----AE Check----\n'
    for msg in failed_ae_msg:
      error_msg = f'{error_msg}{msg}\n'
  if failed_af_msg:
    error_msg = f'{error_msg}----AF Check----\n'
    for msg in failed_af_msg:
      error_msg = f'{error_msg}{msg}\n'
  return error_msg


def _check_orientation_and_flip(props, uw_img, w_img, img_name_stem):
  """Checks the sensor orientation and flips image.

  The preview stream captures are flipped based on the sensor
  orientation while using the front camera. In such cases, check the
  sensor orientation and flip the image if needed.

  Args:
    props: camera properties object.
    uw_img: image captured using UW lens.
    w_img: image captured using W lens.
    img_name_stem: prefix for the img name to be saved

  Returns:
    numpy array of uw_img and w_img.
  """
  uw_img = (
      preview_processing_utils.mirror_preview_image_by_sensor_orientation(
          props['android.sensor.orientation'], uw_img))
  w_img = (
      preview_processing_utils.mirror_preview_image_by_sensor_orientation(
          props['android.sensor.orientation'], w_img))
  uw_img_name = f'{img_name_stem}_uw.png'
  w_img_name = f'{img_name_stem}_w.png'
  image_processing_utils.write_image(uw_img / _CH_FULL_SCALE, uw_img_name)
  image_processing_utils.write_image(w_img / _CH_FULL_SCALE, w_img_name)
  return uw_img, w_img


def _do_ae_check(uw_img, w_img, log_path, suffix):
  """Checks that the luma change is within range.

  Args:
    uw_img: image captured using UW lens.
    w_img: image captured using W lens.
    log_path: path to save the image.
    suffix: str; patch suffix to be used in file name.
  Returns:
    failed_ae_msg: Failed AE check messages if any. None otherwise.
    uw_y_avg: y_avg value for UW lens
    w_y_avg: y_avg value for W lens
  """
  failed_ae_msg = []
  file_stem = f'{os.path.join(log_path, _NAME)}_{suffix}'
  uw_y = _extract_y(
      uw_img, f'{file_stem}_uw_y.png')
  uw_y_avg = np.average(uw_y)
  logging.debug('UW y_avg: %.4f', uw_y_avg)

  w_y = _extract_y(w_img, f'{file_stem}_w_y.png')
  w_y_avg = np.average(w_y)
  logging.debug('W y_avg: %.4f', w_y_avg)

  y_avg_change_percent = (abs(w_y_avg-uw_y_avg)/uw_y_avg)*100
  logging.debug('y_avg_change_percent: %.4f', y_avg_change_percent)

  if not math.isclose(uw_y_avg, w_y_avg, rel_tol=_AE_RTOL, abs_tol=_AE_ATOL):
    failed_ae_msg.append('y_avg change is greater than threshold value for '
                         f'patch: {suffix} '
                         f'diff: {abs(w_y_avg-uw_y_avg):.4f} '
                         f'ATOL: {_AE_ATOL} '
                         f'RTOL: {_AE_RTOL} '
                         f'uw_y_avg: {uw_y_avg:.4f} '
                         f'w_y_avg: {w_y_avg:.4f} ')
  return failed_ae_msg, uw_y_avg, w_y_avg


def _do_af_check(uw_img, w_img):
  """Checks the AF behavior between the uw and w img.

  Args:
    uw_img: image captured using UW lens.
    w_img: image captured using W lens.

  Returns:
    failed_af_msg: Failed AF check messages if any. None otherwise.
    sharpness_uw: sharpness value for UW lens
    sharpness_w: sharpness value for W lens
  """
  failed_af_msg = []
  sharpness_uw = image_processing_utils.compute_image_sharpness(uw_img)
  logging.debug('Sharpness for UW patch: %.2f', sharpness_uw)
  sharpness_w = image_processing_utils.compute_image_sharpness(w_img)
  logging.debug('Sharpness for W patch: %.2f', sharpness_w)

  if not math.isclose(sharpness_w, sharpness_uw, abs_tol=_AF_ATOL):
    failed_af_msg.append('Sharpness difference > threshold value.'
                         f' ATOL: {_AF_ATOL} '
                         f'sharpness_w: {sharpness_w:.4f} '
                         f'sharpness_uw: {sharpness_uw:.4f}')
  return failed_af_msg, sharpness_uw, sharpness_w


def _do_awb_check(uw_img, w_img):
  """Checks the ratio of R/G and B/G for UW and W img.

  Args:
    uw_img: image captured using UW lens.
    w_img: image captured using W lens.
  Returns:
    failed_awb_msg: Failed AWB check messages if any. None otherwise.
  """
  failed_awb_msg = []
  uw_r_g_ratio, uw_b_g_ratio = _get_color_ratios(uw_img, 'UW')
  w_r_g_ratio, w_b_g_ratio = _get_color_ratios(w_img, 'W')

  if not math.isclose(uw_r_g_ratio, w_r_g_ratio,
                      abs_tol=_AWB_ATOL):
    failed_awb_msg.append(f'R/G change is greater than the threshold value: '
                          f'ATOL: {_AWB_ATOL} '
                          f'uw_r_g_ratio: {uw_r_g_ratio:.4f} '
                          f'w_r_g_ratio: {w_r_g_ratio:.4f}')
  if not math.isclose(uw_b_g_ratio, w_b_g_ratio,
                      abs_tol=_AWB_ATOL):
    failed_awb_msg.append(f'B/G change is greater than the threshold value: '
                          f'ATOL: {_AWB_ATOL} '
                          f'uw_b_g_ratio: {uw_b_g_ratio:.4f} '
                          f'w_b_g_ratio: {w_b_g_ratio:.4f}')
  return failed_awb_msg


def _extract_main_patch(corners, ids, img_rgb, img_path, lens_suffix):
  """Extracts the main rectangle patch from the captured frame.

  Find aruco markers in the captured image and detects if the
  expected number of aruco markers have been found or not.
  It then, extracts the main rectangle patch and saves it
  without the aruco markers in it.

  Args:
    corners: list of detected corners.
    ids: list of int ids for each ArUco markers in the input_img.
    img_rgb: An openCV image in RGB order.
    img_path: Path to save the image.
    lens_suffix: str; suffix used to save the image.
  Returns:
    rectangle_patch: numpy float image array of the rectangle patch.
  """
  rectangle_patch = opencv_processing_utils.get_patch_from_aruco_markers(
      img_rgb, corners, ids)
  patch_path = img_path.with_name(
      f'{img_path.stem}_{lens_suffix}_patch{img_path.suffix}')
  image_processing_utils.write_image(rectangle_patch/_CH_FULL_SCALE, patch_path)
  return rectangle_patch


def _extract_y(img_uint8, file_name):
  """Converts an RGB uint8 image to YUV and returns Y.

  The Y img is saved with file_name in the test dir.

  Args:
    img_uint8: An openCV image in RGB order.
    file_name: file name along with the path to save the image.

  Returns:
    An openCV image converted to Y.
  """
  y_uint8 = opencv_processing_utils.convert_to_y(img_uint8, 'RGB')
  y_uint8 = np.expand_dims(y_uint8, axis=2)  # add plane to save image
  image_processing_utils.write_image(y_uint8/_CH_FULL_SCALE, file_name)
  return y_uint8


def _find_aruco_markers(img_bw, img_path, lens_suffix):
  """Detect ArUco markers in the input image.

  Args:
    img_bw: input img in black and white with ArUco markers.
    img_path: path to save the image.
    lens_suffix: suffix used to save the image.
  Returns:
    corners: list of detected corners.
    ids: list of int ids for each ArUco markers in the input_img.
  """
  aruco_path = img_path.with_name(
      f'{img_path.stem}_{lens_suffix}_aruco{img_path.suffix}')
  corners, ids, _ = opencv_processing_utils.find_aruco_markers(
      img_bw, aruco_path)
  if len(ids) != _ARUCO_MARKERS_COUNT:
    raise AssertionError(
        f'{_ARUCO_MARKERS_COUNT} ArUco markers should be detected.')
  return corners, ids


def _get_color_ratios(img, identifier):
  """Computes the ratios of R/G and B/G for img.

  Args:
    img: RGB img in numpy format.
    identifier: str; identifier for logging statement. ie. 'UW' or 'W'

  Returns:
    r_g_ratio: Ratio of R and G channel means.
    b_g_ratio: Ratio of B and G channel means.
  """
  img_means = image_processing_utils.compute_image_means(img)
  r = img_means[0]
  g = img_means[1]
  b = img_means[2]
  logging.debug('%s R mean: %.4f', identifier, r)
  logging.debug('%s G mean: %.4f', identifier, g)
  logging.debug('%s B mean: %.4f', identifier, b)
  r_g_ratio = r/g
  b_g_ratio = b/g
  logging.debug('%s R/G ratio: %.4f', identifier, r_g_ratio)
  logging.debug('%s B/G ratio: %.4f', identifier, b_g_ratio)
  return r_g_ratio, b_g_ratio


def _get_four_quadrant_patches(img, img_path, lens_suffix):
  """Divides the img in 4 equal parts and returns the patches.

  Args:
    img: an openCV image in RGB order.
    img_path: path to save the image.
    lens_suffix: str; suffix used to save the image.
  Returns:
    four_quadrant_patches: list of 4 patches.
  """
  num_rows = 2
  num_columns = 2
  size_x = math.floor(img.shape[1])
  size_y = math.floor(img.shape[0])
  four_quadrant_patches = []
  for i in range(0, num_rows):
    for j in range(0, num_columns):
      x = size_x / num_rows * j
      y = size_y / num_columns * i
      h = size_y / num_columns
      w = size_x / num_rows
      patch = img[int(y):int(y+h), int(x):int(x+w)]
      patch_path = img_path.with_name(
          f'{img_path.stem}_{lens_suffix}_patch_'
          f'{i}_{j}{img_path.suffix}')
      image_processing_utils.write_image(patch/_CH_FULL_SCALE, patch_path)
      cropped_patch = patch[_PATCH_MARGIN:-_PATCH_MARGIN,
                            _PATCH_MARGIN:-_PATCH_MARGIN]
      four_quadrant_patches.append(cropped_patch)
      cropped_patch_path = img_path.with_name(
          f'{img_path.stem}_{lens_suffix}_cropped_patch_'
          f'{i}_{j}{img_path.suffix}')
      image_processing_utils.write_image(
          cropped_patch/_CH_FULL_SCALE, cropped_patch_path)
  return four_quadrant_patches


def _get_slanted_edge_patch(img, img_path, lens_suffix):
  """Crops the central slanted edge part of the img and returns the patch.

  Args:
    img: an openCV image in RGB order.
    img_path: path to save the image.
    lens_suffix: str; suffix used to save the image. ie: 'w' or 'uw'.

  Returns:
    slanted_edge_patch: list of 4 coordinates.
  """
  num_rows = 3
  num_columns = 5
  size_x = math.floor(img.shape[1])
  size_y = math.floor(img.shape[0])
  slanted_edge_patch = []
  x = int(round(size_x / num_columns * (num_columns // 2), 0))
  y = int(round(size_y / num_rows * (num_rows // 2), 0))
  w = int(round(size_x / num_columns, 0))
  h = int(round(size_y / num_rows, 0))
  patch = img[y:y+h, x:x+w]
  slanted_edge_patch = patch[_PATCH_MARGIN:-_PATCH_MARGIN,
                             _PATCH_MARGIN:-_PATCH_MARGIN]
  filename_with_path = img_path.with_name(
      f'{img_path.stem}_{lens_suffix}_slanted_edge{img_path.suffix}'
  )
  image_processing_utils.write_rgb_uint8_image(
      slanted_edge_patch, filename_with_path
  )
  return slanted_edge_patch


class MultiCameraSwitchTest(its_base_test.ItsBaseTest):
  """Test that the switch from UW to W lens has similar RGB values.

  This test uses various zoom ratios within range android.control.zoomRatioRange
  to capture images and find the point when the physical camera changes
  to determine the crossover point of change from UW to W.
  It does preview recording at UW and W crossover point to verify that
  the AE, AWB and AF behavior remains the same.
  """

  def test_multi_camera_switch(self):
    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:
      props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(props)
      chart_distance = self.chart_distance
      failed_awb_msg = []
      failed_ae_msg = []
      failed_af_msg = []

      # check SKIP conditions
      first_api_level = its_session_utils.get_first_api_level(self.dut.serial)
      camera_properties_utils.skip_unless(
          first_api_level >= its_session_utils.ANDROID15_API_LEVEL and
          camera_properties_utils.zoom_ratio_range(props) and
          camera_properties_utils.logical_multi_camera(props))

      # Check the zoom range
      zoom_range = props['android.control.zoomRatioRange']
      logging.debug('zoomRatioRange: %s', zoom_range)
      camera_properties_utils.skip_unless(
          len(zoom_range) > 1 and
          (zoom_range[0] <= _ZOOM_RANGE_UW_W[0] <= zoom_range[1]) and
          (zoom_range[0] <= _ZOOM_RANGE_UW_W[1] <= zoom_range[1]))

      its_session_utils.load_scene(
          cam, props, self.scene, self.tablet, chart_distance)
      # Tap tablet to remove gallery buttons
      if self.tablet:
        self.tablet.adb.shell(
            f'input tap {_TAP_COORDINATES[0]} {_TAP_COORDINATES[1]}')

      preview_test_size = preview_processing_utils.get_max_preview_test_size(
          cam, self.camera_id)
      cam.do_3a()

      # Start dynamic preview recording and collect results
      capture_results, file_list = (
          preview_processing_utils.preview_over_zoom_range(
              self.dut, cam, preview_test_size, _ZOOM_RANGE_UW_W[0],
              _ZOOM_RANGE_UW_W[1], _ZOOM_STEP, self.log_path)
      )

      physical_id_before = None
      counter = 0  # counter for the index of crossover point result
      lens_changed = False

      for capture_result in capture_results:
        counter += 1
        physical_id = capture_result[
            'android.logicalMultiCamera.activePhysicalId']
        if not physical_id_before:
          physical_id_before = physical_id
        zoom_ratio = float(capture_result['android.control.zoomRatio'])
        if physical_id_before == physical_id:
          continue
        else:
          logging.debug('Active physical id changed')
          logging.debug('Crossover zoom ratio point: %f', zoom_ratio)
          physical_id_before = physical_id
          lens_changed = True
          break

      # Raise error is lens did not switch within the range
      # _ZOOM_RANGE_UW_W
      # TODO(ruchamk): Add lens_changed to the CameraITS metrics
      if not lens_changed:
        e_msg = 'Crossover point not found. Try running the test again!'
        raise AssertionError(e_msg)

      img_uw_file = file_list[counter-2]
      capture_result_uw = capture_results[counter-2]
      logging.debug('Capture results uw crossover: %s', capture_result_uw)
      img_w_file = file_list[counter-1]
      capture_result_w = capture_results[counter-1]
      logging.debug('Capture results w crossover: %s', capture_result_w)

      # Remove unwanted frames and only save the UW and
      # W crossover point frames along with mp4 recording
      its_session_utils.remove_frame_files(self.log_path, [
          os.path.join(self.log_path, img_uw_file),
          os.path.join(self.log_path, img_w_file)])

      # Add suffix to the UW and W image files
      uw_path = pathlib.Path(os.path.join(self.log_path, img_uw_file))
      uw_name = uw_path.with_name(f'{uw_path.stem}_uw{uw_path.suffix}')
      os.rename(os.path.join(self.log_path, img_uw_file), uw_name)

      w_path = pathlib.Path(os.path.join(self.log_path, img_w_file))
      w_name = w_path.with_name(f'{w_path.stem}_w{w_path.suffix}')
      os.rename(os.path.join(self.log_path, img_w_file), w_name)

      # Convert UW and W img to numpy array
      uw_img = image_processing_utils.convert_image_to_numpy_array(
          str(uw_name))
      w_img = image_processing_utils.convert_image_to_numpy_array(
          str(w_name))

      # Check the sensor orientation and flip image
      if (props['android.lens.facing'] ==
          camera_properties_utils.LENS_FACING['FRONT']):
        img_name_stem = os.path.join(self.log_path, 'flipped_preview')
        uw_img, w_img = _check_orientation_and_flip(
            props, uw_img, w_img, img_name_stem
        )

      # Convert UW and W img to black and white
      uw_img_bw = (
          opencv_processing_utils.convert_image_to_high_contrast_black_white(
              uw_img))
      w_img_bw = (
          opencv_processing_utils.convert_image_to_high_contrast_black_white(
              w_img))

      # Find ArUco markers in the image with UW lens
      # and extract the outer box patch
      corners, ids = _find_aruco_markers(uw_img_bw, uw_path, 'uw')
      uw_chart_patch = _extract_main_patch(
          corners, ids, uw_img, uw_path, 'uw')
      uw_four_patches = _get_four_quadrant_patches(
          uw_chart_patch, uw_path, 'uw')

      # Find ArUco markers in the image with W lens
      # and extract the outer box patch
      corners, ids = _find_aruco_markers(w_img_bw, w_path, 'w')
      w_chart_patch = _extract_main_patch(
          corners, ids, w_img, w_path, 'w')
      w_four_patches = _get_four_quadrant_patches(
          w_chart_patch, w_path, 'w')

      ae_uw_y_avgs = {}
      ae_w_y_avgs = {}

      for uw_patch, w_patch, color in zip(
          uw_four_patches, w_four_patches, _COLORS):
        logging.debug('Checking for quadrant color: %s', color)

        # AE Check: Extract the Y component from rectangle patch
        failed_ae_msg, uw_y_avg, w_y_avg = _do_ae_check(
            uw_patch, w_patch, self.log_path, color)
        ae_uw_y_avgs.update({color: f'{uw_y_avg:.4f}'})
        ae_w_y_avgs.update({color: f'{w_y_avg:.4f}'})

        # AWB Check : Verify that R/G and B/G ratios are within the limits
        failed_awb_msg = _do_awb_check(uw_patch, w_patch)

      # Below print statements are for logging purpose.
      # Do not replace with logging.
      print(f'{_NAME}_ae_uw_y_avgs: ', ae_uw_y_avgs)
      print(f'{_NAME}_ae_w_y_avgs: ', ae_w_y_avgs)

      # AF check using slanted edge
      uw_slanted_edge_patch = _get_slanted_edge_patch(
          uw_chart_patch, uw_path, 'uw')
      w_slanted_edge_patch = _get_slanted_edge_patch(
          w_chart_patch, w_path, 'w')
      failed_af_msg, sharpness_uw, sharpness_w = _do_af_check(
          uw_slanted_edge_patch, w_slanted_edge_patch)
      print(f'{_NAME}_uw_sharpness: {sharpness_uw:.4f}')
      print(f'{_NAME}_w_sharpness: {sharpness_w:.4f}')

      if failed_awb_msg or failed_ae_msg or failed_af_msg:
        error_msg = _get_error_msg(failed_awb_msg, failed_ae_msg, failed_af_msg)
        raise AssertionError(f'{_NAME} failed with following errors:\n'
                             f'{error_msg}')

if __name__ == '__main__':
  test_runner.main()
