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
"""Verifies changing AE/AWB regions changes images AE/AWB results."""


import logging
import os.path

from mobly import test_runner
import numpy

import camera_properties_utils
import capture_request_utils
import image_processing_utils
import its_base_test
import its_session_utils
import opencv_processing_utils
import video_processing_utils

_AE_CHANGE_THRESH = 1  # Incorrect behavior is empirically < 0.5 percent
_AWB_CHANGE_THRESH = 2  # Incorrect behavior is empirically < 1.5 percent
_AE_AWB_METER_WEIGHT = 1000  # 1 - 1000 with 1000 as the highest
_ARUCO_MARKERS_COUNT = 4
_AE_AWB_REGIONS_AVAILABLE = 1  # Valid range is >= 0, and unavailable if 0
_MIRRORED_PREVIEW_SENSOR_ORIENTATIONS = (0, 180)
_NAME = os.path.splitext(os.path.basename(__file__))[0]
_NUM_AE_AWB_REGIONS = 4
_PERCENTAGE = 100
_REGION_DURATION_MS = 1800  # 1.8 seconds
_TAP_COORDINATES = (500, 500)  # Location to tap tablet screen via adb


def _convert_image_coords_to_sensor_coords(
    aa_width, aa_height, coords, img_width, img_height):
  """Transform image coordinates to sensor coordinate system.

  Calculate the difference between sensor active array and image aspect ratio.
  Taking the difference into account, figure out if the width or height has been
  cropped. Using this information, transform the image coordinates to sensor
  coordinates.

  Args:
    aa_width: int; active array width.
    aa_height: int; active array height.
    coords: coordinates; defined by aruco markers from camera capture.
    img_width: int; width of image.
    img_height: int; height of image.
  Returns:
    sensor_coords: coordinates; corresponding coorediates on
      sensor coordinate system.
  """
  # TODO: b/330382627 - find out if distortion correction is ON/OFF
  aa_aspect_ratio = aa_width / aa_height
  image_aspect_ratio = img_width / img_height
  if aa_aspect_ratio >= image_aspect_ratio:
    # If aa aspect ratio is greater than image aspect ratio, then
    # sensor width is being cropped
    aspect_ratio_multiplication_factor = aa_height / img_height
    crop_width = img_width * aspect_ratio_multiplication_factor
    buffer = (aa_width - crop_width) / 2
    sensor_coords = (coords[0] * aspect_ratio_multiplication_factor + buffer,
                     coords[1] * aspect_ratio_multiplication_factor)
  else:
    # If aa aspect ratio is less than image aspect ratio, then
    # sensor height is being cropped
    aspect_ratio_multiplication_factor = aa_width / img_width
    crop_height = img_height * aspect_ratio_multiplication_factor
    buffer = (aa_height - crop_height) / 2
    sensor_coords = (coords[0] * aspect_ratio_multiplication_factor,
                     coords[1] * aspect_ratio_multiplication_factor + buffer)
  logging.debug('Sensor coordinates: %s', sensor_coords)
  return sensor_coords


def _define_metering_regions(img, img_path, chart_path, props, width, height):
  """Define 4 metering rectangles for AE/AWB regions based on ArUco markers.

  Args:
    img: numpy array; RGB image.
    img_path: str; image file location.
    chart_path: str; chart file location.
    props: dict; camera properties object.
    width: int; preview's width in pixels.
    height: int; preview's height in pixels.
  Returns:
    ae_awb_regions: metering rectangles; AE/AWB metering regions.
  """
  # Extract chart coordinates from aruco markers
  # TODO: b/330382627 - get chart boundary from 4 aruco markers instead of 2
  aruco_corners, aruco_ids, _ = opencv_processing_utils.find_aruco_markers(
      img, img_path)
  tl, tr, br, bl = (
      opencv_processing_utils.get_chart_boundary_from_aruco_markers(
          aruco_corners, aruco_ids, img, chart_path))

  # Convert image coordinates to sensor coordinates for metering rectangles
  aa = props['android.sensor.info.activeArraySize']
  aa_width, aa_height = aa['right'] - aa['left'], aa['bottom'] - aa['top']
  logging.debug('Active array size: %s', aa)
  sc_tl = _convert_image_coords_to_sensor_coords(
      aa_width, aa_height, tl, width, height)
  sc_tr = _convert_image_coords_to_sensor_coords(
      aa_width, aa_height, tr, width, height)
  sc_br = _convert_image_coords_to_sensor_coords(
      aa_width, aa_height, br, width, height)
  sc_bl = _convert_image_coords_to_sensor_coords(
      aa_width, aa_height, bl, width, height)

  # Define AE/AWB regions through ArUco markers' positions
  region_blue, region_light, region_dark, region_yellow = (
      opencv_processing_utils.define_metering_rectangle_values(
          props, sc_tl, sc_tr, sc_br, sc_bl, aa_width, aa_height))

  # Create a dictionary of AE/AWB regions for testing
  ae_awb_regions = {
      'aeAwbRegionOne': region_blue,
      'aeAwbRegionTwo': region_light,
      'aeAwbRegionThree': region_dark,
      'aeAwbRegionFour': region_yellow,
  }
  return ae_awb_regions


def _do_ae_check(light, dark, file_name_with_path):
  """Checks luma change between two images is above threshold.

  Checks that the Y-average of image with darker metering region
  is higher than the Y-average of image with lighter metering
  region. Y stands for brightness, or "luma".

  Args:
    light: RGB image; metering light region.
    dark: RGB image; metering dark region.
    file_name_with_path: str; path to preview recording.
  """
  # Converts img to YUV and returns Y-average
  light_y = opencv_processing_utils.convert_to_y(light, 'RGB')
  light_y_avg = numpy.average(light_y)
  dark_y = opencv_processing_utils.convert_to_y(dark, 'RGB')
  dark_y_avg = numpy.average(dark_y)
  logging.debug('Light image Y-average: %.4f', light_y_avg)
  logging.debug('Dark image Y-average: %.4f', dark_y_avg)
  # Checks average change in Y-average between two images
  y_avg_change = (
      (dark_y_avg-light_y_avg)/light_y_avg)*_PERCENTAGE
  logging.debug('Y-average percentage change: %.4f', y_avg_change)

  # Don't change print to logging. Used for KPI.
  print(f'{_NAME}_ae_y_change: ', y_avg_change)

  if y_avg_change < _AE_CHANGE_THRESH:
    raise AssertionError(
        f'Luma change {y_avg_change} is less than the threshold: '
        f'{_AE_CHANGE_THRESH}')
  else:
    its_session_utils.remove_mp4_file(file_name_with_path)


def _do_awb_check(blue, yellow):
  """Checks the ratio of red over blue between two RGB images.

  Checks that the R/B of image with blue metering region
  is higher than the R/B of image with yellow metering
  region.

  Args:
    blue: RGB image; metering blue region.
    yellow: RGB image; metering yellow region.
  Returns:
    failure_messages: (list of strings) of error messages.
  """
  # Calculates average red value over average blue value in images
  blue_r_b_ratio = _get_red_blue_ratio(blue)
  yellow_r_b_ratio = _get_red_blue_ratio(yellow)
  logging.debug('Blue image R/B ratio: %s', blue_r_b_ratio)
  logging.debug('Yellow image R/B ratio: %s', yellow_r_b_ratio)
  # Calculates change in red over blue values between two images
  r_b_ratio_change = (
      (blue_r_b_ratio-yellow_r_b_ratio)/yellow_r_b_ratio)*_PERCENTAGE
  logging.debug('R/B ratio change in percentage: %.4f', r_b_ratio_change)

  # Don't change print to logging. Used for KPI.
  print(f'{_NAME}_awb_rb_change: ', r_b_ratio_change)

  if r_b_ratio_change < _AWB_CHANGE_THRESH:
    raise AssertionError(
        f'R/B ratio change {r_b_ratio_change} is less than the'
        f' threshold: {_AWB_CHANGE_THRESH}')


def _extract_and_process_key_frames_from_recording(log_path, file_name):
  """Extract key frames (1 frame/second) from recordings.

  Args:
    log_path: str; file location.
    file_name: str; file name for saved video.
  Returns:
    dictionary of images.
  """
  # TODO: b/330382627 - Add function to preview_processing_utils
  # Extract key frames from video
  key_frame_files = video_processing_utils.extract_key_frames_from_video(
      log_path, file_name)

  # Process key frame files
  key_frames = []
  for file in key_frame_files:
    img = image_processing_utils.convert_image_to_numpy_array(
        os.path.join(log_path, file))
    key_frames.append(img)
  logging.debug('Frame size %d x %d', key_frames[0].shape[1],
                key_frames[0].shape[0])
  return key_frames


def _get_largest_common_aspect_ratio_preview_size(cam, camera_id):
  """Get largest, supported preview size that matches sensor's aspect ratio.

  Args:
    cam: obj; camera object.
    camera_id: int; device id.
  Returns:
    preview_size: str; largest, supported preview size w/ 4:3, or 16:9
        aspect ratio.
  """
  preview_sizes = cam.get_supported_preview_sizes(camera_id)
  dimensions = lambda s: (int(s.split('x')[0]), int(s.split('x')[1]))
  for size in reversed(preview_sizes):
    if capture_request_utils.is_common_aspect_ratio(dimensions(size)):
      preview_size = size
      logging.debug('Largest common aspect ratio preview size: %s',
                    preview_size)
      break
  return preview_size


def _get_red_blue_ratio(img):
  """Computes the ratios of average red over blue in img.

  Args:
    img: numpy array; RGB image.
  Returns:
    r_b_ratio: float; ratio of R and B channel means.
  """
  img_means = image_processing_utils.compute_image_means(img)
  r_b_ratio = img_means[0]/img_means[2]
  return r_b_ratio


class AeAwbRegions(its_base_test.ItsBaseTest):
  """Tests that changing AE and AWB regions changes image's RGB values.

  Test records an 8 seconds preview recording, and meters a different
  AE/AWB region (blue, light, dark, yellow) for every 2 seconds.
  Extracts a frame from each second of recording with a total of 8 frames
  (2 from each region). For AE check, a frame from light is compared to the
  dark region. For AWB check, a frame from blue is compared to the yellow
  region.

  """

  def test_ae_awb_regions(self):
    """Test AE and AWB regions."""

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
          cam, props, self.scene, self.tablet, self.chart_distance,
          log_path)

      # Tap tablet to remove gallery buttons
      if self.tablet:
        self.tablet.adb.shell(
            f'input tap {_TAP_COORDINATES[0]} {_TAP_COORDINATES[1]}')

      # Check skip conditions
      max_ae_regions = props['android.control.maxRegionsAe']
      max_awb_regions = props['android.control.maxRegionsAwb']
      first_api_level = its_session_utils.get_first_api_level(self.dut.serial)
      camera_properties_utils.skip_unless(
          first_api_level >= its_session_utils.ANDROID15_API_LEVEL and
          camera_properties_utils.ae_regions(props) and
          (max_awb_regions >= _AE_AWB_REGIONS_AVAILABLE or
           max_ae_regions >= _AE_AWB_REGIONS_AVAILABLE))
      logging.debug('maximum AE regions: %d', max_ae_regions)
      logging.debug('maximum AWB regions: %d', max_awb_regions)

      # Find largest preview size to define capture size to find aruco markers
      preview_size = _get_largest_common_aspect_ratio_preview_size(
          cam, self.camera_id)
      width = int(preview_size.split('x')[0])
      height = int(preview_size.split('x')[1])
      req = capture_request_utils.auto_capture_request()
      fmt = {'format': 'yuv', 'width': width, 'height': height}
      cam.do_3a()
      cap = cam.do_capture(req, fmt)

      # Save image and convert to numpy array
      img = image_processing_utils.convert_capture_to_rgb_image(
          cap, props=props)
      img_path = f'{test_name_with_log_path}_aruco_markers.jpg'
      image_processing_utils.write_image(img, img_path)
      img = image_processing_utils.convert_image_to_uint8(img)

      # Define AE/AWB metering regions
      chart_path = f'{test_name_with_log_path}_chart_boundary.jpg'
      ae_awb_regions = _define_metering_regions(
          img, img_path, chart_path, props, width, height)

      # Do preview recording with pre-defined AE/AWB regions
      recording_obj = cam.do_preview_recording_with_dynamic_ae_awb_region(
          preview_size, ae_awb_regions, _REGION_DURATION_MS)
      logging.debug('Tested quality: %s', recording_obj['quality'])

      # Grab the video from the save location on DUT
      self.dut.adb.pull([recording_obj['recordedOutputPath'], log_path])
      file_name = recording_obj['recordedOutputPath'].split('/')[-1]
      file_name_with_path = os.path.join(log_path, file_name)
      logging.debug('file_name: %s', file_name)

      # Extract 8 key frames per 8 seconds of preview recording
      # Meters each region of 4 (blue, light, dark, yellow) for 2 seconds
      # Unpack frames based on metering region's color
      # If testing front camera with preview mirrored, reverse order.
      # pylint: disable=unbalanced-tuple-unpacking
      if ((props['android.lens.facing'] ==
           camera_properties_utils.LENS_FACING['FRONT']) and
          props['android.sensor.orientation'] in
          _MIRRORED_PREVIEW_SENSOR_ORIENTATIONS):
        _, yellow, _, dark, _, light, _, blue = (
            _extract_and_process_key_frames_from_recording(
                log_path, file_name))
      else:
        _, blue, _, light, _, dark, _, yellow = (
            _extract_and_process_key_frames_from_recording(
                log_path, file_name))

      # AWB Check : Verify R/B ratio change is greater than threshold
      if max_awb_regions >= _AE_AWB_REGIONS_AVAILABLE:
        _do_awb_check(blue, yellow)

      # AE Check: Extract the Y component from rectangle patch
      if max_ae_regions >= _AE_AWB_REGIONS_AVAILABLE:
        _do_ae_check(light, dark, file_name_with_path)

if __name__ == '__main__':
  test_runner.main()
