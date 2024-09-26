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
"""Verify that frames from UW and W cameras are not distorted."""

import collections
import logging
import os
import cv2
import math
import numpy as np

from cv2 import aruco
from mobly import test_runner

import its_base_test
import camera_properties_utils
import image_processing_utils
import its_session_utils
import preview_processing_utils

_ACCURACY = 0.001
_ARUCO_COUNT = 8
_ARUCO_DIST_TOL = 0.15
_ARUCO_SIZE = (3, 3)
_ASPECT_RATIO_4_3 = 4/3
_CH_FULL_SCALE = 255
_CHESSBOARD_CORNERS = 24
_CHKR_DIST_TOL = 0.05
_CROSS_SIZE = 6
_CROSS_THICKNESS = 1
_FONT_SCALE = 0.3
_FONT_THICKNESS = 1
_GREEN_LIGHT = (80, 255, 80)
_GREEN_DARK = (0, 190, 0)
_MAX_ITER = 30
_NAME = os.path.splitext(os.path.basename(__file__))[0]
_RED = (255, 0, 0)
_VALID_CONTROLLERS = ('arduino', 'external')
_WIDE_ZOOM = 1
_ZOOM_STEP = 0.5
_ZOOM_STEP_REDUCTION = 0.1
_ZOOM_TOL = 0.1


def get_chart_coverage(image, corners):
  """Calculates the chart coverage in the image.

  Args:
    image: image containing chessboard
    corners: corners of the chart

  Returns:
    chart_coverage: percentage of the image covered by chart corners
    chart_diagonal_pixels: pixel count from the first corner to the last corner
  """
  first_corner = corners[0].tolist()[0]
  logging.debug('first_corner: %s', first_corner)
  last_corner = corners[-1].tolist()[0]
  logging.debug('last_corner: %s', last_corner)
  chart_diagonal_pixels = math.dist(first_corner, last_corner)
  logging.debug('chart_diagonal_pixels: %s', chart_diagonal_pixels)

  # Calculate chart coverage relative to image diagonal
  image_diagonal = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
  logging.debug('image.shape: %s', image.shape)
  logging.debug('Image diagonal (pixels): %s', image_diagonal)
  chart_coverage = chart_diagonal_pixels / image_diagonal * 100
  logging.debug('Chart coverage: %s', chart_coverage)

  return chart_coverage, chart_diagonal_pixels


def plot_corners(image, corners, cross_color=_RED, text_color=_RED):
  """Plot corners to the given image.

  Args:
    image: image
    corners: points in the image
    cross_color: color of cross
    text_color: color of text

  Returns:
    image: image with cross and text for each corner
  """
  for i, corner in enumerate(corners):
    x, y = int(corner.ravel()[0]), int(corner.ravel()[1])

    # Draw corner index
    cv2.putText(image, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                _FONT_SCALE, text_color, _FONT_THICKNESS, cv2.LINE_AA)

  for corner in corners:
    x, y = corner.ravel()

    # Ensure coordinates are integers and within image boundaries
    x = max(0, min(int(x), image.shape[1] - 1))
    y = max(0, min(int(y), image.shape[0] - 1))

    # Draw horizontal line
    cv2.line(image, (x - _CROSS_SIZE, y), (x + _CROSS_SIZE, y), cross_color,
             _CROSS_THICKNESS)
    # Draw vertical line
    cv2.line(image, (x, y - _CROSS_SIZE), (x, y + _CROSS_SIZE), cross_color,
             _CROSS_THICKNESS)

  return image


def get_ideal_points(pattern_size):
  """Calculate the ideal points for pattern.

  These are just corners at unit intervals of the same dimensions
  as pattern_size. Looks like..
   [[ 0.  0.  0.]
    [ 1.  0.  0.]
    [ 2.  0.  0.]
     ...
    [21. 23.  0.]
    [22. 23.  0.]
    [23. 23.  0.]]

  Args:
    pattern_size: pattern size. Example (24, 24)

  Returns:
    ideal_points: corners at unit interval.
  """
  ideal_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
  ideal_points[:,:2] = (
      np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
  )

  return ideal_points


def get_distortion_error(image, corners, ideal_points, rotation_vector,
                         translation_vector, camera_matrix):
  """Get distortion error by comparing corners and ideal points.

  compare corners and ideal points to derive the distortion error

  Args:
    image: image containing chessboard and ArUco
    corners: corners of the chart. Shape = (number of corners, 1, 2)
    ideal_points: corners at unit interval. Shape = (number of corners, 3)
    rotation_vector: rotation vector based on chart's rotation. Shape = (3, 1)
    translation_vector: translation vector based on chart's rotation.
                        Shape = (3, 1)
    camera_matrix: camera intrinsic matrix. Shape = (3, 3)

  Returns:
    normalized_distortion_error_percentage: normalized distortion error
      percentage. None if all corners based on pattern_size not found.
    chart_coverage: percentage of the image covered by corners
  """
  chart_coverage, chart_diagonal_pixels = get_chart_coverage(image, corners)
  logging.debug('Chart coverage: %s', chart_coverage)

  projected_points = cv2.projectPoints(ideal_points, rotation_vector,
                                       translation_vector, camera_matrix, None)
  # Reshape projected points to 2D array
  projected = projected_points[0].reshape(-1, 2)
  corners_reshaped = corners.reshape(-1, 2)
  logging.debug('projected: %s', projected)

  plot_corners(image, projected, _GREEN_LIGHT, _GREEN_DARK)

  # Calculate the distortion error
  distortion_errors = [
      math.dist(projected_point, corner_point)
      for projected_point, corner_point in zip(projected, corners_reshaped)
  ]
  logging.debug('distortion_error: %s', distortion_errors)

  # Get RMS of error
  rms_error = math.sqrt(np.mean(np.square(distortion_errors)))
  logging.debug('RMS distortion error: %s', rms_error)

  # Calculate as a percentage of the chart diagonal
  normalized_distortion_error_percentage = (
      rms_error / chart_diagonal_pixels * 100
  )
  logging.debug('Normalized percent distortion error: %s',
                normalized_distortion_error_percentage)

  return normalized_distortion_error_percentage, chart_coverage


def get_chessboard_corners(pattern_size, image):
  """Find chessboard corners from image.

  Args:
    pattern_size: (int, int) chessboard corners.
    image: image containing chessboard

  Returns:
    corners: corners of the chessboard chart
    ideal_points: ideal pattern of chessboard corners
                  i.e. points at unit intervals
  """
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Find the checkerboard corners
  found_corners, corners_pass1 = cv2.findChessboardCorners(gray_image,
                                                           pattern_size)
  logging.debug('Found corners: %s', found_corners)
  logging.debug('corners_pass1: %s', corners_pass1)

  if not found_corners:
    logging.debug('Chessboard pattern not found.')
    return None, None

  # Refine corners
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _MAX_ITER,
              _ACCURACY)
  corners = cv2.cornerSubPix(gray_image, corners_pass1, (11, 11), (-1, -1),
                             criteria)
  logging.debug('Refined Corners: %s', corners)

  plot_corners(image, corners)

  ideal_points = get_ideal_points(pattern_size)
  logging.debug('ideal_points: %s', ideal_points)

  return corners, ideal_points


def get_aruco_corners(image):
  """Find ArUco corners from image.

  Args:
    image: image containing ArUco markers

  Returns:
    corners: First corner of each ArUco markers in the image.
             None if expected ArUco corners are not found.
    ideal_points: ideal pattern of the ArUco marker corners.
                  None if expected ArUco corners are not found.
  """
  # Detect ArUco markers
  aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
  corners, ids, _ = aruco.detectMarkers(image, aruco_dict)

  logging.debug('corners: %s', corners)
  logging.debug('ids: %s', ids)

  if ids is None:
    logging.debug('ArUco markers are not found')
    return None, None

  aruco.drawDetectedMarkers(image, corners, ids, _RED)

  # Convert to numpy array
  corners = np.concatenate(corners, axis=0).reshape(-1, 4, 2)

  # Extract first corners efficiently
  corners = corners[:, 0, :]
  logging.debug('corners: %s', corners)

  # Create marker_dict using efficient vectorization
  marker_dict = dict(zip(ids.flatten(), corners))

  if len(marker_dict) != _ARUCO_COUNT:
    logging.debug('%s arUCO markers found instead of %s',
                  len(ids), _ARUCO_COUNT)
    return None, None

  # Arrange corners based on ids
  arranged_corners = np.array([marker_dict[i] for i in range(len(corners))])

  # Add a dimension to match format for cv2.calibrateCamera
  corners = np.expand_dims(arranged_corners, axis=1)
  logging.debug('updated corners: %s', corners)

  plot_corners(image, corners)

  ideal_points = get_ideal_points(_ARUCO_SIZE)

  # No ArUco marker in the center, so remove the middle point
  middle_index = (_ARUCO_SIZE[0] // 2) * _ARUCO_SIZE[1] + (_ARUCO_SIZE[1] // 2)
  ideal_points = np.delete(ideal_points, middle_index, axis=0)
  logging.debug('ideal_points: %s', ideal_points)

  return corners, ideal_points


def get_preview_frame(dut, cam, preview_size, zoom, log_path):
  """Captures preview frame at given zoom ratio.

  Args:
    dut: device under test
    cam: camera object
    preview_size: str; preview resolution. ex. '1920x1080'
    zoom: zoom ratio
    log_path: str; path for video file directory

  Returns:
    img_name: the filename of the first captured image
    capture_result: total capture results of the preview frame
  """
  # Define zoom fields such that preview recording is at only one zoom level
  z_min = zoom
  z_max = z_min + _ZOOM_STEP - _ZOOM_STEP_REDUCTION

  # Capture preview images over zoom range
  # TODO: b/343200676 - use do_preview_recording instead of
  #                     preview_over_zoom_range
  capture_results, file_list = preview_processing_utils.preview_over_zoom_range(
      dut, cam, preview_size, z_min, z_max, _ZOOM_STEP, log_path
  )

  # Get first captured image
  img_name = file_list[0]
  capture_result = capture_results[0]

  return img_name, capture_result


def add_update_to_filename(file_name, update_str='_update'):
  """Adds the provided update string to the base name of a file.

  Args:
    file_name (str): The full path to the file to be modified.
    update_str (str, optional): The string to insert before the extension

  Returns:
    file_name: The full path to the new file with the update string added.
  """

  directory, file_with_ext = os.path.split(file_name)
  base_name, ext = os.path.splitext(file_with_ext)

  new_file_name = os.path.join(directory, f'{base_name}_{update_str}{ext}')

  return new_file_name


def get_distortion_errors(props, img_name):
  """Calculates the distortion error using checkerboard and ArUco markers.

  Args:
    props: camera properties object.
    img_name: image name including complete file path

  Returns:
    chkr_chart_coverage: normalized distortion error percentage for chessboard
      corners. None if all corners based on pattern_size not found.
    chkr_chart_coverage: percentage of the image covered by chessboard chart
    arc_distortion_error: normalized distortion error percentage for ArUco
      corners. None if all corners based on pattern_size not found.
    arc_chart_coverage: percentage of the image covered by ArUco corners

  """
  image = cv2.imread(img_name)
  if (props['android.lens.facing'] ==
      camera_properties_utils.LENS_FACING['FRONT']):
    image = preview_processing_utils.mirror_preview_image_by_sensor_orientation(
        props['android.sensor.orientation'], image)

  pattern_size = (_CHESSBOARD_CORNERS, _CHESSBOARD_CORNERS)

  chess_corners, chess_ideal_points = get_chessboard_corners(pattern_size,
                                                             image)
  aruco_corners, aruco_ideal_points = get_aruco_corners(image)

  if chess_corners is None:
    return None, None, None, None

  ideal_points = [chess_ideal_points]
  image_corners = [chess_corners]

  if aruco_corners is not None:
    ideal_points.append(aruco_ideal_points)
    image_corners.append(aruco_corners)

  # Calculate the distortion error
  # Do this by:
  # 1) Calibrate the camera from the detected checkerboard points
  # 2) Project the ideal points, using the camera calibration data.
  # 3) Except, do not use distortion coefficients so we model ideal pinhole
  # 4) Calculate the error of the detected corners relative to the ideal
  # 5) Normalize the average error by the size of the chart
  calib_flags = (
      cv2.CALIB_FIX_K1
      + cv2.CALIB_FIX_K2
      + cv2.CALIB_FIX_K3
      + cv2.CALIB_FIX_K4
      + cv2.CALIB_FIX_K5
      + cv2.CALIB_FIX_K6
      + cv2.CALIB_ZERO_TANGENT_DIST
  )
  ret, camera_matrix, dist_coeffs, rotation_vectors, translation_vectors = (
      cv2.calibrateCamera(ideal_points, image_corners, image.shape[:2],
                          None, None, flags=calib_flags)
  )
  logging.debug('Projection error: %s dist_coeffs: %s', ret, dist_coeffs)
  logging.debug('rotation_vector: %s', rotation_vectors)
  logging.debug('translation_vector: %s', translation_vectors)
  logging.debug('matrix: %s', camera_matrix)

  chkr_distortion_error, chkr_chart_coverage = (
      get_distortion_error(image, chess_corners, chess_ideal_points,
                           rotation_vectors[0], translation_vectors[0],
                           camera_matrix)
  )

  if aruco_corners is not None:
    arc_distortion_error, arc_chart_coverage = get_distortion_error(
        image, aruco_corners, aruco_ideal_points, rotation_vectors[1],
        translation_vectors[1], camera_matrix
    )
  else:
    arc_distortion_error, arc_chart_coverage = None, None

  img_name_update = add_update_to_filename(img_name)
  image_processing_utils.write_image(image / _CH_FULL_SCALE, img_name_update)

  return (chkr_distortion_error, chkr_chart_coverage,
          arc_distortion_error, arc_chart_coverage)


class PreviewDistortionTest(its_base_test.ItsBaseTest):
  """Test that frames from UW and W cameras are not distorted.

  Captures preview frames at different zoom levels. If whole chart is visible
  in the frame, detect the distortion error. Pass the test if distortion error
  is within the pre-determined TOL.
  """

  def test_preview_distortion(self):
    rot_rig = {}
    log_path = self.log_path

    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:

      props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(props)
      camera_properties_utils.skip_unless(
          camera_properties_utils.zoom_ratio_range(props))

      # Raise error if not FRONT or REAR facing camera
      camera_properties_utils.check_front_or_rear_camera(props)

      # Initialize rotation rig
      rot_rig['cntl'] = self.rotator_cntl
      rot_rig['ch'] = self.rotator_ch
      if rot_rig['cntl'].lower() not in _VALID_CONTROLLERS:
        raise AssertionError(
            f'You must use the {_VALID_CONTROLLERS} controller for {_NAME}.')

      # Determine preview size
      preview_size = preview_processing_utils.get_max_preview_test_size(
          cam, self.camera_id, _ASPECT_RATIO_4_3)
      logging.debug('preview_size: %s', preview_size)

      # Determine test zoom range
      z_range = props['android.control.zoomRatioRange']
      logging.debug('z_range: %s', z_range)

      # Collect preview frames and associated capture results
      PreviewFrameData = collections.namedtuple(
          'PreviewFrameData', ['img_name', 'capture_result', 'z_level']
      )
      preview_frames = []
      z_levels = [z_range[0]]  # Min zoom
      if z_range[0] < _WIDE_ZOOM:
        z_levels.append(_WIDE_ZOOM)

      for z in z_levels:
        img_name, capture_result = get_preview_frame(
            self.dut, cam, preview_size, z, log_path
        )
        if img_name:
          frame_data = PreviewFrameData(img_name, capture_result, z)
          preview_frames.append(frame_data)

      failure_msg = []
      # Determine distortion error and chart coverage for each frames
      for frame in preview_frames:
        img_full_name = f'{os.path.join(log_path, frame.img_name)}'
        (chkr_distortion_err, chkr_chart_coverage, arc_distortion_err,
         arc_chart_coverage) = get_distortion_errors(props, img_full_name)

        zoom = float(frame.capture_result['android.control.zoomRatio'])
        if camera_properties_utils.logical_multi_camera(props):
          cam_id = frame.capture_result[
              'android.logicalMultiCamera.activePhysicalId'
          ]
        else:
          cam_id = None
        logging.debug('Zoom: %.2f, cam_id: %s, img_name: %s',
                      zoom, cam_id, img_name)

        if math.isclose(zoom, z_levels[0], rel_tol=_ZOOM_TOL):
          z_str = 'min'
        else:
          z_str = 'max'

        # Don't change print to logging. Used for KPI.
        print(f'{_NAME}_{z_str}_zoom: ', zoom)
        print(f'{_NAME}_{z_str}_physical_id: ', cam_id)
        print(f'{_NAME}_{z_str}_chkr_distortion_error: ', chkr_distortion_err)
        print(f'{_NAME}_{z_str}_chkr_chart_coverage: ', chkr_chart_coverage)
        print(f'{_NAME}_{z_str}_aruco_distortion_error: ', arc_distortion_err)
        print(f'{_NAME}_{z_str}_aruco_chart_coverage: ', arc_chart_coverage)
        logging.debug('%s_%s_zoom: %s', _NAME, z_str, zoom)
        logging.debug('%s_%s_physical_id: %s', _NAME, z_str, cam_id)
        logging.debug('%s_%s_chkr_distortion_error: %s', _NAME, z_str,
                      chkr_distortion_err)
        logging.debug('%s_%s_chkr_chart_coverage: %s', _NAME, z_str,
                      chkr_chart_coverage)
        logging.debug('%s_%s_aruco_distortion_error: %s', _NAME, z_str,
                      arc_distortion_err)
        logging.debug('%s_%s_aruco_chart_coverage: %s', _NAME, z_str,
                      arc_chart_coverage)

        if arc_distortion_err is None:
          if zoom < _WIDE_ZOOM:
            failure_msg.append('Unable to find all ArUco markers in '
                               f'{img_name}')
            logging.debug(failure_msg[-1])
        else:
          if arc_distortion_err > _ARUCO_DIST_TOL:
            failure_msg.append('ArUco Distortion error '
                               f'{arc_distortion_err:.3f} is greater than '
                               f'tolerance {_ARUCO_DIST_TOL}')
            logging.debug(failure_msg[-1])

        if chkr_distortion_err is None:
          # Checkerboard corners shall be detected at minimum zoom level
          failure_msg.append(f'Unable to find full checker board in {img_name}')
          logging.debug(failure_msg[-1])
        else:
          if chkr_distortion_err > _CHKR_DIST_TOL:
            failure_msg.append('Chess Distortion error '
                               f'{chkr_distortion_err:.3f} is greater than '
                               f'tolerance {_CHKR_DIST_TOL}')
            logging.debug(failure_msg[-1])

      if failure_msg:
        raise AssertionError(f'{its_session_utils.NOT_YET_MANDATED_MESSAGE}'
                             f'\n\n{failure_msg}')

if __name__ == '__main__':
  test_runner.main()

