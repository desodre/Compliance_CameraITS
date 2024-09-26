# Copyright 2022 The Android Open Source Project
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
"""Verify preview is stable during phone movement."""

import logging
import os

from mobly import test_runner

import its_base_test
import camera_properties_utils
import its_session_utils
import preview_processing_utils
import video_processing_utils

_NAME = os.path.splitext(os.path.basename(__file__))[0]
# 1080P with 16:9 aspect ratio, 720P and VGA resolutions
_TARGET_PREVIEW_SIZES = ('1920x1080', '1280x720', '640x480')
_TEST_REQUIRED_MPC_FRONT = 34
_TEST_REQUIRED_MPC_REAR = 33
_ZOOM_RATIO_UW = 0.9
_ZOOM_RATIO_W = 1.0


def _get_preview_sizes(cam, camera_id):
  """Determine preview sizes to test based on DUT's supported sizes.

  Targeting 1080P (16:9 ratio), 720P and VGA.

  Args:
    cam: ItsSession camera object.
    camera_id: str; unique identifier assigned to each camera.
  Returns:
    preview sizes to test.
  """
  preview_sizes_to_test = cam.get_supported_preview_sizes(camera_id)
  preview_sizes_to_test = [size for size in preview_sizes_to_test
                           if size in _TARGET_PREVIEW_SIZES]
  logging.debug('Preview sizes to test: %s', preview_sizes_to_test)
  return preview_sizes_to_test


class PreviewStabilizationTest(its_base_test.ItsBaseTest):
  """Tests if preview is stabilized.

  Camera is moved in sensor fusion rig on an arc of 15 degrees.
  Speed is set to mimic hand movement (and not be too fast).
  Preview is captured after rotation rig starts moving, and the
  gyroscope data is dumped.

  The recorded preview is processed to dump all of the frames to
  PNG files. Camera movement is extracted from frames by determining
  max angle of deflection in video movement vs max angle of deflection
  in gyroscope movement. Test is a PASS if rotation is reduced in video.
  """

  def test_preview_stabilization(self):
    rot_rig = {}
    log_path = self.log_path

    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:

      props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(props)
      first_api_level = its_session_utils.get_first_api_level(self.dut.serial)
      camera_properties_utils.skip_unless(
          first_api_level >= its_session_utils.ANDROID13_API_LEVEL,
          'First API level should be {} or higher. Found {}.'.format(
              its_session_utils.ANDROID13_API_LEVEL, first_api_level))

      supported_stabilization_modes = props[
          'android.control.availableVideoStabilizationModes'
      ]

      # Check media performance class
      should_run = (supported_stabilization_modes is not None and
                    camera_properties_utils.STABILIZATION_MODE_PREVIEW in
                    supported_stabilization_modes)
      media_performance_class = its_session_utils.get_media_performance_class(
          self.dut.serial)
      if (props['android.lens.facing'] ==
          camera_properties_utils.LENS_FACING['FRONT']):
        if (media_performance_class >= _TEST_REQUIRED_MPC_FRONT
            and not should_run):
          its_session_utils.raise_mpc_assertion_error(
              _TEST_REQUIRED_MPC_FRONT, _NAME, media_performance_class)
      else:
        if (media_performance_class >= _TEST_REQUIRED_MPC_REAR
            and not should_run):
          its_session_utils.raise_mpc_assertion_error(
              _TEST_REQUIRED_MPC_REAR, _NAME, media_performance_class)

      camera_properties_utils.skip_unless(should_run)

      # Log ffmpeg version being used
      video_processing_utils.log_ffmpeg_version()

      # Raise error if not FRONT or REAR facing camera
      facing = props['android.lens.facing']
      camera_properties_utils.check_front_or_rear_camera(props)

      # Check zoom range
      zoom_range = props['android.control.zoomRatioRange']
      logging.debug('zoomRatioRange: %s', str(zoom_range))

      # If device doesn't support UW, only test W
      # If device's UW's zoom ratio is bigger than 0.9x, use that value
      test_zoom_ratios = [_ZOOM_RATIO_W]
      if (zoom_range[0] < _ZOOM_RATIO_W and
          first_api_level >= its_session_utils.ANDROID15_API_LEVEL):
        test_zoom_ratios.append(max(_ZOOM_RATIO_UW, zoom_range[0]))

      # Initialize rotation rig
      rot_rig['cntl'] = self.rotator_cntl
      rot_rig['ch'] = self.rotator_ch
      if rot_rig['cntl'].lower() != 'arduino':
        raise AssertionError(
            f'You must use the arduino controller for {_NAME}.')

      # Determine preview sizes to test
      preview_sizes_to_test = _get_preview_sizes(cam, self.camera_id)

      # Preview recording with camera movement
      stabilization_result = {}
      for preview_size in preview_sizes_to_test:
        for zoom_ratio in test_zoom_ratios:
          recording_obj = preview_processing_utils.collect_data(
              cam, self.tablet_device, preview_size,
              stabilize=True, rot_rig=rot_rig, zoom_ratio=zoom_ratio)

          # Get gyro events
          logging.debug('Reading out inertial sensor events')
          gyro_events = cam.get_sensor_events()['gyro']
          logging.debug('Number of gyro samples %d', len(gyro_events))

          # Grab the video from the save location on DUT
          self.dut.adb.pull([recording_obj['recordedOutputPath'], log_path])

          # Verify stabilization was applied to preview stream
          stabilization_result[preview_size] = (
              preview_processing_utils.verify_preview_stabilization(
                  recording_obj, gyro_events, _NAME, log_path, facing,
                  zoom_ratio)
          )

      # Assert PASS/FAIL criteria
      test_failures = []
      for _, result_per_size in stabilization_result.items():
        if result_per_size['failure'] is not None:
          test_failures.append(result_per_size['failure'])

      if test_failures:
        raise AssertionError(test_failures)


if __name__ == '__main__':
  test_runner.main()
