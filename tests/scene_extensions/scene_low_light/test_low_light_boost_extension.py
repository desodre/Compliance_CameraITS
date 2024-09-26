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
"""Verify low light boost api is activated correctly when requested."""


import cv2
import logging
import os.path

from mobly import test_runner
import numpy as np

import its_base_test
import camera_properties_utils
import capture_request_utils
import image_processing_utils
import its_session_utils
import lighting_control_utils
import low_light_utils
import preview_processing_utils

_AE_LOW_LIGHT_BOOST_MODE = 6

_CONTROL_AF_MODE_AUTO = 1
_CONTROL_AWB_MODE_AUTO = 1
_CONTROL_MODE_AUTO = 1
_CONTROL_VIDEO_STABILIZATION_MODE_OFF = 0
_LENS_OPTICAL_STABILIZATION_MODE_OFF = 0

_EXTENSION_NIGHT = 4  # CameraExtensionCharacteristics#EXTENSION_NIGHT
_EXTENSION_NONE = -1  # Use Camera2 instead of a Camera Extension
_NAME = os.path.splitext(os.path.basename(__file__))[0]
_NUM_FRAMES_TO_WAIT = 40  # The preview frame number to capture
_TABLET_BRIGHTNESS_REAR_CAMERA = '6'  # Target brightness on a supported tablet
_TABLET_BRIGHTNESS_FRONT_CAMERA = '12'  # Target brightness on a supported
                                        # tablet
_TAP_COORDINATES = (500, 500)  # Location to tap tablet screen via adb

_AVG_DELTA_LUMINANCE_THRESH = 18
_AVG_LUMINANCE_THRESH = 70

_CAPTURE_REQUEST = {
    'android.control.mode': _CONTROL_MODE_AUTO,
    'android.control.aeMode': _AE_LOW_LIGHT_BOOST_MODE,
    'android.control.awbMode': _CONTROL_AWB_MODE_AUTO,
    'android.control.afMode': _CONTROL_AF_MODE_AUTO,
    'android.lens.opticalStabilizationMode':
        _LENS_OPTICAL_STABILIZATION_MODE_OFF,
    'android.control.videoStabilizationMode':
        _CONTROL_VIDEO_STABILIZATION_MODE_OFF,
}


def _capture_and_analyze(cam, file_stem, camera_id, preview_size, extension,
                         mirror_output):
  """Capture a preview frame and then analyze it.

  Args:
    cam: ItsSession object to send commands.
    file_stem: File prefix for captured images.
    camera_id: Camera ID under test.
    preview_size: Target size of preview.
    extension: Extension mode or -1 to use Camera2.
    mirror_output: If the output should be mirrored across the vertical axis.
  """
  frame_bytes = cam.do_capture_preview_frame(camera_id,
                                             preview_size,
                                             _NUM_FRAMES_TO_WAIT,
                                             extension,
                                             _CAPTURE_REQUEST)
  np_array = np.frombuffer(frame_bytes, dtype=np.uint8)
  img_rgb = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
  if mirror_output:
    img_rgb = cv2.flip(img_rgb, 1)
  low_light_utils.analyze_low_light_scene_capture(
      file_stem,
      img_rgb,
      _AVG_LUMINANCE_THRESH,
      _AVG_DELTA_LUMINANCE_THRESH)


class LowLightBoostTest(its_base_test.ItsBaseTest):
  """Tests low light boost mode under dark lighting conditions.

  The test checks if low light boost AE mode is available. The test is skipped
  if it is not available for Camera2 and Camera Extensions Night Mode.

  Low light boost is enabled and a frame from the preview stream is captured
  for analysis. The analysis applies the following operations:
    1. Crops the region defined by a red square outline
    2. Detects the presence of 20 boxes
    3. Computes the luminance bounded by each box
    4. Determines the average luminance of the 6 darkest boxes according to the
      Hilbert curve arrangement of the grid.
    5. Determines the average difference in luminance of the 6 successive
      darkest boxes.
    6. Checks for passing criteria: the avg luminance must be at least 90 or
      greater, the avg difference in luminance between successive boxes must be
      at least 18 or greater.
  """

  def test_low_light_boost(self):
    self.scene = 'scene_low_light'
    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:
      props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(props)
      test_name = os.path.join(self.log_path, _NAME)

      # Check SKIP conditions
      # Determine if DUT is at least Android 15
      first_api_level = its_session_utils.get_first_api_level(self.dut.serial)
      camera_properties_utils.skip_unless(
          first_api_level >= its_session_utils.ANDROID15_API_LEVEL)

      # Determine if low light boost is available
      is_low_light_boost_supported = (
          cam.is_low_light_boost_available(self.camera_id, _EXTENSION_NONE))
      is_low_light_boost_supported_night = (
          cam.is_low_light_boost_available(self.camera_id, _EXTENSION_NIGHT))
      should_run = (is_low_light_boost_supported or
                    is_low_light_boost_supported_night)
      camera_properties_utils.skip_unless(should_run)

      tablet_name_unencoded = self.tablet.adb.shell(
          ['getprop', 'ro.product.device']
      )
      tablet_name = str(tablet_name_unencoded.decode('utf-8')).strip()
      logging.debug('Tablet name: %s', tablet_name)

      if (tablet_name.lower() not in
          low_light_utils.TABLET_LOW_LIGHT_SCENES_ALLOWLIST):
        raise AssertionError('Tablet not supported for low light scenes.')

      if tablet_name == its_session_utils.TABLET_LEGACY_NAME:
        raise AssertionError(f'Incompatible tablet! Please use a tablet with '
                             'display brightness of at least '
                             f'{its_session_utils.TABLET_DEFAULT_BRIGHTNESS} '
                             'according to '
                             f'{its_session_utils.TABLET_REQUIREMENTS_URL}.')

      # Establish connection with lighting controller
      arduino_serial_port = lighting_control_utils.lighting_control(
          self.lighting_cntl, self.lighting_ch)

      # Turn OFF lights to darken scene
      lighting_control_utils.set_lighting_state(
          arduino_serial_port, self.lighting_ch, 'OFF')

      # Check that tablet is connected and turn it off to validate lighting
      self.turn_off_tablet()

      # Turn off DUT to reduce reflections
      lighting_control_utils.turn_off_device_screen(self.dut)

      # Validate lighting, then setup tablet
      cam.do_3a(do_af=False)
      cap = cam.do_capture(
          capture_request_utils.auto_capture_request(), cam.CAP_YUV)
      y_plane, _, _ = image_processing_utils.convert_capture_to_planes(cap)
      its_session_utils.validate_lighting(
          y_plane, self.scene, state='OFF', log_path=self.log_path,
          tablet_state='OFF')
      self.setup_tablet()

      its_session_utils.load_scene(
          cam, props, self.scene, self.tablet, self.chart_distance,
          lighting_check=False, log_path=self.log_path)

      # Tap tablet to remove gallery buttons
      if self.tablet:
        self.tablet.adb.shell(
            f'input tap {_TAP_COORDINATES[0]} {_TAP_COORDINATES[1]}')

      # Set tablet brightness to darken scene
      props = cam.get_camera_properties()
      if (props['android.lens.facing'] ==
          camera_properties_utils.LENS_FACING['BACK']):
        self.set_screen_brightness(_TABLET_BRIGHTNESS_REAR_CAMERA)
      elif (props['android.lens.facing'] ==
            camera_properties_utils.LENS_FACING['FRONT']):
        self.set_screen_brightness(_TABLET_BRIGHTNESS_FRONT_CAMERA)
      else:
        logging.debug('Only front and rear camera supported. '
                      'Skipping for camera ID %s',
                      self.camera_id)
        camera_properties_utils.skip_unless(False)

      cam.do_3a()

      # Mirror the capture across the vertical axis if captured by front facing
      # camera
      should_mirror = (props['android.lens.facing'] ==
                       camera_properties_utils.LENS_FACING['FRONT'])

      # Since low light boost can be supported by Camera2 and Night Mode
      # Extensions, run the test for both (if supported)

      if is_low_light_boost_supported:
        # Determine preview width and height to test
        target_preview_size = (
            preview_processing_utils.get_max_preview_test_size(
                cam, self.camera_id))
        logging.debug('target_preview_size: %s', target_preview_size)

        logging.debug('capture frame using camera2')
        file_stem = f'{test_name}_{self.camera_id}_camera2'
        _capture_and_analyze(cam, file_stem, self.camera_id,
                             target_preview_size, _EXTENSION_NONE,
                             should_mirror)

      if is_low_light_boost_supported_night:
        # Determine preview width and height to test
        target_preview_size = (
            preview_processing_utils.get_max_extension_preview_test_size(
                cam, self.camera_id, _EXTENSION_NIGHT))
        logging.debug('target_preview_size: %s', target_preview_size)

        logging.debug('capture frame using night mode extension')
        file_stem = f'{test_name}_{self.camera_id}_camera_extension'
        _capture_and_analyze(cam, file_stem, self.camera_id,
                             target_preview_size, _EXTENSION_NIGHT,
                             should_mirror)


if __name__ == '__main__':
  test_runner.main()
