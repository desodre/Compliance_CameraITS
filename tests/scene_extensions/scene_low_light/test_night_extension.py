# Copyright 2023 The Android Open Source Project
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
"""Verify night extension is activated correctly when requested."""


import logging
import os.path

import cv2
from mobly import test_runner

import its_base_test
import camera_properties_utils
import capture_request_utils
import image_processing_utils
import its_session_utils
import lighting_control_utils
import low_light_utils

_NAME = os.path.splitext(os.path.basename(__file__))[0]
_EXTENSION_NIGHT = 4  # CameraExtensionCharacteristics.EXTENSION_NIGHT
_TABLET_BRIGHTNESS = '6'  # Highest minimum brightness on a supported tablet
_TAP_COORDINATES = (500, 500)  # Location to tap tablet screen via adb
_TEST_REQUIRED_MPC = 34

_AVG_DELTA_LUMINANCE_THRESH = 17
_AVG_LUMINANCE_THRESH = 90

_IMAGE_FORMATS_TO_CONSTANTS = (('yuv', 35), ('jpeg', 256))

_X_STRING = 'x'


def _convert_capture(cap, file_stem=None):
  """Obtains y plane and numpy image from a capture.

  Args:
    cap: A capture object as returned by its_session_utils.do_capture.
    file_stem: str; location and name to save files.
  Returns:
    numpy image, with the np.uint8 data type.
  """
  img = image_processing_utils.convert_capture_to_rgb_image(cap)
  if file_stem:
    image_processing_utils.write_image(img, f'{file_stem}.jpg')
  return image_processing_utils.convert_image_to_uint8(img)


class NightExtensionTest(its_base_test.ItsBaseTest):
  """Tests night extension under dark lighting conditions.

  A capture is taken with the night extension ON, after AE converges.
  The capture is analyzed in the same way as test_low_light_boost_extension,
  checking luminance and the average difference in luminance between
  successive boxes.
  """

  def _take_capture(self, cam, req, out_surfaces):
    """Takes capture with Night extension ON.

    Args:
      cam: its_session_utils object.
      req: capture request.
      out_surfaces: dictionary of output surfaces.
    Returns:
      cap: capture object.
    """
    cap = cam.do_capture_with_extensions(req, _EXTENSION_NIGHT, out_surfaces)
    metadata = cap['metadata']
    logging.debug('capture exposure time: %s',
                  metadata['android.sensor.exposureTime'])
    logging.debug('capture sensitivity: %s',
                  metadata['android.sensor.sensitivity'])
    return cap

  def test_night_extension(self):
    # Handle subdirectory
    self.scene = 'scene_low_light'
    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:
      props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(props)
      test_name = os.path.join(self.log_path, _NAME)

      # Determine camera supported extensions
      supported_extensions = cam.get_supported_extensions(self.camera_id)
      logging.debug('Supported extensions: %s', supported_extensions)

      # Check media performance class
      should_run = _EXTENSION_NIGHT in supported_extensions
      media_performance_class = its_session_utils.get_media_performance_class(
          self.dut.serial)
      if (media_performance_class >= _TEST_REQUIRED_MPC and
          cam.is_primary_camera() and
          not should_run):
        its_session_utils.raise_mpc_assertion_error(
            _TEST_REQUIRED_MPC, _NAME, media_performance_class)

      # Check SKIP conditions
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

      # Determine capture width, height, and format
      for format_name, format_constant in _IMAGE_FORMATS_TO_CONSTANTS:
        capture_sizes = capture_request_utils.get_available_output_sizes(
            format_name, props)
        extension_capture_sizes_str = cam.get_supported_extension_sizes(
            self.camera_id, _EXTENSION_NIGHT, format_constant
        )
        if not extension_capture_sizes_str:
          continue
        extension_capture_sizes = [
            tuple(int(size_part) for size_part in s.split(_X_STRING))
            for s in extension_capture_sizes_str
        ]
        # Extension capture sizes ordered in ascending area order by default
        extension_capture_sizes.reverse()
        logging.debug('Capture sizes: %s', capture_sizes)
        logging.debug('Extension capture sizes: %s', extension_capture_sizes)
        logging.debug('Accepted capture format: %s', format_name)
        width, height = extension_capture_sizes[0]
        accepted_format = format_name
        break
      else:
        raise AssertionError('No supported sizes/formats found!')

      # Set tablet brightness to darken scene
      self.set_screen_brightness(_TABLET_BRIGHTNESS)

      file_stem = f'{test_name}_{self.camera_id}_{accepted_format}_{width}x{height}'
      out_surfaces = {
          'format': accepted_format, 'width': width, 'height': height}
      req = capture_request_utils.auto_capture_request()

      logging.debug('Taking auto capture with night mode ON')
      night_cap = self._take_capture(
          cam, req, out_surfaces)
      rgb_night_img = _convert_capture(night_cap, f'{file_stem}_night')

      # Assert correct behavior and create luminosity plots
      low_light_utils.analyze_low_light_scene_capture(
          f'{file_stem}_night',
          cv2.cvtColor(rgb_night_img, cv2.COLOR_RGB2BGR),
          _AVG_LUMINANCE_THRESH,
          _AVG_DELTA_LUMINANCE_THRESH
      )

if __name__ == '__main__':
  test_runner.main()
