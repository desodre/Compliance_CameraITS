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
"""Test the camera in-sensor zoom behavior."""

import logging
import os.path

import camera_properties_utils
import capture_request_utils
import cv2
import image_processing_utils
import its_base_test
import its_session_utils
import zoom_capture_utils

from mobly import test_runner
import numpy as np

_NAME = os.path.splitext(os.path.basename(__file__))[0]
_NUM_STEPS = 10
_THRESHOLD_MAX_RMS_DIFF_CROPPED_RAW_USE_CASE = 0.06


class InSensorZoomTest(its_base_test.ItsBaseTest):

  """Use case CROPPED_RAW: verify that CaptureResult.RAW_CROP_REGION matches cropped RAW image."""

  def test_in_sensor_zoom(self):
    with its_session_utils.ItsSession(
        device_id=self.dut.serial,
        camera_id=self.camera_id,
        hidden_physical_id=self.hidden_physical_id) as cam:
      logical_props = cam.get_camera_properties()
      props = cam.override_with_hidden_physical_camera_props(logical_props)
      name_with_log_path = os.path.join(self.log_path, _NAME)
      debug = self.debug_mode
      # Skip the test if CROPPED_RAW is not present in stream use cases
      camera_properties_utils.skip_unless(
          camera_properties_utils.cropped_raw_stream_use_case(props))

      # Load chart for scene
      its_session_utils.load_scene(
          cam, props, self.scene, self.tablet, self.chart_distance)

      z_range = props['android.control.zoomRatioRange']
      logging.debug('In sensor zoom: testing zoomRatioRange: %s', str(z_range))

      z_min, z_max = float(z_range[0]), float(z_range[1])
      camera_properties_utils.skip_unless(
          z_max >= z_min * zoom_capture_utils.ZOOM_MIN_THRESH)
      z_list = np.arange(z_min, z_max, float(z_max - z_min) / (_NUM_STEPS - 1))
      z_list = np.append(z_list, z_max)

      a = props['android.sensor.info.activeArraySize']
      aw, ah = a['right'] - a['left'], a['bottom'] - a['top']

      # Capture a RAW frame without any zoom
      raw_size = capture_request_utils.get_available_output_sizes(
          'raw', props)[0]
      output_surfaces = [{'format' : 'raw',
                          'width': raw_size[0],
                          'height': raw_size[1]}]
      if self.hidden_physical_id:
        output_surfaces[0].update({'physicalCamera' : self.hidden_physical_id})
      imgs = {}
      cam.do_3a(out_surfaces=output_surfaces)
      req = capture_request_utils.auto_capture_request()
      req['android.statistics.lensShadingMapMode'] = (
          image_processing_utils.LENS_SHADING_MAP_ON)
      cap_raw_full = cam.do_capture(
          req,
          output_surfaces,
          reuse_session=True)
      rgb_full_img = image_processing_utils.convert_raw_capture_to_rgb_image(
          cap_raw_full, props, 'raw', name_with_log_path)
      image_processing_utils.write_image(
          rgb_full_img, f'{name_with_log_path}_raw_full.jpg')
      imgs['raw_full'] = rgb_full_img
      output_surfaces[0].update({'useCase' : its_session_utils.USE_CASE_CROPPED_RAW})
      # Capture RAW images with different zoom ratios with stream use case
      # CROPPED_RAW set
      for _, z in enumerate(z_list):
        req['android.control.zoomRatio'] = z
        cam.do_3a(out_surfaces=output_surfaces)
        cap_zoomed_raw = cam.do_capture(
            req,
            output_surfaces,
            reuse_session=True)
        rgb_zoomed_raw = (
            image_processing_utils.convert_raw_capture_to_rgb_image(
                cap_zoomed_raw, props, 'raw', name_with_log_path))
        # Dump zoomed in RAW image
        img_name = f'{name_with_log_path}_zoomed_raw_{z:.2f}.jpg'
        image_processing_utils.write_image(rgb_zoomed_raw, img_name)
        size_raw = [cap_zoomed_raw['width'], cap_zoomed_raw['height']]
        logging.debug('Finding center circle for zoom %f: size [%d x %d],'
                      ' (min zoom %f)', z, cap_zoomed_raw['width'],
                      cap_zoomed_raw['height'], z_list[0])
        meta = cap_zoomed_raw['metadata']
        result_raw_crop_region = meta['android.scaler.rawCropRegion']
        rl = result_raw_crop_region['left']
        rt = result_raw_crop_region['top']
        # Make sure that scale factor for width and height scaling is the same.
        rw = result_raw_crop_region['right'] - rl
        rh = result_raw_crop_region['bottom'] - rt
        logging.debug('RAW_CROP_REGION reported for zoom %f: [%d %d %d %d]',
                      z, rl, rt, rw, rh)
        # Effective zoom ratio. May not be == z since its possible the HAL
        # wasn't able to crop RAW.
        effective_zoom_ratio = aw / rw
        logging.debug('Effective zoom ratio: %f', effective_zoom_ratio)
        inv_scale_factor = rw / aw
        if aw / rw != ah / rh:
          raise AssertionError('RAW_CROP_REGION width and height aspect ratio'
                               f' != active array AR, region size: {rw} x {rh} '
                               f' active array size: {aw} x {ah}')
        # Find FoV to determine minimum circle size for
        # find_center_circle's parameter
        fov_ratio = zoom_capture_utils._DEFAULT_FOV_RATIO
        if self.hidden_physical_id is not None:
          logical_cam_fov = float(cam.calc_camera_fov(logical_props))
          cam_fov = float(cam.calc_camera_fov(props))
          logging.debug('Logical camera FoV: %f', logical_cam_fov)
          logging.debug(
              'Camera %s under test FoV: %f', self.hidden_physical_id, cam_fov)
          if cam_fov > logical_cam_fov:
            fov_ratio = logical_cam_fov / cam_fov
        # Find the center circle in img
        circle = zoom_capture_utils.find_center_circle(
            rgb_zoomed_raw, img_name, size_raw, effective_zoom_ratio,
            z_list[0], fov_ratio=fov_ratio, debug=True)
        # Zoom is too large to find center circle, break out
        if circle is None:
          break

        xnorm = rl / aw
        ynorm = rt / ah
        wnorm = rw / aw
        hnorm = rh / ah
        logging.debug('Image patch norm for zoom %.2f: [%.2f %.2f %.2f %.2f]',
                      z, xnorm, ynorm, wnorm, hnorm)
        # Crop the full FoV RAW to result_raw_crop_region
        rgb_full_cropped = image_processing_utils.get_image_patch(
            rgb_full_img, xnorm, ynorm, wnorm, hnorm)

        # Downscale the zoomed-in RAW image returned by the camera sub-system
        rgb_zoomed_downscale = cv2.resize(
            rgb_zoomed_raw, None, fx=inv_scale_factor, fy=inv_scale_factor)

        # Debug dump images being rms compared
        img_name_downscaled = f'{name_with_log_path}_downscale_raw_{z:.2f}.jpg'
        image_processing_utils.write_image(
            rgb_zoomed_downscale, img_name_downscaled)

        img_name_cropped = f'{name_with_log_path}_full_cropped_raw_{z:.2f}.jpg'
        image_processing_utils.write_image(rgb_full_cropped, img_name_cropped)

        rms_diff = image_processing_utils.compute_image_rms_difference_3d(
            rgb_zoomed_downscale, rgb_full_cropped)
        msg = f'RMS diff for CROPPED_RAW use case: {rms_diff:.4f}'
        logging.debug('%s', msg)
        if rms_diff >= _THRESHOLD_MAX_RMS_DIFF_CROPPED_RAW_USE_CASE:
          raise AssertionError('RMS diff of downscaled cropped RAW & full > 1%')


if __name__ == '__main__':
  test_runner.main()
