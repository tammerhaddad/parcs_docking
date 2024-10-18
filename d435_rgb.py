# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import abc
from typing import Tuple

import numpy as np
import pyrealsense2 as rs


WIDTH, HEIGHT, FPS = 1920, 1080, 15
#WIDTH, HEIGHT, FPS = 1280, 720, 15
#WIDTH, HEIGHT, FPS = 640, 480, 30
# WIDTH, HEIGHT, FPS = 640, 480, 15


class Realsense(abc.ABC):
    def __init__(self, exposure: str = "auto"):
        self.exposure = exposure
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = self.pipeline.start(self.config)

    def get_image(self) -> Tuple[np.ndarray]:
        depth_frame, color_frame = self.get_frames()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def get_message(self) -> dict:
        """Get a message that can be sent via ZMQ"""
        color_camera_info = self.get_camera_info()
        color_image = self.get_image()
        realsense_output = {
            "color_camera_info": color_camera_info,
            "color_image": color_image,
        }
        return realsense_output

    def wait_for_frames(self):
        return self.pipeline.wait_for_frames()

    def get_frame(self):
        frames = self.wait_for_frames()
        color_frame = frames.get_color_frame()
        return color_frame

    def read_camera_info(self):
        color_frame = self.get_frame()
        return get_camera_info(color_frame)

    def get_camera_info(self):
        raise NotImplementedError()


def get_camera_info(frame):
    """Get camera info for a realsense"""
    intrinsics = rs.video_stream_profile(frame.profile).get_intrinsics()

    # from Intel's documentation
    # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.intrinsics.html#pyrealsense2.intrinsics
    # "
    # coeffs	Distortion coefficients
    # fx	Focal length of the image plane, as a multiple of pixel width
    # fy	Focal length of the image plane, as a multiple of pixel height
    # height	Height of the image in pixels
    # model	Distortion model of the image
    # ppx	Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
    # ppy	Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
    # width	Width of the image in pixels
    # "

    # out = {
    #     'dist_model' : intrinsics.model,
    #     'dist_coeff' : intrinsics.coeffs,
    #     'fx' : intrinsics.fx,
    #     'fy' : intrinsics.fy,
    #     'height' : intrinsics.height,
    #     'width' : intrinsics.width,
    #     'ppx' : intrinsics.ppx,
    #     'ppy' : intrinsics.ppy
    #     }

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ]
    )

    distortion_model = intrinsics.model

    distortion_coefficients = np.array(intrinsics.coeffs)

    camera_info = {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion_coefficients,
        "distortion_model": distortion_model,
    }

    return camera_info



class D435i(Realsense):
    """Wrapper for accessing data from a D435 realsense camera, used as the head camera on Stretch RE1, RE2, and RE3."""

    def __init__(self, exposure: str = "auto", camera_number: int = 0):
        print("Connecting to D435i and getting camera info...")
        self._setup_camera(exposure=exposure, number=camera_number)
        self.color_camera_info = self.read_camera_info()
        print(f"  color camera: {self.color_camera_info}")

    def get_camera_info(self):
        return self.color_camera_info

    def get_image(self) -> Tuple[np.ndarray]:
        """Get a pair of numpy arrays for the images we want to use."""

        # Get the frames from the realsense
        frames = self.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def _setup_camera(self, exposure: str = "auto", number: int = 0):
        """
        Args:
            number(int): which camera to pick in order.
        """

        camera_info = [
            {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
            }
            for device in rs.context().devices
        ]
        print("Searching for D435i...")
        d435i_infos = []
        for i, info in enumerate(camera_info):
            print(i, info["name"], info["serial_number"])
            if "D435I" in info["name"]:
                d435i_infos.append(info)

        if len(d435i_infos) == 0:
            raise RuntimeError("could not find any supported d435i cameras")

        self.exposure = exposure
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Specifically enable the camera we want to use - make sure it's d435i
        self.config.enable_device(d435i_infos[number]["serial_number"])
        self.config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(self.config)

        if exposure == "auto":
            # Use autoexposre
            self.stereo_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            self.stereo_sensor.set_option(rs.option.enable_auto_exposure, True)
        else:
            default_exposure = 33000
            if exposure == "low":
                exposure_value = int(default_exposure / 3.0)
            elif exposure == "medium":
                exposure_value = 30000
            else:
                exposure_value = int(exposure)

            self.stereo_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            self.stereo_sensor.set_option(rs.option.exposure, exposure_value)


if __name__ == "__main__":
    camera = D435i()
    print(camera.get_message())
