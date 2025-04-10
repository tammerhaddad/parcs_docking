import d435_rgb as dc
import d405_helpers as dh
import numpy as np
import cv2
import normalized_velocity_control as nvc
# OLD
import stretch_body.robot as rb
import time
import aruco_detector as ad
import yaml
from yaml.loader import SafeLoader
from hello_helpers import hello_misc as hm
import argparse
import loop_timer as lt
import pprint as pp
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from trh_msgs.action import StringAction

####################################
# Miscellaneous Parameters

motion_on = True
print_timing = False #True

# Defines a deadzone for mobile base rotation, since low values can
# lead to no motion and noises on some surfaces like carpets.
min_base_speed = 0.0 #0.05

successful_pre_docking_err_m = 0.01
successful_pre_docking_err_ang = 0.05
successful_rotate_err_ang = 0.008 #0.01
successful_pan_err = 0.2

pre_docking_distance_m = 0.65 #0.7 #0.55 #0.63 #0.5

####################################
## Gains for Visual Servoing

overall_visual_servoing_velocity_scale = 0.02 #0.01 #1.0

joint_visual_servoing_velocity_scale = {
    'base_forward' : 0.1, #15.0
    'base_counterclockwise' : 400.0,
    'head_pan_counterclockwise' : 1.8 #2.0
}

####################################
## Initial Pose

initial_joint_state = {
    'head_pan_pos': -(np.pi + np.pi/4.0), #-np.pi,
    'head_tilt_pos': (-np.pi/2.0) + (np.pi/14.0), #(-np.pi/2.0) + (np.pi/10.0), 
    'lift_pos' : 0.3,
    'arm_pos': 0.01,
    'wrist_yaw_pos': (np.pi * (3.5/4.0)),
    'wrist_pitch_pos': 0.0,
    'wrist_roll_pos': 0.0,
    'gripper_pos': 10.46
}

####################################
## Allowed Ranges of Motion

min_joint_state = {
    'base_odom_theta' : -100.0, #-0.8,
    'base_odom_x' : -100.0, #-0.2
    'head_pan_pos' : -(np.pi + np.pi/4.0)
    }

max_joint_state = {
    'base_odom_theta' : 100.0, #0.8,
    'base_odom_x' : 100.0, #0.2
    'head_pan_pos' : np.pi * 3.0/4.0
    }


####################################
## Zero Velocity Command

zero_vel = {
    'base_forward': 0.0,
    'base_counterclockwise': 0.0,
    'head_pan_counterclockwise': 0.0
}

####################################
## Translate Between Keys

pos_to_vel_cmd = {
    'base_odom_x' : 'base_forward', 
    'base_odom_theta' : 'base_counterclockwise',
    'head_pan_pos' : 'head_pan_counterclockwise'
}

vel_cmd_to_pos = { v:k for (k,v) in pos_to_vel_cmd.items() }

####################################

class ServoingError:

    def __init__(self):
        self.direction = 0.0
        self.distance = 0.0
        self.pan = 0.0

    def __str__(self):
        return f'ServoingError: direction = {self.direction}, distance = {self.distance}, pan = {self.pan}'

        
class Marker:
    def __init__(self, marker_dict):
        self.origin = np.copy(marker_dict['pos'])
        self.x_axis = np.copy(marker_dict['x_axis'])
        self.y_axis = np.copy(marker_dict['y_axis'])
        self.z_axis = np.copy(marker_dict['z_axis'])

        
class CoordSys:

    def __init__(self, origin, x_axis, y_axis):
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis

    def point_in_image(self, dock_x_coord_m, dock_y_coord_m, camera_info):

        point_xyz = (dock_x_coord_m * self.x_axis) + (dock_y_coord_m * self.y_axis) + self.origin
        image_point_xy = dh.pixel_from_3d(point_xyz, camera_info)
        return image_point_xy
        
    def vector_in_image(self, dock_x_coord_m, dock_y_coord_m, camera_info):
        end_xy = self.point_in_image(dock_x_coord_m, dock_y_coord_m, camera_info)
        start_xy = dh.pixel_from_3d(self.origin, camera_info)
        vec_xy = end_xy - start_xy
        return vec_xy

    def draw(self, image, camera_info):

        # Draw origin
        origin_xy = dh.pixel_from_3d(self.origin, camera_info)
        radius = 6
        thickness = -1
        color = [255, 0, 0]
        origin_xy = np.round(origin_xy).astype(np.int32)
        cv2.circle(image, origin_xy, radius, color, -1, lineType=cv2.LINE_AA)

        start_xy = origin_xy
        length_m = 0.15

        # Draw x-axis in red
        thickness = 2
        color = [0, 0, 255]
        end_xy = dh.pixel_from_3d(self.origin + (length_m * self.x_axis), camera_info)
        end_xy = np.round(end_xy).astype(np.int32)
        cv2.line(image, start_xy, end_xy, color, thickness, lineType=cv2.LINE_AA)

        # Draw y-axis in green
        thickness = 2
        color = [0, 255, 0]
        end_xy = dh.pixel_from_3d(self.origin + (length_m * self.y_axis), camera_info)
        end_xy = np.round(end_xy).astype(np.int32)
        cv2.line(image, start_xy, end_xy, color, thickness, lineType=cv2.LINE_AA)

        
class DockCoordSys(CoordSys):

    def __init__(self, dock_marker, base_marker):

        ####################################
        # This estimates a planar coordinate system for the docking station that is used to define image-space objectives for visual servoing, including position way points.
        #
        # The origin of the coordinate system is the center of the docking station's ArUco marker.
        #
        # The x and y axes for the coordinate system sit in the plane defined by the x-axis for the docking station's ArUco marker and the vector that connects the center of the docking station's ArUco marker and the center of the mobile base's ArUco marker.
        #
        # The y-axis points along the midline of the docking station in the same direction as the charging connector.
        #
        # The x-axis points to the left of the docking station when you are looking at the ArUco marker from in front of the docking station.
        #
        # The x-axis and y-axis form a right-handed coordinate system where the z-axis is normal to the floor.
        #
        ####################################

        # The origin is the 3D origin of the docking station's ArUco marker
        #self.origin = np.copy(dock_marker.origin) - (0.01 * dock_marker.y_axis)
        self.origin = np.copy(dock_marker.origin)
        
        # The docking station's x-axis is the negative of the docking station's ArUco marker's x-axis
        self.x_axis = -dock_marker.x_axis

        # Find the docking station's y-axis by using its x-axis and the vector that connects the origins of the docking station's ArUco marker and the mobile base's ArUco marker. This should result in a less noisy estimate, as long as they are not parallel, which should be unlikely during docking and can be explicitly detected and excluded. 

        diff = base_marker.origin - self.origin
        diff_mag = np.linalg.norm(diff)
        if diff_mag > 0.0: 
            diff = diff / diff_mag

            # Remove the component of diff that is non-orthogonal to the x-axis
            self.y_axis = diff - (np.dot(diff, self.x_axis) * self.x_axis)
            y_axis_mag = np.linalg.norm(self.y_axis)

            if y_axis_mag > 0.0:
                self.y_axis = self.y_axis / y_axis_mag
            else:
                self.y_axis = None
        else:
            self.y_axis = None

                   
class BaseCoordSys(CoordSys):

    def __init__(self, base_marker):
        
        ####################################
        # This estimates a planar coordinate system for the mobile base that is used to define image-space objectives for visual servoing.
        #
        # The origin of the coordinate system is the center of rotation for the mobile base.
        #
        # The x and y axes for the coordinate system sit in the plane and are derived from the x and y axes of the ArUco marker mounted to the left front corner of the mobile base.
        #
        # The y-axis points along the midline of the mobile base towards the back of the robot
        #
        # The x-axis points to the right of the robot when you're looking at the the robot from the front.
        #
        # The x-axis and y-axis form a right-handed coordinate system where the z-axis is normal to the floor.
        #
        ####################################

        self.origin = base_marker.origin + ((0.13) * base_marker.x_axis)
        self.y_axis = -base_marker.y_axis
        self.x_axis = -base_marker.x_axis
            

def display_visual_servoing_features(center_xy, midline_xy, image, color=None, length=100.0):
    if (center_xy is None) or (midline_xy is None):
        return

    radius = 6
    thickness = -1
    if color is None:
        color = [255, 255, 0]
    center = np.round(center_xy).astype(np.int32)
    cv2.circle(image, center, radius, color, -1, lineType=cv2.LINE_AA)

    radius = 6
    thickness = 2
    if color is None: 
        color = [255, 0, 0]
    start = center
    end = np.round(center_xy + (length * midline_xy)).astype(np.int32)
    cv2.line(image, start, end, color, thickness, lineType=cv2.LINE_AA)


def vector_error(target, current):
    err_mag = 1.0 - np.dot(target, current)
    err_sign = np.sign(np.cross(target, current))
    err = err_sign * err_mag
    return err

def get_pix_per_m(camera_info):
    # Set the pixel per meter conversion value for the
    # current image resolution. Success is sensitive
    # to this value, so it's better to set it to a
    # constant value instead of estimating it from the
    # ArUco markers.

    # PLEASE NOTE THAT THIS IS AN APPROXIMATION THAT DOES NOT TAKE
    # INTO ACCOUNT THE TILT OF THE CAMERA AND THUS DOES NOT ACCOUNT
    # FOR VARIATION ACROSS THE IMAGE DUE TO THE FLOOR PLANE BEING AT
    # AN ANGLE RELATIVE TO THE IMAGE.
    
    fx = camera_info['camera_matrix'][0,0]
    pix_per_m = fx * (1050.0/1362.04443)
    print(f'{fx=}')
    print(f'{pix_per_m=}')
    return pix_per_m


def pre_docking_center(dock_origin_marker_xyz, dock_y_axis_marker_xyz, camera_info):
    pre_docking_center_marker_xyz = (pre_docking_distance_m * dock_y_axis_marker_xyz) + dock_origin_marker_xyz
    pre_docking_center_xy = dh.pixel_from_3d(pre_docking_center_marker_xyz, camera_info)
    return pre_docking_center_xy


def docking_pose(dock_coord_sys, base_coord_sys):
    side = np.dot(dock_coord_sys.x_axis, base_coord_sys.origin - dock_coord_sys.origin)
    if side > 0.0:
        left_of_dock = True
    else:
        left_of_dock = False

    front = np.dot(dock_coord_sys.x_axis, base_coord_sys.y_axis)
    if left_of_dock: 
        if front < 0.0:
            facing_dock = True 
        else:
            facing_dock = False 
    else:
        if front < 0.0:
            facing_dock = False
        else:
            facing_dock = True

    return facing_dock, left_of_dock


def move_to_initial_pose(robot):
    robot.head.move_to('head_pan', initial_joint_state['head_pan_pos'])
    robot.head.move_to('head_tilt', initial_joint_state['head_tilt_pos'])
    robot.push_command()
    robot.wait_command()

    robot.end_of_arm.get_joint('wrist_yaw').move_to(initial_joint_state['wrist_yaw_pos'])
    robot.end_of_arm.get_joint('wrist_pitch').move_to(initial_joint_state['wrist_pitch_pos'])
    robot.push_command()
    robot.wait_command()

    robot.arm.move_to(initial_joint_state['arm_pos'])
    robot.push_command()
    robot.wait_command()

    robot.lift.move_to(initial_joint_state['lift_pos'])
    robot.push_command()
    robot.wait_command()

    robot.end_of_arm.get_joint('stretch_gripper').move_to(initial_joint_state['gripper_pos'])
    robot.push_command()
    robot.wait_command()
        

####################################################################
# behaviors

def look_for_markers(dock_coord_sys, base_coord_sys, servoing_error):
    # The look_for_markers behavior holds the keeps the mobile base staionary while it pans the D435if head until it sees the dock and mobile base ArUco markers
    servoing_error.distance = 0.0
    servoing_error.direction = 0.0
    servoing_error.pan = -2.0
    next_behavior = 'look_for_markers'
    if (dock_coord_sys is not None) and (base_coord_sys is not None):
        facing_dock, left_of_dock = docking_pose(dock_coord_sys, base_coord_sys)
        print(f'{facing_dock=}')
        print(f'{left_of_dock=}')
        if not left_of_dock:
            # Sometimes there are false positives due to noise, so this should only print a warning.
            #raise NotImplementedError('THE DOCKING DEMO DOES NOT YET WORK WHEN DOCKING FROM THE RIGHT SIDE OF THE DOCKING STATION.')
            print('WARNING: THE DOCKING DEMO DOES NOT YET WORK WHEN DOCKING FROM THE RIGHT SIDE OF THE DOCKING STATION.')
        if facing_dock:
            next_behavior = 'rotate_to_predocking_position'
        else: 
            next_behavior = 'rotate_to_starting_pose'
    return next_behavior


def rotate_to_starting_pose(dock_coord_sys, base_coord_sys, camera_info, servoing_error):
    servoing_error.distance = 0.0
    servoing_error.direction = 0.0
    
    next_behavior = 'rotate_to_starting_pose'
    
    dock_center_xy = dock_coord_sys.point_in_image(0.0, 0.0, camera_info)
    base_center_xy = base_coord_sys.point_in_image(0.0, 0.0, camera_info)
    center_diff_xy = dock_center_xy - base_center_xy
    center_diff_xy = center_diff_xy / np.linalg.norm(center_diff_xy)

    base_midline_xy = base_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    base_midline_xy = base_midline_xy / np.linalg.norm(base_midline_xy)
                                                       
    facing_dock, left_of_dock = docking_pose(dock_coord_sys, base_coord_sys)
    if facing_dock:
        next_behavior = 'rotate_to_predocking_position'
        
    if left_of_dock:
        servoing_error.direction = 1.0
    else:
        servoing_error.direction = -1.0

    return next_behavior


def keep_markers_in_view(dock_coord_sys, base_coord_sys, camera_info, servoing_error):
    dock_center_xy = dock_coord_sys.point_in_image(0.0, 0.0, camera_info)
    base_center_xy = base_coord_sys.point_in_image(0.0, 0.0, camera_info)
    pan_goal = dock_center_xy - base_center_xy
    pan_goal = pan_goal / np.linalg.norm(pan_goal)
    pan_curr = np.array([1.0, 0.0])
    servoing_error.pan = vector_error(pan_goal, pan_curr)
    if abs(servoing_error.pan) < successful_pan_err:
        servoing_error.pan = 0.0


def rotate_to_predocking_position(dock_coord_sys, base_coord_sys, camera_info, color_image, servoing_error):
    servoing_error.distance = 0.0
    servoing_error.direction = 0.0
    
    next_behavior = 'rotate_to_predocking_position'
    
    facing_dock, left_of_dock = docking_pose(dock_coord_sys, base_coord_sys)
    print(f'{facing_dock=}')
    print(f'{left_of_dock=}')

    #pre_docking_center_xy = pre_docking_center(dock_center_xy, dock_midline_xy, pix_per_m)

    base_center_xy = base_coord_sys.point_in_image(0.0, 0.0, camera_info)
    base_midline_xy = base_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    base_midline_xy = base_midline_xy / np.linalg.norm(base_midline_xy)

    dock_center_xy = dock_coord_sys.point_in_image(0.0, 0.0, camera_info)
    dock_midline_xy = dock_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    dock_midline_xy = dock_midline_xy / np.linalg.norm(dock_midline_xy)
    
    pre_docking_center_xy = pre_docking_center(dock_coord_sys.origin, dock_coord_sys.y_axis, camera_info)
    pre_docking_center_xy = dock_coord_sys.point_in_image(0.0, pre_docking_distance_m, camera_info)
    pre_docking_midline_xy = dock_midline_xy

    display_visual_servoing_features(pre_docking_center_xy, pre_docking_midline_xy, color_image)

    # find error to the pre-docking waypoint location
    direction = pre_docking_center_xy - base_center_xy

    display_visual_servoing_features(base_center_xy, direction, color_image, [0,0,255], 1.0)

    distance = np.linalg.norm(direction)
    if distance > 0.0:
        direction = direction / distance

    direction_success = False

    if left_of_dock: 
        servoing_error.direction = vector_error(direction, base_midline_xy)
    else:
        servoing_error.direction = vector_error(direction, -base_midline_xy)
            
    if abs(servoing_error.direction) < (2.0 * successful_rotate_err_ang):
        servoing_error.direction = 0.0
        next_behavior = 'move_to_predocking_position'

    return next_behavior


def move_to_predocking_position(dock_coord_sys, base_coord_sys, camera_info, color_image, servoing_error):
    servoing_error.distance = 0.0
    servoing_error.direction = 0.0
    
    next_behavior = 'move_to_predocking_position'
    
    facing_dock, left_of_dock = docking_pose(dock_coord_sys, base_coord_sys)
    print(f'{facing_dock=}')
    print(f'{left_of_dock=}')

    #pre_docking_center_xy = pre_docking_center(dock_center_xy, dock_midline_xy, pix_per_m)

    base_center_xy = base_coord_sys.point_in_image(0.0, 0.0, camera_info)
    base_midline_xy = base_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    base_midline_xy = base_midline_xy / np.linalg.norm(base_midline_xy)

    dock_center_xy = dock_coord_sys.point_in_image(0.0, 0.0, camera_info)
    dock_midline_xy = dock_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    dock_midline_xy = dock_midline_xy / np.linalg.norm(dock_midline_xy)
    
    pre_docking_center_xy = pre_docking_center(dock_coord_sys.origin, dock_coord_sys.y_axis, camera_info)
    pre_docking_center_xy = dock_coord_sys.point_in_image(0.0, pre_docking_distance_m, camera_info)
    pre_docking_midline_xy = dock_midline_xy

    display_visual_servoing_features(pre_docking_center_xy, pre_docking_midline_xy, color_image)

    # find error to the pre-docking waypoint location
    direction = pre_docking_center_xy - base_center_xy

    display_visual_servoing_features(base_center_xy, direction, color_image, [0,0,255], 1.0)

    distance = np.linalg.norm(direction)
    if distance > 0.0:
        direction = direction / distance

    distance_success = False
    direction_success = False

    if left_of_dock: 
        servoing_error.distance = distance
    else:
        servoing_error.distance = -distance

    # This needs work
    pix_per_m = get_pix_per_m(camera_info)
    if abs(servoing_error.distance) < (pix_per_m * successful_pre_docking_err_m):
        servoing_error.distance = 0.0
        distance_success = True

    if not distance_success:
        if left_of_dock: 
            servoing_error.direction = vector_error(direction, base_midline_xy)
        else:
            servoing_error.direction = vector_error(direction, -base_midline_xy)
        if (abs(servoing_error.direction) < successful_pre_docking_err_ang):
            servoing_error.direction = 0.0
            direction_success = True
    else:
        servoing_error.direction = 0.0
        direction_success = True

    if direction_success and distance_success:
        next_behavior = 'rotate_for_docking'

    return next_behavior


def rotate_for_docking(dock_coord_sys, base_coord_sys, camera_info, servoing_error):
    servoing_error.distance = 0.0
    servoing_error.direction = 0.0
    next_behavior = 'rotate_for_docking'

    base_midline_xy = base_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    base_midline_xy = base_midline_xy / np.linalg.norm(base_midline_xy)

    dock_midline_xy = dock_coord_sys.vector_in_image(0.0, 1.0, camera_info)
    dock_midline_xy = dock_midline_xy / np.linalg.norm(dock_midline_xy)

    # find rotational error to make the base midline parallel to the pre-docking direction
    parallel_err = vector_error(-dock_midline_xy, base_midline_xy)
    servoing_error.direction = 2.0 * parallel_err
    if abs(servoing_error.direction) < successful_rotate_err_ang:
        servoing_error.direction = 0.0
        next_behavior = 'back_into_dock'

    return next_behavior


def back_into_dock(joint_state, servoing_error):
    # The back_into_dock behavior slowly drives backwards until the battery charging status is true. It currently drives backwards without any visual feedback. It also does not currently handle failure conditions. In the future, it would be good for it to at least have a timeout. It might also check for the distance traveled and drive wheel effort. When the mobile base is close to being fully docked, the robot's shoulder occludes the docking station's ArUco marker from the view of the D435if head camera.  
    servoing_error.distance = 0.0
    servoing_error.direction = 0.0
    servoing_error.pan = 0.0
    next_behavior = 'back_into_dock'
    
    battery_charging = joint_state['battery_charging']
    print()
    print(f'{battery_charging=}')
    if battery_charging:
        print('FINISHED DOCKING!')
        next_behavior = 'docked'
    else:
        servoing_error.distance = 100.0

    return next_behavior

####################################################################

def run(exposure):

    try:
        camera = dc.D435i(exposure=exposure)

        time.sleep(1.0)
        
        robot = rb.Robot()
        robot.startup()
        move_to_initial_pose(robot)

        marker_info = {}
        with open('aruco_marker_info.yaml') as f:
            marker_info = yaml.load(f, Loader=SafeLoader)

        aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=True, use_apriltag_refinement=False, brighten_images=True)

        controller = nvc.NormalizedVelocityControl(robot)
        controller.reset_base_odometry()
        
        loop_timer = lt.LoopTimer()

        behavior = 'look_for_markers'

        while behavior != 'docked':
            print('_______________________________________')
                            
            loop_timer.start_of_iteration()

            camera_info = camera.get_camera_info()
            pix_per_m = get_pix_per_m(camera_info)
            
            color_image = camera.get_image()
                
            aruco_detector.update(color_image, camera_info)

            markers = aruco_detector.get_detected_marker_dict()

            base_marker = None
            dock_marker = None

            for k in markers:
                m = markers[k]
                name = m['info']['name']
                if name == 'base_left':
                    base_marker = Marker(m)
                if name == 'docking_station':
                    dock_marker = Marker(m)
                    
            dock_coord_sys = None
            if (base_marker is not None) and (dock_marker is not None):
                dock_coord_sys = DockCoordSys(dock_marker, base_marker)
                dock_coord_sys.draw(color_image, camera_info)

            base_coord_sys = None
            if base_marker is not None:
                base_coord_sys = BaseCoordSys(base_marker)
                base_coord_sys.draw(color_image, camera_info)

            servoing_error = ServoingError()
            joint_state = controller.get_joint_state()
            # convert base odometry angle to be in the range -pi to pi
            joint_state['base_odom_theta'] = hm.angle_diff_rad(joint_state['base_odom_theta'], 0.0)

            print(f'{behavior=}')
                
            if behavior == 'look_for_markers':
                
                behavior = look_for_markers(dock_coord_sys, base_coord_sys, servoing_error)
                
            elif behavior == 'back_into_dock':
                
                behavior = back_into_dock(joint_state, servoing_error)
                
            elif (dock_coord_sys is not None) and (base_coord_sys is not None): 

                keep_markers_in_view(dock_coord_sys, base_coord_sys, camera_info, servoing_error)

                if behavior == 'rotate_to_starting_pose':

                    behavior = rotate_to_starting_pose(dock_coord_sys, base_coord_sys, camera_info, servoing_error)

                if behavior == 'rotate_to_predocking_position':

                    behavior = rotate_to_predocking_position(dock_coord_sys, base_coord_sys, camera_info, color_image, servoing_error)
                
                if behavior == 'move_to_predocking_position':

                    behavior = move_to_predocking_position(dock_coord_sys, base_coord_sys, camera_info, color_image, servoing_error)

                elif behavior == 'rotate_for_docking':

                    behavior = rotate_for_docking(dock_coord_sys, base_coord_sys, camera_info, servoing_error)

            # Set joint velocities based on the servoing error
            print()
            print(servoing_error)

            print()
            base_rotational_velocity = servoing_error.direction
            if abs(base_rotational_velocity) < min_base_speed:
                base_rotational_velocity = 0.0

            base_translational_velocity = -servoing_error.distance
            if abs(base_translational_velocity) < min_base_speed:
                base_translational_velocity = 0.0

            head_pan_velocity = -servoing_error.pan

            cmd = {}
            cmd['base_forward'] = base_translational_velocity
            cmd['base_counterclockwise'] = base_rotational_velocity
            cmd['head_pan_counterclockwise'] = head_pan_velocity 

            print('cmd before scaling velocities')
            pp.pprint(cmd)

            cmd = {k: overall_visual_servoing_velocity_scale * v for (k,v) in cmd.items()}
            cmd = {k: joint_visual_servoing_velocity_scale[k] * v for (k,v) in cmd.items()}

            if motion_on:
                print()
                print('cmd before checking joint limits')
                pp.pprint(cmd)
                cmd = { k: ( 0.0 if ((v < 0.0) and (joint_state[vel_cmd_to_pos[k]] < min_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}
                cmd = { k: ( 0.0 if ((v > 0.0) and (joint_state[vel_cmd_to_pos[k]] > max_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}

                print()
                print('cmd before being executed')
                pp.pprint(cmd)
                controller.set_command(cmd)

            cv2.imshow('Features Used for Visual Servoing', color_image)
            cv2.waitKey(1)

            loop_timer.end_of_iteration()
            if print_timing: 
                loop_timer.pretty_print()

    finally:

        robot.stop()

class DockingActionServer(Node):

    def __init__(self, exposure):
        super().__init__('docking_action_server')
        self._action_server = ActionServer(
            self,
            StringAction,  # Replace with your custom action if needed
            'docking_action',
            self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing docking action...')
        try:
            run(goal_handle.request.exposure)  # Pass exposure if needed
            goal_handle.succeed()
            return StringAction.Result(success=True, strresult="Docking completed successfully.")
        except Exception as e:
            self.get_logger().error(f'Docking failed: {e}')
            goal_handle.abort()
            return StringAction.Result(success=False, strresult=f"Docking failed: {e}")
        
def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exposure', type=str, required=True, help='Exposure setting for the D435 camera')
    parsed_args = parser.parse_args()
    node = DockingActionServer(parsed_args.exposure)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Stretch 3 Docking Demo',
        description='This application provides a demonstration of using visual servoing to autonomously dock with the official Hello Robot docking station.')

    parser.add_argument('-e', '--exposure', action='store', type=str, default='auto', help=f'Set the D435 exposure to {dh.exposure_keywords} or an integer in the range {dh.exposure_range}') 
    
    args = parser.parse_args()
    exposure = args.exposure

    if not dh.exposure_argument_is_valid(exposure):
        raise argparse.ArgumentTypeError(f'The provided exposure setting, {exposure}, is not a valide keyword, {dh.exposure_keywords}, or is outside of the allowed numeric range, {dh.exposure_range}.')    
    
    main(exposure)