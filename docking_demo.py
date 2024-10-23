import d435_rgb as dc
import d405_helpers as dh
import numpy as np
import cv2
import normalized_velocity_control as nvc
import stretch_body.robot as rb
import time
import aruco_detector as ad
import yaml
from yaml.loader import SafeLoader
from hello_helpers import hello_misc as hm
import argparse
import loop_timer as lt
import pprint as pp


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

pre_docking_distance_m = 0.55 #0.63 #0.5

####################################
## Gains for Visual Servoing

overall_visual_servoing_velocity_scale = 0.02 #0.01 #1.0

joint_visual_servoing_velocity_scale = {
    'base_forward' : 0.1, #15.0
    'base_counterclockwise' : 400.0,
    'head_pan_counterclockwise' : 2.0
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


######################################
# Naming conventions
#
# _marker_xyz is a three dimensional quantity derived from an ArUco marker
# _xy is a two dimensional quantity defined with respect to the image
#
######################################

def center_and_midline_in_image(center_marker_xyz, midline_marker_xyz, camera_info):
    if (center_marker_xyz is None) or (midline_marker_xyz is None):
        return None, None
    
    center_xy = dh.pixel_from_3d(center_marker_xyz, camera_info)
    
    length = 1.0
    end_marker_xyz = center_marker_xyz + (length * midline_marker_xyz)
    end_xy = dh.pixel_from_3d(end_marker_xyz, camera_info)
    
    midline_xy = (end_xy - center_xy)
    midline_xy_mag = np.linalg.norm(midline_xy)
    if midline_xy_mag > 0.0:
        midline_xy = midline_xy / midline_xy_mag
    else:
        midline_xy = None

    return center_xy, midline_xy


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
    

def dock_coord_sys(dock_origin_marker_xyz, dock_x_axis_marker_xyz, dock_y_axis_marker_xyz,
                   base_origin_marker_xyz, base_x_axis_marker_xyz, base_y_axis_marker_xyz, base_z_axis_marker_xyz):
    
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

    if (((dock_origin_marker_xyz is None) or (dock_x_axis_marker_xyz is None) or (dock_y_axis_marker_xyz is None)) or
        ((base_origin_marker_xyz is None)) or (base_x_axis_marker_xyz is None) or (base_y_axis_marker_xyz is None) or
         (base_z_axis_marker_xyz is None)):
        return None, None, None
    
    # The origin is the 3D origin of the docking station's ArUco marker
    origin = np.copy(dock_origin_marker_xyz)

    # The docking station's x-axis is the negative of the docking station's ArUco marker's x-axis
    x_axis = -dock_x_axis_marker_xyz

    # Find the docking station's y-axis by using its x-axis and the vector that connects the origins of the docking station's ArUco marker and the mobile base's ArUco marker. This should result in a less noisy estimate, as long as they are not parallel, which should be unlikely during docking and can be explicitly detected and excluded. 

    diff = base_origin_marker_xyz - dock_origin_marker_xyz
    diff_mag = np.linalg.norm(diff)
    if diff_mag > 0.0: 
        diff = diff / diff_mag

        # Remove the component of diff that is non-orthogonal to the x-axis
        y_axis = diff - (np.dot(diff, x_axis) * x_axis)
        y_axis_mag = np.linalg.norm(y_axis)

        if y_axis_mag > 0.0:
            y_axis = y_axis / y_axis_mag
        else:
            y_axis = None
    else:
        y_axis = None

    print(f'{origin=}, {x_axis=}, {y_axis=}')
    return origin, x_axis, y_axis


def display_dock_coord_sys(origin, x_axis, y_axis, image, camera_info):
    if (origin is None) or (x_axis is None) or (y_axis is None):
        return

    # Draw origin
    origin_xy = dh.pixel_from_3d(origin, camera_info)
    radius = 6
    thickness = -1
    color = [255, 0, 0]
    origin_xy = np.round(origin_xy).astype(np.int32)
    cv2.circle(image, origin_xy, radius, color, -1, lineType=cv2.LINE_AA)

    start_xy = origin_xy
    length_m = 0.2
    
    # Draw x-axis in red
    thickness = 2
    color = [0, 0, 255]
    end_xy = dh.pixel_from_3d(origin + (length_m * x_axis), camera_info)
    end_xy = np.round(end_xy).astype(np.int32)
    cv2.line(image, start_xy, end_xy, color, thickness, lineType=cv2.LINE_AA)
    
    # Draw y-axis in green
    thickness = 2
    color = [0, 255, 0]
    end_xy = dh.pixel_from_3d(origin + (length_m * y_axis), camera_info)
    end_xy = np.round(end_xy).astype(np.int32)
    cv2.line(image, start_xy, end_xy, color, thickness, lineType=cv2.LINE_AA)
    

def pre_docking_center_2(dock_origin_marker_xyz, dock_y_axis_marker_xyz, camera_info):
    pre_docking_center_marker_xyz = (pre_docking_distance_m * dock_y_axis_marker_xyz) + dock_origin_marker_xyz
    pre_docking_center_xy = dh.pixel_from_3d(pre_docking_center_marker_xyz, camera_info)
    return pre_docking_center_xy


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

def pre_docking_center(dock_center_xy, dock_midline_xy, pix_per_m):
    # find the pre-docking waypoint
    dist_pix = pix_per_m * pre_docking_distance_m
    pre_docking_center_xy = dock_center_xy + (dist_pix * dock_midline_xy)
    return pre_docking_center_xy

def docking_pose(base_center_xy, base_midline_xy, dock_center_xy, dock_midline_xy, pix_per_m):
    pre_docking_center_xy = pre_docking_center(dock_center_xy, dock_midline_xy, pix_per_m)
    
    center_diff_xy = base_center_xy - pre_docking_center_xy
    dock_side_sign = np.sign(np.cross(dock_midline_xy, center_diff_xy))
    if dock_side_sign < 0.0:
        left_of_dock = False
    else:
        left_of_dock = True

    center_diff_xy = dock_center_xy - base_center_xy
    center_diff_xy = center_diff_xy / np.linalg.norm(center_diff_xy)
    if left_of_dock:
        abs_direction_err = abs(1.0 - np.dot(center_diff_xy, base_midline_xy))
        facing_sign = np.dot(base_midline_xy, dock_midline_xy)
    else: 
        abs_direction_err = abs(1.0 - np.dot(center_diff_xy, -base_midline_xy))
        facing_sign = np.dot(-base_midline_xy, dock_midline_xy)

    if (facing_sign > 0.0) and (abs_direction_err < (1.0 - np.cos(np.pi/2.0))):
        facing_dock = True
    else:
        facing_dock = False
        
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
        

def main(exposure):

    try:
        pix_per_m_av = None
        pix_per_m_n = 0
        
        camera = dc.D435i(exposure=exposure)

        time.sleep(1.0)
        
        robot = rb.Robot()
        robot.startup()
        move_to_initial_pose(robot)

        marker_info = {}
        with open('aruco_marker_info.yaml') as f:
            marker_info = yaml.load(f, Loader=SafeLoader)

        aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=True, use_apriltag_refinement=False, brighten_images=True)

        first_frame = True

        controller = nvc.NormalizedVelocityControl(robot)
        controller.reset_base_odometry()
        
        loop_timer = lt.LoopTimer()

        behaviors = ['look_for_markers', 'rotate_to_starting_pose', 'move_to_predocking_position', 'rotate_for_docking', 'back_into_dock', 'docked']
        behavior = 'look_for_markers'
        facing_dock = None
        left_of_dock = None

        while behavior != 'docked':
            print('_______________________________________')
                            
            loop_timer.start_of_iteration()

            camera_info = camera.get_camera_info()
            pix_per_m = get_pix_per_m(camera_info)
            
            color_image = camera.get_image()
                
            aruco_detector.update(color_image, camera_info)

            markers = aruco_detector.get_detected_marker_dict()
            base_center_marker_xyz = None
            base_midline_marker_xyz = None
            dock_center_marker_xyz = None
            dock_midline_marker_xyz = None

            base_origin_marker_xyz = None
            base_x_axis_marker_xyz = None
            base_y_axis_marker_xyz = None
            base_z_axis_marker_xyz = None
                   
            dock_origin_marker_xyz = None
            dock_x_axis_marker_xyz = None
            dock_y_axis_marker_xyz = None
            
            for k in markers:
                m = markers[k]
                name = m['info']['name']
                if name == 'base_left':
                    base_center_marker_xyz = m['pos'] + ((0.13) * m['x_axis'])
                    base_midline_marker_xyz = -m['y_axis']

                    base_origin_marker_xyz = np.copy(m['pos'])
                    base_x_axis_marker_xyz = np.copy(m['x_axis'])
                    base_y_axis_marker_xyz = np.copy(m['y_axis'])
                    base_z_axis_marker_xyz = np.copy(m['z_axis'])
                    
                if name == 'docking_station':
                    dock_center_marker_xyz = m['pos']
                    dock_midline_marker_xyz = -m['y_axis']
                    
                    dock_origin_marker_xyz = np.copy(m['pos'])
                    dock_x_axis_marker_xyz = np.copy(m['x_axis'])
                    dock_y_axis_marker_xyz = np.copy(m['y_axis'])


            dock_origin_marker_xyz, dock_x_axis_marker_xyz, dock_y_axis_marker_xyz = dock_coord_sys(dock_origin_marker_xyz,
                                                                                                    dock_x_axis_marker_xyz, dock_y_axis_marker_xyz,
                                                                                                    base_origin_marker_xyz, base_x_axis_marker_xyz,
                                                                                                    base_y_axis_marker_xyz, base_z_axis_marker_xyz)

            display_dock_coord_sys(dock_origin_marker_xyz, dock_x_axis_marker_xyz, dock_y_axis_marker_xyz, color_image, camera_info) 
                        
            # compute and display image-based task-relevant features for visual servoing
            base_center_xy, base_midline_xy = center_and_midline_in_image(base_center_marker_xyz, base_midline_marker_xyz, camera_info)
            display_visual_servoing_features(base_center_xy, base_midline_xy, color_image)
            
            dock_center_xy, dock_midline_xy = center_and_midline_in_image(dock_center_marker_xyz, dock_midline_marker_xyz, camera_info)
            display_visual_servoing_features(dock_center_xy, dock_midline_xy, color_image)

            direction_err = 0.0
            distance_err = 0.0
            pan_err = 0.0

            joint_state = controller.get_joint_state()
            # convert base odometry angle to be in the range -pi to pi
            joint_state['base_odom_theta'] = hm.angle_diff_rad(joint_state['base_odom_theta'], 0.0)

            print(f'{behavior=}')
                
            if behavior == 'look_for_markers':
                pan_err = -2.0
                if (base_center_xy is not None) and (base_midline_xy is not None) and (dock_center_xy is not None) and (dock_midline_xy is not None):
                    facing_dock, left_of_dock = docking_pose(base_center_xy, base_midline_xy, dock_center_xy, dock_midline_xy, pix_per_m)
                    print(f'{facing_dock=}')
                    print(f'{left_of_dock=}')
                    if not left_of_dock:
                        raise NotImplementedError('THE DOCKING DEMO DOES NOT YET WORK WHEN DOCKING FROM THE RIGHT SIDE OF THE DOCKING STATION.')
                    if facing_dock:
                        behavior = 'move_to_predocking_position'
                    else: 
                        behavior = 'rotate_to_starting_pose'
            elif behavior == 'back_into_dock':
                battery_charging = joint_state['battery_charging']
                print()
                print(f'{battery_charging=}')
                if battery_charging:
                    print('FINISHED DOCKING!')
                    behavior = 'docked'
                else:
                    distance_err = 100.0
            elif (base_center_xy is not None) and (base_midline_xy is not None) and (dock_center_xy is not None) and (dock_midline_xy is not None):
                pan_goal = dock_center_xy - base_center_xy
                pan_goal = pan_goal / np.linalg.norm(pan_goal)
                pan_curr = np.array([1.0, 0.0])
                pan_err = vector_error(pan_goal, pan_curr)
                if abs(pan_err) < successful_pan_err:
                    pan_err = 0.0

                if behavior == 'rotate_to_starting_pose':
                    center_diff_xy = dock_center_xy - base_center_xy
                    center_diff_xy = center_diff_xy / np.linalg.norm(center_diff_xy)
                                                        
                    if left_of_dock:
                        abs_direction_err = abs(1.0 - np.dot(center_diff_xy, base_midline_xy))
                        direction_err = 1.0
                    else:
                        abs_direction_err = abs(1.0 - np.dot(-center_diff_xy, base_midline_xy))
                        direction_err = -1.0

                    print(f'{abs_direction_err=}')
                    if abs_direction_err < (1.0 - np.cos(np.pi/2.0)):
                        direction_err = 0.0
                        behavior = 'move_to_predocking_position'
                        
                        
                if behavior == 'move_to_predocking_position':
                    #pre_docking_center_xy = pre_docking_center(dock_center_xy, dock_midline_xy, pix_per_m)
                    pre_docking_center_xy = pre_docking_center_2(dock_origin_marker_xyz, dock_y_axis_marker_xyz, camera_info)
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
                        distance_err = distance
                    else:
                        distance_err = -distance
                        
                    if abs(distance_err) < (pix_per_m * successful_pre_docking_err_m):
                        distance_err = 0.0
                        distance_success = True

                    if not distance_success:
                        if left_of_dock: 
                            direction_err = vector_error(direction, base_midline_xy)
                        else:
                            direction_err = vector_error(direction, -base_midline_xy)
                        if (abs(direction_err) < successful_pre_docking_err_ang):
                            direction_err = 0.0
                            direction_success = True
                    else:
                        direction_err = 0.0
                        direction_success = True
                        
                    if direction_success and distance_success:
                        behavior = 'rotate_for_docking'

                elif behavior == 'rotate_for_docking':
                    # find rotational error to make the base midline parallel to the pre-docking direction
                    parallel_err = vector_error(-dock_midline_xy, base_midline_xy)
                    direction_err = 2.0 * parallel_err
                    if abs(direction_err) < successful_rotate_err_ang:
                        direction_err = 0.0
                        behavior = 'back_into_dock'

            print()
            print('visual servoing errors')
            print(f'{direction_err=}')
            print(f'{distance_err=}')
            print(f'{pan_err=}')

            print()
            base_rotational_velocity = direction_err
            if abs(base_rotational_velocity) < min_base_speed:
                base_rotational_velocity = 0.0

            base_translational_velocity = -distance_err
            if abs(base_translational_velocity) < min_base_speed:
                base_translational_velocity = 0.0

            head_pan_velocity = -pan_err

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
