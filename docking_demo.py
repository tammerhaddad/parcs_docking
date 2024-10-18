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
from stretch_body import robot_params
from stretch_body import hello_utils as hu
import pprint as pp

def compute_visual_servoing_features(center_xyz, midline_xyz, camera_info):
    if (center_xyz is None) or (midline_xyz is None):
        return None, None
    
    center_xy = dh.pixel_from_3d(center_xyz, camera_info)
    
    length = 1.0
    end_xyz = center_xyz + (length * midline_xyz)
    end_xy = dh.pixel_from_3d(end_xyz, camera_info)
    
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
    

def draw_origin(image, camera_info, origin_xyz, color):
    radius = 6
    thickness = -1
    center = np.round(dh.pixel_from_3d(origin_xyz, camera_info)).astype(np.int32)
    cv2.circle(image, center, radius, color, -1, lineType=cv2.LINE_AA)

def draw_line(image, camera_info, start_xyz, end_xyz, color):
    radius = 6
    thickness = 2
    start = np.round(dh.pixel_from_3d(start_xyz, camera_info)).astype(np.int32)
    end = np.round(dh.pixel_from_3d(end_xyz, camera_info)).astype(np.int32)
    cv2.line(image, start, end, color, thickness, lineType=cv2.LINE_AA)
    
def draw_text(image, origin, text_lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    location = origin + np.array([0, -55])
    location = location.astype(np.int32)
        
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_size, 4)
        (text_width, text_height), text_baseline = text_size
        center = int(text_width / 2)
        offset = np.array([-center, i * (1.7*text_height)]).astype(np.int32)
        cv2.putText(image, line, location + offset, font, font_size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, location + offset, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)

def get_dxl_joint_limits(joint):
    # method to get dynamixel joint limits in radians from robot params
    # Refer https://github.com/hello-robot/stretch_body/blob/master/body/stretch_body/dynamixel_hello_XL430.py#L1196:L1199

    range_t = robot_params.RobotParams().get_params()[1][joint]['range_t']
    flip_encoder_polarity = robot_params.RobotParams().get_params()[1][joint]['flip_encoder_polarity']
    gr = robot_params.RobotParams().get_params()[1][joint]['gr']
    zero_t = robot_params.RobotParams().get_params()[1][joint]['zero_t']

    polarity = -1.0 if flip_encoder_polarity else 1.0
    range_rad = []
    for t in range_t:
        x = t - zero_t
        rad_world = polarity*hu.deg_to_rad((360.0 * x / 4096.0))/gr
        range_rad.append(rad_world)
    return range_rad


def vector_error(target, current):
    err_mag = 1.0 - np.dot(target, current)
    err_sign = np.sign(np.cross(target, current))
    err = err_sign * err_mag
    return err


####################################
# Miscellaneous Parameters

motion_on = True
print_timing = False #True

# Defines a deadzone for mobile base rotation, since low values can
# lead to no motion and noises on some surfaces like carpets.
min_base_speed = 0.0 #0.05

successful_pre_docking_err_m = 0.01
successful_pre_docking_err_ang = 0.05
successful_rotate_err_ang = 0.01
successful_pan_err = 0.2

pre_docking_distance_m = 0.55 #0.63 #0.5

####################################
## Gains for Reach Behavior

overall_visual_servoing_velocity_scale = 0.02 #0.01 #1.0

joint_visual_servoing_velocity_scale = {
    'base_forward' : 0.1, #15.0
    'base_counterclockwise' : 400.0,
    'head_pan_counterclockwise' : 2.0
}

####################################
## Initial Pose

joint_state_center = {
    'head_pan_pos': -np.pi,
    'head_tilt_pos': (-np.pi/2.0) + (np.pi/10.0), 
    'lift_pos' : 0.3, #0.25, #0.2,
    'arm_pos': 0.01,
    'wrist_yaw_pos': (np.pi * (3.5/4.0)),
#    'wrist_yaw_pos': 0.0,
    'wrist_pitch_pos': 0.0, #-0.6
    'wrist_roll_pos': 0.0,
    'gripper_pos': 10.46
}

####################################
## Allowed Range of Motion

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


def recenter_robot(robot):
    robot.head.move_to('head_pan', joint_state_center['head_pan_pos'])
    robot.head.move_to('head_tilt', joint_state_center['head_tilt_pos'])
    robot.push_command()
    robot.wait_command()

    robot.end_of_arm.get_joint('wrist_yaw').move_to(joint_state_center['wrist_yaw_pos'])
    robot.end_of_arm.get_joint('wrist_pitch').move_to(joint_state_center['wrist_pitch_pos'])
    robot.push_command()
    robot.wait_command()

    robot.arm.move_to(joint_state_center['arm_pos'])
    robot.push_command()
    robot.wait_command()

    robot.lift.move_to(joint_state_center['lift_pos'])
    robot.push_command()
    robot.wait_command()

    robot.end_of_arm.get_joint('stretch_gripper').move_to(joint_state_center['gripper_pos'])
    robot.push_command()
    robot.wait_command()
        

def main(use_yolo, use_remote_computer, exposure):

    try:
        pix_per_m_av = None
        pix_per_m_n = 0
        
        camera = dc.D435i()

        time.sleep(1.0)
        
        robot = rb.Robot()
        robot.startup()
        recenter_robot(robot)

        marker_info = {}
        with open('aruco_marker_info.yaml') as f:
            marker_info = yaml.load(f, Loader=SafeLoader)

        aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=True, use_apriltag_refinement=False, brighten_images=True)

        first_frame = True

        controller = nvc.NormalizedVelocityControl(robot)
        controller.reset_base_odometry()
        
        loop_timer = lt.LoopTimer()

        behaviors = ['pre_docking', 'rotate', 'drive_backwards', 'docked']
        behavior = 'pre_docking'
        
        while behavior != 'docked':
            print('_______________________________________')
                            
            loop_timer.start_of_iteration()

            color_camera_info = camera.get_camera_info()
            camera_info = color_camera_info
            color_image = camera.get_image()

            aruco_detector.update(color_image, camera_info)

            markers = aruco_detector.get_detected_marker_dict()
            base_center_xyz = None
            base_midline_xyz = None
            dock_center_xyz = None
            dock_midline_xyz = None
            for k in markers:
                m = markers[k]
                name = m['info']['name']
                if name == 'base_left':
                    base_center_xyz = m['pos'] + ((0.13) * m['x_axis'])
                    base_midline_xyz = -m['y_axis']
                if name == 'docking_station':
                    dock_center_xyz = m['pos']
                    dock_midline_xyz = -m['y_axis']

            # compute and display image-based task-relevant features for visual servoing
            base_center_xy, base_midline_xy = compute_visual_servoing_features(base_center_xyz, base_midline_xyz, camera_info)
            display_visual_servoing_features(base_center_xy, base_midline_xy, color_image)
            
            dock_center_xy, dock_midline_xy = compute_visual_servoing_features(dock_center_xyz, dock_midline_xyz, camera_info)
            display_visual_servoing_features(dock_center_xy, dock_midline_xy, color_image)


            direction_err = 0.0
            distance_err = 0.0
            pan_err = 0.0

            joint_state = controller.get_joint_state()
            # convert base odometry angle to be in the range -pi to pi
            joint_state['base_odom_theta'] = hm.angle_diff_rad(joint_state['base_odom_theta'], 0.0)
            
            if behavior == 'drive_backwards':
                print('drive_backwards!')
                battery_charging = joint_state['battery_charging']
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

                if behavior == 'pre_docking':
                    print('pre_docking!')

                    # find the pre-docking waypoint
                    start_xy = dh.pixel_from_3d(base_center_xyz, camera_info)
                    end_xy = dh.pixel_from_3d(base_center_xyz + base_midline_xyz, camera_info)
                    pix_per_m = np.linalg.norm(end_xy - start_xy)
                    if False: 
                        if pix_per_m_av is None:
                            pix_per_m_av = pix_per_m
                            pix_per_m_n = pix_per_m_n + 1
                        else:
                            pix_per_m_av = ((pix_per_m_av * pix_per_m_n) + pix_per_m) / (pix_per_m_n + 1.0)
                            pix_per_m_n = pix_per_m_n + 1
                    else:
                        if pix_per_m_av is None: 
                            pix_per_m_av = pix_per_m

                    dist_pix = pix_per_m_av * pre_docking_distance_m
                    print(f'{pix_per_m_av=}')

                    pre_docking_center_xy = dock_center_xy + (dist_pix * dock_midline_xy)
                    pre_docking_midline_xy = dock_midline_xy

                    display_visual_servoing_features(pre_docking_center_xy, pre_docking_midline_xy, color_image)

                    if False: 
                        # find rotation to make the base midline orthogonal to the dock midline
                        proj_mag = np.dot(base_midline_xy, dock_midline_xy)
                        orth_vec = None
                        if abs(proj_mag) > 0.0: 
                            orth_vec = base_midline_xy - (proj_mag * dock_midline_xy)
                            orth_vec_mag = np.linalg.norm(orth_vec)
                            if orth_vec_mag > 0.0:
                                orth_vec = orth_vec / orth_vec_mag
                            else:
                                orth_vec = None
                        if (orth_vec is not None) and (base_midline_xy is not None):
                            orth_err = np.dot(orth_vec, base_midline_xy)
                            print(f'{orth_err=}')

                        display_visual_servoing_features(base_center_xy, orth_vec, color_image)

                    # find error to the pre-docking waypoint location
                    direction = pre_docking_center_xy - base_center_xy

                    display_visual_servoing_features(base_center_xy, direction, color_image, [0,0,255], 1.0)

                    distance = np.linalg.norm(direction)
                    if distance > 0.0:
                        direction = direction / distance

                    distance_success = False
                    direction_success = False
                        
                    distance_err = distance
                    if abs(distance_err) < (pix_per_m_av * successful_pre_docking_err_m):
                        distance_err = 0.0
                        distance_success = True

                    if not distance_success: 
                        direction_err = vector_error(direction, base_midline_xy)
                        if (abs(direction_err) < successful_pre_docking_err_ang):
                            direction_err = 0.0
                            direction_success = True
                    else:
                        direction_err = 0.0
                        direction_success = True

                    if direction_success and distance_success:
                        behavior = 'rotate'

                elif behavior == 'rotate':
                    print('rotate!')
                        
                    # find rotational error to make the base midline parallel to the pre-docking direction
                    parallel_err = vector_error(-dock_midline_xy, base_midline_xy)
                    direction_err = 2.0 * parallel_err
                    if abs(direction_err) < successful_rotate_err_ang:
                        direction_err = 0.0
                        behavior = 'drive_backwards'
                    
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

            print()
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

        robot.stop()

    finally:

        robot.stop()

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(
        prog='Stretch 3 Visual Servoing Demo',
        description='This application provides a demonstration of interactive grasping using visual servoing and the gripper-mounted D405.',
    )
    parser.add_argument('-y', '--yolo', action='store_true', help = 'Receive task-relevant features for visual servoing from an external process using YOLOv8. The default is to servo to a cube with an ArUco marker with a single process using OpenCV. To use YOLOv8, you will need to use this option. You will also need to run send_d405_images.py and recv_and_yolo_d405_images.py. Prior to using this option, configure the network with the file yolo_networking.py.')

    parser.add_argument('-r', '--remote', action='store_true', help = 'Use this argument when allowing a remote computer to send task-relevant information for visual servoing, such as 3D positions for the fingertips and target objects. Prior to using this option, configure the network with the file yolo_networking.py.')

    parser.add_argument('-e', '--exposure', action='store', type=str, default='low', help=f'Set the D405 exposure to {dh.exposure_keywords} or an integer in the range {dh.exposure_range}') 
            
    
    args = parser.parse_args()
    use_yolo = args.yolo
    use_remote_computer = args.remote

    exposure = args.exposure

    if not dh.exposure_argument_is_valid(exposure):
        raise argparse.ArgumentTypeError(f'The provided exposure setting, {exposure}, is not a valide keyword, {dh.exposure_keywords}, or is outside of the allowed numeric range, {dh.exposure_range}.')    
    
    main(use_yolo, use_remote_computer, exposure)
