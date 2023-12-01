import os
from datetime import datetime
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import random
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch
from numpy.lib.npyio import save
from PIL import Image
from typing import *
from pprint import pprint

from .camera import Camera
from .robot import setup_sisbot


CAM_X, CAM_Y, CAM_Z = 0.05, -0.52, 1.9
DEPTH_RADIUS = 1
IMG_SIZE = 224

TARGET_LOCATIONS = {
  'top left corner':     (0.35,        -0.7,       1.0),
  'top side':            (0,           -0.7,       1.0),
  'top right corner':    (-0.35,        -0.7,       1.0),
  'left side':           (0.35,        -0.45,      1.0),
  'middle':              (0,           -0.45,      1.0),
  'right side':          (-0.35,       -0.45,      1.0),
  'bottom left corner':  (0.35,       -0.15,      1.0),
  'bottom side':         (0,           -0.15,      1.0),
  'bottom right corner': (-0.35,       -0.15,      1.0),
}

class FailToReachTargetError(RuntimeError):
    pass


class Environment:
    OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.25
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.4
    TARGET_ZONE_POS = [0.7, 0.0, 0.685]
    SIMULATION_STEP_DELAY = 0.0005 #speed of simulator - the lower the fatser ## should be a param
    FINGER_LENGTH = 0.06
    Z_TABLE_TOP = 0.785
    GRIP_REDUCTION = 0.60
    DIST_BACKGROUND = 1.115
    IMG_SIZE = IMG_SIZE 
    DEPTH_RADIUS = DEPTH_RADIUS
    TARGET_LOCATIONS = TARGET_LOCATIONS

    def __init__(self, camera: Camera, asset_root="./assets", vis=False, debug=False, gripper_type='140', finger_length=0.06) -> None:
        self.vis = vis
        self.debug = debug
        self.camera = camera

        self.assets_root = asset_root

        self.obj_init_pos = (camera.x, camera.y)
        self.obj_ids = []
        self.obj_names = []
        self.obj_positions = []
        self.obj_orientations = []
        self.obj_grasps = [] # a list of possible grasps per object

        if gripper_type not in ('85', '140'):
            raise NotImplementedError(
                'Gripper %s not implemented.' % gripper_type)
        self.gripper_type = gripper_type
        self.finger_length = finger_length

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF(os.path.join('plane.urdf'))
        self.tableID = p.loadURDF(os.path.join(self.assets_root, 'urdf/objects/table.urdf'),
                                  [0.0, -0.65, 0.76],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)
        self.target_table_id = p.loadURDF(os.path.join(self.assets_root, 'urdf/objects/target_table.urdf'),
                                          [0.7, 0.0, 0.66],
                                          p.getQuaternionFromEuler([0, 0, 0]),
                                          useFixedBase=True)
        self.target_id = p.loadURDF(os.path.join(self.assets_root, 'urdf/objects/traybox.urdf'),
                                    self.TARGET_ZONE_POS,
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    globalScaling=0.7)
        self.UR5Stand_id = p.loadURDF(os.path.join(self.assets_root, 'urdf/objects/ur5_stand.urdf'),
                                      [-0.7, -0.36, 0.0],
                                      p.getQuaternionFromEuler([0, 0, 0]),
                                      useFixedBase=True)
        self.robot_id = p.loadURDF(os.path.join(self.assets_root, 'urdf/ur5_robotiq_%s.urdf' % gripper_type),
                                   [0, 0, 0.0],  # StartPosition
                                   p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
                                   useFixedBase=True,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
            setup_sisbot(p, self.robot_id, gripper_type)
        self.eef_id = 7  # ee_link

        # Add force sensors
        p.enableJointForceTorqueSensor(
            self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(
            self.robot_id, self.joints['right_inner_finger_pad_joint'].id)

        # Change the friction of the gripper
        p.changeDynamics(
            self.robot_id, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=1)
        p.changeDynamics(
            self.robot_id, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=1)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # Task space (Cartesian space)
        if debug:
            self.xin = p.addUserDebugParameter('x', -0.4, 0.4, 0.11)
            self.yin = p.addUserDebugParameter('y', -0.8, 0, -0.49)
            self.zin = p.addUserDebugParameter('z', 0.9, 1.3, 1.1)
            self.rollId = p.addUserDebugParameter(
                'roll', -3.14, 3.14, 0)  # -1.57 yaw
            self.pitchId = p.addUserDebugParameter(
                'pitch', -3.14, 3.14, np.pi/2)
            self.yawId = p.addUserDebugParameter(
                'yaw', -np.pi/2, np.pi/2, 0)  # -3.14 pitch
            self.gripper_opening_length_control = p.addUserDebugParameter(
                'gripper_opening_length', 0, 0.1, 0.085)

        # Add debug lines for end effector and camera
        if vis:
            self.eef_debug_lineID = None
            # p.addUserDebugLine([camera.x, camera.y, camera.z - 0.15], [camera.x, camera.y, camera.z - 0.4], [0, 1, 0], lineWidth=5)
            dist = 1.5
            yaw = 30
            pitch = -50
            target = [0.1, -0.30, 0.95]
            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

            # camera model
            alpha = 0.025 #m 
            z_cam_l1 = camera.z #m 

            color = [0.3, 0, 0]
            p.addUserDebugText("RGB-D camera", [camera.x-0.1, camera.y, camera.z+0.02], color, textSize=2)
             
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y-alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y-alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y+alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y+alpha, z_cam_l1], color, lineWidth=4)
        
            z_cam_l2 = camera.z - 0.07 #m 
               
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l2], [camera.x+alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l2], [camera.x-alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l2], [camera.x-alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l2], [camera.x+alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)

            ### body
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l1], [camera.x+alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l1], [camera.x-alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)

            ### Third rectangle             
            z_cam_l3 = camera.z - 0.125 #m 
            beta = alpha * 2 
            p.addUserDebugLine([camera.x+beta, camera.y+beta, z_cam_l3], [camera.x+beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x+beta, camera.y-beta, z_cam_l3], [camera.x-beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-beta, camera.y-beta, z_cam_l3], [camera.x-beta, camera.y+beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-beta, camera.y+beta, z_cam_l3], [camera.x+beta, camera.y+beta, z_cam_l3], color, lineWidth=4)

            ### body
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l2], [camera.x+beta, camera.y+beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l2], [camera.x+beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l2], [camera.x-beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l2], [camera.x-beta, camera.y+beta, z_cam_l3], color, lineWidth=4)

            ### working area             
            working_area = 0.79 #m 
            beta = 0.4 
            p.addUserDebugLine([camera.x+beta, camera.y+beta, working_area], [camera.x+beta, camera.y-beta, working_area], [0, 1, 0], lineWidth=5)
            p.addUserDebugLine([camera.x+beta, camera.y-beta, working_area], [camera.x-beta, camera.y-beta, working_area], [0, 1, 0], lineWidth=5)
            p.addUserDebugLine([camera.x-beta, camera.y-beta, working_area], [camera.x-beta, camera.y+beta, working_area], [0, 1, 0], lineWidth=5)
            p.addUserDebugLine([camera.x-beta, camera.y+beta, working_area], [camera.x+beta, camera.y+beta, working_area], [0, 1, 0], lineWidth=5)

        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.1)
        self.ee_position_limit = ((-0.8, 0.8),
                                  (-0.8, 0.8),
                                  (0.785, 1.4))
        self.reset_robot()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            if self.debug:
                if self.eef_debug_lineID is not None:
                    p.removeUserDebugItem(self.eef_debug_lineID)
                eef_xyz = p.getLinkState(self.robot_id, self.eef_id)[0:1]
                end = np.array(eef_xyz[0])
                end[2] -= 0.5
                self.eef_debug_lineID = p.addUserDebugLine(np.array(eef_xyz[0]), end, [1, 0, 0])
            time.sleep(self.SIMULATION_STEP_DELAY)

    @staticmethod
    def is_still(handle):
        still_eps = 1e-3
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        # print(np.abs(lin_vel).sum() + np.abs(ang_vel).sum())
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def wait_until_still(self, objID, max_wait_epochs=100):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if self.is_still(objID):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def wait_until_all_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if np.all(list(self.is_still(obj_id) for obj_id in self.obj_ids)):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(
            self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def reset_robot(self):
        user_parameters = (0, -1.5446774605904932, 1.54, -1.54,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        for _ in range(60):
            for i, name in enumerate(self.controlJoints):
                
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                self.step_simulation()
                
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=0.085)
            self.step_simulation()

    def move_arm_away(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                    targetPosition=0., force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

                                    
            self.step_simulation()

    def check_grasped(self, obj_id):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +
                          contact_right if item[2] in [obj_id])
        if len(contact_ids) == 1:
            return True
        return False

    def check_grasped_id(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +
                          contact_right if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            if self.debug:
                print('Warning: Multiple items in hand!')
        return list(item_id for item_id in contact_ids if item_id in self.obj_ids)

    def check_contact(self, id_a, id_b):
        contact_a = p.getContactPoints(bodyA=id_a)
        contact_ids = set(item[2] for item in contact_a if item[2] in [id_b])
        if len(contact_ids) == 1:
            return True
        return False

    def check_target_reached(self, obj_id):
        aabb = p.getAABB(self.target_id, -1)
        x_min, x_max = aabb[0][0], aabb[1][0]
        y_min, y_max = aabb[0][1], aabb[1][1]
        pos = p.getBasePositionAndOrientation(obj_id)
        x, y = pos[0][0], pos[0][1]
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        return False

    def gripper_contact(self, bool_operator='and', force=250):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints( bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints( bodyA=self.robot_id, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robot_id, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robot_id, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip( gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
            
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_opening_angle)

            self.step_simulation()

    def auto_close_gripper(self, step: int = 120, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robot_id, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self.gripper_contact():
                return True
        return False

    def calc_z_offset(self, gripper_opening_length: float):
        gripper_opening_length = np.clip(
            gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - \
            math.asin((gripper_opening_length - 0.010) / 0.1143)
        if self.gripper_type == '140':
            gripper_length = 10.3613 * \
                np.sin(1.64534-0.24074 * (gripper_opening_angle / np.pi)) - 10.1219
        else:
            gripper_length = 1.231 - 1.1
        return gripper_length

    def remove_obj(self, obj_id):
        # Get index of obj in id list, then remove object from simulation
        idx = self.obj_ids.index(obj_id)
        self.obj_orientations.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_ids.pop(idx)
        self.obj_names.pop(idx)
        self.obj_grasps.pop(idx)
        p.removeBody(obj_id)

    def remove_all_obj(self):
        self.obj_positions.clear()
        self.obj_orientations.clear()
        self.obj_names.clear()
        self.obj_grasps.clear()
        for obj_id in self.obj_ids:
            p.removeBody(obj_id)
        self.obj_ids.clear()

    def reset_all_obj(self):
        for i, obj_id in enumerate(self.obj_ids):
            p.resetBasePositionAndOrientation(
                obj_id, self.obj_positions[i], self.obj_orientations[i])
            self.obj_grasps[i] = None
        self.wait_until_all_still()

    def update_obj_states(self):
        for i, obj_id in enumerate(self.obj_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.obj_positions[i] = pos
            self.obj_orientations[i] = orn
            self.obj_grasps[i] = None
    
    def get_obj_states(self):
        return [{'id': i, 'name': n, 'pos': p, 'orn': o} for i,n,p,o in zip(
                                                             self.obj_ids,
                                                             self.obj_names, 
                                                             self.obj_positions,
                                                             self.obj_orientations
                                                            )]
    
    def load_obj(self, path, name, pos, yaw, mod_orn=False, mod_stiffness=False):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        obj_id = p.loadURDF(path, pos, orn)
        # adjust position according to height
        aabb = p.getAABB(obj_id, -1)
        if mod_orn:
            minm, maxm = aabb[0][1], aabb[1][1]
            orn = p.getQuaternionFromEuler([0, np.pi*0.5, yaw])
        else:
            minm, maxm = aabb[0][2], aabb[1][2]

        pos[2] += (maxm - minm) / 2
        p.resetBasePositionAndOrientation(obj_id, pos, orn)
        # change dynamics
        if mod_stiffness:
            p.changeDynamics(obj_id,
                             -1, lateralFriction=1,
                             rollingFriction=0.001,
                             spinningFriction=0.002,
                             restitution=0.01,
                             contactStiffness=100000,
                             contactDamping=0.0)
        else:
            p.changeDynamics(obj_id,
                             -1, lateralFriction=1,
                             rollingFriction=0.002,
                             spinningFriction=0.001,
                             restitution=0.01)
        self.obj_ids.append(obj_id)
        self.obj_positions.append(pos)
        self.obj_orientations.append(orn)
        self.obj_names.append(name)
        self.obj_grasps.append(None)
        return obj_id, name, pos, orn

    def load_isolated_obj(self, path, name, mod_orn=False, mod_stiffness=False):
        r_x = random.uniform(
            self.obj_init_pos[0] - 0.3, self.obj_init_pos[0] + 0.3)
        r_y = random.uniform(
            self.obj_init_pos[1] - 0.3, self.obj_init_pos[1] + 0.3)
        yaw = random.uniform(0, np.pi)

        pos = [r_x, r_y, self.Z_TABLE_TOP]
        obj_id, _, _, _ = self.load_obj(path, name, pos, yaw, mod_orn, mod_stiffness)
        for _ in range(100):
            self.step_simulation()
            
        self.wait_until_still(obj_id)
        self.update_obj_states()

    def set_obj_grasps(self, obj_id, grasps):
        """
        Set list of possible grasps to object based on oracle / network
        """
        idx = self.obj_ids.index(obj_id)
        self.obj_grasps[idx] = grasps

    def get_obj_grasps(self, obj_id):
        idx = self.obj_ids.index(obj_id)
        return self.obj_grasps[idx]

    def create_temp_box(self, width, num):
        box_width = width
        box_height = 0.2
        box_z = self.Z_TABLE_TOP + (box_height/2)
        id1 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0] - box_width /
                             2, self.obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0] + box_width /
                             2, self.obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0], self.obj_init_pos[1] +
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0], self.obj_init_pos[1] -
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        return [id1, id2, id3, id4]

    def move_obj_along_axis(self, obj_id, axis, operator, step, stop):
        collison = False
        while not collison:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = list(pos)
            if operator == '+':
                new_pos[axis] += step
                if new_pos[axis] > stop:
                    break
            else:
                new_pos[axis] -= step
                if new_pos[axis] < stop:
                    break
            # Move object towards center
            p.resetBasePositionAndOrientation(obj_id, new_pos, orn)
            p.stepSimulation()
            contact_a = p.getContactPoints(obj_id)
            # If object collides with any other object, stop
            contact_ids = set(item[2]
                              for item in contact_a if item[2] in self.obj_ids)
            if len(contact_ids) != 0:
                collison = True
        # Move one step back
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        new_pos = list(pos)
        if operator == '+':
            new_pos[axis] -= step
        else:
            new_pos[axis] += step
        p.resetBasePositionAndOrientation(obj_id, new_pos, orn)


    def move_ee(self, action, max_step=300, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
              0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix

        real_xyz, real_xyzw = p.getLinkState(self.robot_id, self.eef_id)[0:2]
        alpha = 0.2 # this parameter can be tuned to make the movement  smoother
        
        for _ in range(max_step):

            # apply IK
            x_tmp = alpha * x + (1-alpha)*real_xyz[0]
            y_tmp = alpha * y + (1-alpha)*real_xyz[1]
            z_tmp = alpha * z + (1-alpha)*real_xyz[2]
            
            joint_poses = p.calculateInverseKinematics(bodyUniqueId=self.robot_id, endEffectorLinkIndex=self.eef_id, 
                                                       targetPosition=[x_tmp, y_tmp, z_tmp], targetOrientation=orn, 
                                                       maxNumIterations=200)

            # Filter out the gripper
            for i, name in enumerate(self.controlJoints[:-1]):
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            self.step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)

            # Check if contact with objects
            if check_collision_config and self.gripper_contact(**check_collision_config):
                if self.debug:
                    print('Collision detected!', self.check_grasped_id())
                return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]

            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(
                self.robot_id, self.eef_id)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        if self.debug:
            print('Failed to reach the target')
        return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]

    def pick_obj_from_center(self, obj_id: int):
        """
        Dummy method to grasp object from its center
        """
        pos = self.obj_positions[self.obj_ids.index(obj_id)]
        orn = self.obj_orientations[self.obj_ids.index(obj_id)]
        yaw = np.pi - p.getEulerFromQuaternion(orn)[-1]
        gripper_opening_length = 0.04
        obj_height = self.DIST_BACKGROUND -  pos[-1]
        return self.grasp(pos, yaw, gripper_opening_length, obj_height)

    def pick_obj(self, obj_id: int, vis: bool = False):
        """
        Pick object by id (from pre-set grasp pose)
        """
        succes_grasp, grasped_obj_id = False, None
        idx = self.obj_ids.index(obj_id)
        grasps = self.obj_grasps[idx]
        assert grasps is not None, "first have to set grasps with env.set_obj_grasps()"
        LID = []
        for g in grasps:
            x, y, z, yaw, opening_len, obj_height = g
            
            if vis:
                if len(LID) > 0:
                    self.remove_drawing(LID)
                    LID = []
                LID = self.draw_predicted_grasp(g, color=[0,1,0], lineIDs=LID)
                self.dummy_simulation_steps(10)
                
            succes_grasp, grasped_obj_id = self.grasp((x, y, z), yaw, opening_len, obj_height)
            if succes_grasp:
                assert grasped_obj_id == obj_id, f'Grasped wrong object {self.obj_names[self.obj_ids.index(grasped_obj_id)]}'
                if vis:
                    self.remove_drawing(LID)        
                return True, obj_id, g
            
            print('Grasping failed. Retrying...')
        if vis:
            self.remove_drawing(LID)
            self.dummy_simulation_steps(10)
            
        return succes_grasp, grasped_obj_id, None

    def place_in_loc(self, target_loc: str, vis: bool = False):
        """
        Place object in target location
        """
        try:
            grasped_obj_id = self.check_grasped_id()[0]
        except:
            print('No object currently in gripper. Exitting.')
            return False
            
        # Move object to target zone
        assert target_loc in self.TARGET_LOCATIONS.keys(), f'unknown location {target_loc}'
        x, y, z = self.TARGET_LOCATIONS[target_loc]
        tgt_orn = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])

        self.move_ee([x, y, z, tgt_orn])
        self.move_gripper(0.085)
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, tgt_orn])
        
        # Wait then check if object is in target zone
        for _ in range(20):
            self.step_simulation()

        succes_target = False
        if self.check_target_loc_reached(grasped_obj_id, [x, y, z]):
            succes_target = True
        
        return succes_target
        
    def put_obj_in_loc(self, obj_id: int, target_loc: str, vis: bool = False):
        """
        Pick object by id and place it in target location
        """
        succes_grasp, succes_target = False, False 
    
        # pick object by id
        succes_grasp, grasped_obj_id, grasp = self.pick_obj(obj_id, vis=vis)

        # Move object to target zone
        succes_target = self.place_in_loc(target_loc, vis=vis)
        
        return succes_grasp, succes_target

    def check_target_loc_reached(self, obj_id, pos):
        if obj_id is None:
            print('Asking to check unknown object')
            return False
        (x, y, z), orn = p.getBasePositionAndOrientation(obj_id)
        if abs(pos[0] - x) < 0.1 and abs(pos[1] - y) < 0.1:
            return True 
        else:
            return False 

    def grasp(self, 
              pos: tuple,
              roll: float,
              gripper_opening_length: float,
              obj_height: float,
              debug: bool = False,
              vis: bool = False
    ):
        """
        Method to perform grasp based on pose.
        pos [x y z]: The axis in real-world coordinate
        roll: float,   for grasp, it should be in [-pi/2, pi/2)
        """
        succes_grasp = False 

        x, y, z = pos
        # Substracht gripper finger length from z
        z -= self.finger_length
        z = np.clip(z, *self.ee_position_limit[2])

        LID = []
        if vis:
            if len(LID) > 0:
                self.remove_drawing(LID)
                LID = []
            g = pos[0], pos[1], pos[2], roll, gripper_opening_length, obj_height
            LID = self.draw_predicted_grasp(g, color=[0,1,0], lineIDs=LID)
            self.dummy_simulation_steps(10)
        
        # Move above target
        #self.reset_robot()
        self.move_gripper(0.1)
        orn = p.getQuaternionFromEuler([roll, np.pi/2, 0.0])
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        gripper_opening_length *= self.GRIP_REDUCTION

        # Grasp and lift object
        z_offset = self.calc_z_offset(gripper_opening_length)
        self.move_ee([x, y, z + z_offset, orn])
        # self.move_gripper(gripper_opening_length)
        self.auto_close_gripper(check_contact=True)
        for _ in range(40):
            self.step_simulation()
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # check if the object has been grasped and lifted off the table
        grasped_id = self.check_grasped_id()
        grasped_obj_id = None
        if len(grasped_id) == 1:
            grasped_obj_id = grasped_id[0]
            succes_grasp = True
        
        if vis:
            self.remove_drawing(LID)
            self.dummy_simulation_steps(10)

        return succes_grasp, grasped_obj_id

    def clean(self, 
              pos: tuple, 
              roll: float, 
              gripper_opening_length: float, 
              obj_height: float, 
              debug: bool = False,
              vis: bool = False
    ):
        """
        Method to pick object and place it in tray location
        pos [x y z]: The axis in real-world coordinate
        roll: float,   for grasp, it should be in [-pi/2, pi/2)
        """
        succes_grasp, succes_target = False, False
        grasped_obj_id = None
        
        LID = []
        if vis:
            if len(LID) > 0:
                self.remove_drawing(LID)
                LID = []
            g = pos[0], pos[1], pos[2], roll, gripper_opening_length, obj_height
            LID = self.draw_predicted_grasp(g, color=[0,1,0], lineIDs=LID)
            self.dummy_simulation_steps(10)
            
        x, y, z = pos
        # Substracht gripper finger length from z
        z -= self.finger_length
        z = np.clip(z, *self.ee_position_limit[2])
            
        # Move above target
        #self.reset_robot()
        self.move_gripper(0.1)
        orn = p.getQuaternionFromEuler([roll, np.pi/2, 0.0])
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        gripper_opening_length *= self.GRIP_REDUCTION
            
        # Grasp and lift object
        z_offset = self.calc_z_offset(gripper_opening_length)
        self.move_ee([x, y, z + z_offset, orn])
        # self.move_gripper(gripper_opening_length)
        self.auto_close_gripper(check_contact=True)
        for _ in range(40):
            self.step_simulation()
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # check if the object has been grasped and lifted off the table
        grasped_id = self.check_grasped_id()
        if len(grasped_id) == 1:
            succes_grasp = True
            grasped_obj_id = grasped_id[0]
        else:
            return succes_target, succes_grasp

        if vis:
            self.remove_drawing(LID)
            self.dummy_simulation_steps(10)
            
        # Move object to target zone
        y_drop = self.TARGET_ZONE_POS[2] + z_offset + obj_height + 0.15
        y_orn = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])

        #self.move_arm_away()
        self.move_ee([self.TARGET_ZONE_POS[0],
                     self.TARGET_ZONE_POS[1], 1.25, y_orn])
        self.move_ee([self.TARGET_ZONE_POS[0],
                     self.TARGET_ZONE_POS[1], y_drop, y_orn])
        self.move_gripper(0.085)
        self.move_ee([self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1],
                     self.GRIPPER_MOVING_HEIGHT, y_orn])

        # Wait then check if object is in target zone
        for _ in range(20):
            self.step_simulation()

        if self.check_target_reached(grasped_obj_id):
            succes_target = True
            #self.remove_obj(grasped_obj_id)

        return succes_grasp, succes_target

    def clean_obj(self, obj_id: int, debug: bool = False, vis: bool = False):
        """
        Method to pick object by id and place it in tray location
        """
        succes_grasp, succes_target = False, False 
    
        # pick object by id
        succes_grasp, grasped_obj_id, grasp = self.pick_obj(obj_id, vis=vis)

        if not succes_grasp:
            print('Grasping failed. Exitting')
            return False, False
            
        # Move object to target zone
        x, y, z, yaw, opening_len, obj_height = grasp
        z_offset = self.calc_z_offset(opening_len)
        y_drop = self.TARGET_ZONE_POS[2] + z_offset + obj_height + 0.15
        y_orn = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])

        #self.move_arm_away()
        self.move_ee([self.TARGET_ZONE_POS[0],
                     self.TARGET_ZONE_POS[1], 1.25, y_orn])
        self.move_ee([self.TARGET_ZONE_POS[0],
                     self.TARGET_ZONE_POS[1], y_drop, y_orn])
        self.move_gripper(0.085)
        self.move_ee([self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1],
                     self.GRIPPER_MOVING_HEIGHT, y_orn])

        # Wait then check if object is in target zone
        for _ in range(20):
            self.step_simulation()

        if self.check_target_reached(grasped_obj_id):
            succes_target = True
            #self.remove_obj(grasped_obj_id)

        return succes_grasp, succes_target
    
    def close(self):
        p.disconnect(self.physicsClient)

    def dummy_simulation_steps(self, n):
        for _ in range(n):
            p.stepSimulation()

    @staticmethod
    def draw_predicted_grasp(grasps, color = [0,0,1], lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs
    
    @staticmethod
    def remove_drawing(lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)

