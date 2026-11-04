"""Gazebo (gz sim) + franka_ros2 FR3 (joint position control + IK node) + RGB-D camera bridge.

    ros2 launch sim_ros2/demo.launch.py

Needs franka_ros2 (jazzy branch) built and sourced, see README.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_X = -0.35  # robot base offset in the world, matches Panda.x_offset
# DEMO_HEADLESS=1: no Gazebo window (tests, dataset generation)
GZ_FLAGS = '-r -s --headless-rendering' if os.environ.get('DEMO_HEADLESS') else '-r'


def generate_launch_description():
    # fingers/arm meshes are resolved through franka_description's parent dir
    os.environ['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join([
        os.path.dirname(get_package_share_directory('franka_description')),
        f'{HERE}/assets',  # model://hand
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ])

    robot_description = xacro.process_file(
        os.path.join(get_package_share_directory('franka_gazebo_bringup'), 'urdf', 'franka_arm.gazebo.xacro'),
        mappings={'robot_type': 'fr3', 'hand': 'true', 'ee_id': 'franka_hand',
                  'gazebo_effort': 'false', 'xyz': f'{ROBOT_X} 0 0'},
    ).toxml()
    # Gazebo ignores the URDF <mimic> unless the follower finger is also declared in ros2_control (state interfaces
    # only); otherwise only finger_joint1 moves and the other finger stays fixed in the middle of the gripper.
    robot_description = robot_description.replace(
        '</ros2_control>',
        '<joint name="fr3_finger_joint2"><state_interface name="position"/><state_interface name="velocity"/>'
        '<state_interface name="effort"/></joint></ros2_control>', 1)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'{GZ_FLAGS} {HERE}/world.sdf'}.items(),
    )
    robot_state_publisher = Node(package='robot_state_publisher', executable='robot_state_publisher',
                                 parameters=[{'robot_description': robot_description}])
    spawn = Node(package='ros_gz_sim', executable='create', arguments=['-topic', '/robot_description'])
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                   '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                   '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image'],
        remappings=[('/camera/image', '/camera/color/image_raw'),
                    ('/camera/depth_image', '/camera/depth/image_raw')],
    )

    def spawner(*names):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[*names, '--controller-manager-timeout', '60',
                               '--param-file', f'{HERE}/controllers.yaml'])

    ik_node = ExecuteProcess(cmd=['python3', f'{HERE}/ik_node.py'], output='screen')

    return LaunchDescription([
        gz_sim, robot_state_publisher, spawn, bridge, ik_node,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[
            spawner('joint_state_broadcaster'),
            spawner('arm_position_controller'),
            spawner('gripper_position_controller'),
        ])),
    ])
