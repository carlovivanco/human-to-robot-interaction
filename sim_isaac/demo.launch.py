"""Isaac Sim (sim_isaac/world.py) + robot_state_publisher + the same IK node as the Gazebo demo.

    ros2 launch sim_isaac/demo.launch.py        (ISAAC_SIM=<install dir>, default ~/isaacsim; DEMO_HEADLESS=1: no window)

Needs franka_ros2 (jazzy branch) built and sourced, see README.
"""
import os
import re

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
ISAAC = os.environ.get('ISAAC_SIM', os.path.expanduser('~/isaacsim'))
ROBOT_X = -0.35  # robot base offset in the world, matches Panda.x_offset (world.py places the robot, not the URDF)
OUT = '/tmp/fr3_isaac'


def generate_launch_description():
    urdf = xacro.process_file(
        os.path.join(get_package_share_directory('franka_gazebo_bringup'), 'urdf', 'franka_arm.gazebo.xacro'),
        mappings={'robot_type': 'fr3', 'hand': 'true', 'ee_id': 'franka_hand', 'gazebo_effort': 'false',
                  'xyz': '0 0 0'},
    ).toxml()
    # robot_state_publisher keeps the package:// URDF (TF only); Isaac's importer gets absolute mesh paths
    isaac_urdf = re.sub(r'package://([^/"]+)/',
                        lambda m: get_package_share_directory(m.group(1)) + '/', urdf)
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/fr3.urdf', 'w') as f:
        f.write(isaac_urdf)

    # the robot sits at ROBOT_X in the world; the TF tree starts at fr3_link0, so the URDF stays at the origin
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': urdf}])
    ik_node = ExecuteProcess(cmd=['python3', f'{os.path.dirname(HERE)}/sim_ros2/ik_node.py'], output='screen')
    isaac = ExecuteProcess(cmd=[f'{ISAAC}/python.sh', f'{HERE}/world.py', '--urdf', f'{OUT}/fr3.urdf'], output='screen')
    return LaunchDescription([rsp, ik_node, isaac])
