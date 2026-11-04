"""Isaac Sim replacement for Gazebo + ros2_control: same scene as sim_ros2/world.sdf, same ROS topics.

Run through sim_isaac/demo.launch.py (needs the URDF it writes), or by hand:
  ~/isaacsim/python.sh sim_isaac/world.py --urdf /tmp/fr3_isaac/fr3.urdf [--headless]

Topics in : /arm_position_controller/commands, /gripper_position_controller/commands (Float64MultiArray)
Topics out: /joint_states, /camera/color/image_raw, /camera/depth/image_raw
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--headless", action="store_true", default=bool(os.environ.get("DEMO_HEADLESS")))
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})  # path tracing is too heavy for 6 GB

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils

app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("omni.scene.optimizer.core")
app_utils.enable_extension("isaacsim.robot.schema")
simulation_app.update()

import omni.graph.core as og
import omni.usd
import rclpy
import usdrt.Sdf
from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_X = -0.35  # matches Panda.x_offset
ARM = [f"fr3_joint{i}" for i in range(1, 8)]
FINGERS = ["fr3_finger_joint1", "fr3_finger_joint2"]

# tools from sim_ros2/world.sdf: name -> (x, y, yaw, [(box centre xyz, box size xyz, rgb)]); z = table top 0.2
TOOLS = {
    "hammer": (0.0, -0.3, 0.0, [((0, 0, 0.015), (0.2, 0.03, 0.03), (0.55, 0.35, 0.15)),
                                ((0.12, 0, 0.015), (0.04, 0.1, 0.03), (0.25, 0.25, 0.25))]),
    "screwdriver": (0.0, -0.1, 0.0, [((-0.04, 0, 0.015), (0.12, 0.03, 0.03), (0.8, 0.1, 0.1)),
                                     ((0.07, 0, 0.015), (0.1, 0.008, 0.008), (0.75, 0.75, 0.8))]),
    "wrench": (0.0, 0.1, 0.0, [((0, 0, 0.0075), (0.18, 0.03, 0.015), (0.9, 0.75, 0.1)),
                               ((-0.115, 0, 0.0075), (0.05, 0.05, 0.015), (0.9, 0.75, 0.1)),
                               ((0.115, 0, 0.0075), (0.05, 0.05, 0.015), (0.9, 0.75, 0.1))]),
    "pliers": (0.0, 0.3, -1.5708, [((0, 0, 0.01), (0.1, 0.03, 0.02), (0.1, 0.25, 0.8)),
                                   ((0.08, 0, 0.01), (0.06, 0.02, 0.02), (0.15, 0.15, 0.15))]),
}


def box(stage, path, centre, size, rgb, collide=True, material=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*centre))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    c.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if material:
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def build_scene(stage):
    UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(500)
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityMagnitudeAttr(9.81)
    mat = UsdShade.Material.Define(stage, "/World/Looks/grip")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(1.5), pm.CreateDynamicFrictionAttr(1.5)

    box(stage, "/World/ground", (0, 0, -0.05), (10, 10, 0.1), (0.7, 0.7, 0.7))
    box(stage, "/World/table", (0.1, 0, 0.1), (0.6, 0.8, 0.2), (0.45, 0.3, 0.2))
    for name, (x, y, yaw, boxes) in TOOLS.items():
        root = UsdGeom.Xform.Define(stage, f"/World/{name}")
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.2))
        xf.AddRotateZOp().Set(float(np.degrees(yaw)))
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.1)
        for i, (centre, size, rgb) in enumerate(boxes):
            box(stage, f"/World/{name}/b{i}", centre, size, rgb, material=mat)

    # the hand that receives the tool: horizontal, palm up (see scripts/make_hand_relief.py)
    stage_utils.add_reference_to_stage(f"{os.path.dirname(HERE)}/sim_ros2/assets/hand/hand.glb", "/World/hand")
    h = UsdGeom.Xformable(stage.GetPrimAtPath("/World/hand"))
    h.ClearXformOpOrder()
    h.AddTranslateOp().Set(Gf.Vec3d(0.06, 0, 0.30))
    h.AddRotateYOp().Set(float(np.degrees(-1.2208)))
    # collision plate under the palm (invisible), same box as sim_ros2/assets/hand/model.sdf: tools rest on it
    plate = box(stage, "/World/hand/palm", (0.011, 0, 0.02), (0.03, 0.12, 0.20), (0, 0, 0), material=UsdShade.Material(stage.GetPrimAtPath("/World/Looks/grip")))
    plate.GetPrim().GetAttribute("visibility").Set("invisible")


def add_robot(stage):
    cfg = URDFImporterConfig(
        urdf_path=args.urdf, usd_path=os.path.join(os.path.dirname(args.urdf), "usd"), fix_base=True,
        joint_target_type="position", joint_drive_type="force",
        override_joint_stiffness={"fr3_joint.*": 4000.0, "fr3_finger.*": 2000.0},
        override_joint_damping={"fr3_joint.*": 400.0, "fr3_finger.*": 40.0})
    usd = URDFImporter(cfg).import_urdf()
    stage_utils.add_reference_to_stage(usd, "/World/fr3")
    r = UsdGeom.Xformable(stage.GetPrimAtPath("/World/fr3"))
    r.ClearXformOpOrder()
    r.AddTranslateOp().Set(Gf.Vec3d(ROBOT_X, 0, 0))


def add_camera(stage):
    # Gazebo camera: looks along +X, rpy (0, 0.7, pi) at (0.5, 0, 0.85). USD camera looks along -Z, up +Y.
    cy, sy, cp, sp = np.cos(np.pi), np.sin(np.pi), np.cos(0.7), np.sin(0.7)
    R_gz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R = R_gz @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])  # columns: usd right, up, back in the gz frame
    cam = UsdGeom.Camera.Define(stage, "/World/camera")
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Matrix3d(*R.T.flatten().tolist()))  # Gf is row-vector convention
    m.SetTranslateOnly(Gf.Vec3d(0.5, 0, 0.85))
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    aperture = 20.955
    cam.GetHorizontalApertureAttr().Set(aperture)
    cam.GetVerticalApertureAttr().Set(aperture * 480 / 640)
    cam.GetFocalLengthAttr().Set(aperture / 2 / np.tan(1.0472 / 2))  # fx = 554.256 px at 640x480
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10))

    keys = og.Controller.Keys
    g, *_ = og.Controller.edit(
        {"graph_path": "/ROS_Camera", "evaluator_name": "push",
         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
        {keys.CREATE_NODES: [("OnTick", "omni.graph.action.OnTick"),
                             ("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                             ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                             ("depth", "isaacsim.ros2.bridge.ROS2CameraHelper")],
         keys.CONNECT: [("OnTick.outputs:tick", "rp.inputs:execIn"),
                        ("rp.outputs:execOut", "rgb.inputs:execIn"), ("rp.outputs:execOut", "depth.inputs:execIn"),
                        ("rp.outputs:renderProductPath", "rgb.inputs:renderProductPath"),
                        ("rp.outputs:renderProductPath", "depth.inputs:renderProductPath")],
         keys.SET_VALUES: [("rp.inputs:cameraPrim", [usdrt.Sdf.Path("/World/camera")]),
                           ("rp.inputs:width", 640), ("rp.inputs:height", 480),
                           ("rgb.inputs:topicName", "/camera/color/image_raw"), ("rgb.inputs:type", "rgb"),
                           ("rgb.inputs:frameId", "camera"),
                           ("depth.inputs:topicName", "/camera/depth/image_raw"), ("depth.inputs:type", "depth"),
                           ("depth.inputs:frameId", "camera")]})
    og.Controller.evaluate_sync(g)


class Bridge(Node):
    def __init__(self, robot):
        super().__init__("isaac_bridge")
        self.robot = robot
        # dataset generation (sim_ros2/make_dataset.py --sim isaac): teleport the tools, read their poses back
        self.tools = {n: RigidPrim(f"/World/{n}", reset_xform_op_properties=True) for n in TOOLS}
        self.create_subscription(Float64MultiArray, "/dataset/set_poses", self.on_set_poses, 10)
        self.poses_pub = self.create_publisher(Float64MultiArray, "/dataset/tool_poses", 10)
        self.names = list(robot.dof_names)
        self.target = None  # filled with the current positions on the first tick
        self.create_subscription(Float64MultiArray, "/arm_position_controller/commands", self.on_arm, 10)
        self.create_subscription(Float64MultiArray, "/gripper_position_controller/commands", self.on_gripper, 10)
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

    def _set(self, joints, values):
        if self.target is not None:
            for j, v in zip(joints, values):
                self.target[self.names.index(j)] = v

    def on_arm(self, msg):
        self._set(ARM, msg.data)

    def on_gripper(self, msg):
        self._set(FINGERS, [msg.data[0]] * 2)  # the URDF <mimic> is not honoured: drive both fingers

    def on_set_poses(self, msg):
        """data = [x, y, z, yaw] per tool in TOOLS order (world frame, z = box-model origin)"""
        for n, (x, y, z, yaw) in zip(self.tools, np.reshape(msg.data, (-1, 4))):
            self.tools[n].set_world_poses(positions=[[x, y, z]], orientations=[[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]])
            self.tools[n].set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])

    def tick(self):
        q = np.asarray(self.robot.get_dof_positions().numpy()).reshape(-1)
        if self.target is None:
            self.target = q.copy()
        self.robot.set_dof_position_targets(self.target.reshape(1, -1))
        js = JointState(name=self.names, position=q.tolist())
        js.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(js)
        poses = []  # [x, y, z, qw, qx, qy, qz] per tool in TOOLS order
        for t in self.tools.values():
            p, o = t.get_world_poses()
            poses += np.asarray(p.numpy()).reshape(-1).tolist() + np.asarray(o.numpy()).reshape(-1).tolist()
        self.poses_pub.publish(Float64MultiArray(data=poses))


stage_utils.set_stage_units(meters_per_unit=1.0)
stage = omni.usd.get_context().get_stage()
build_scene(stage)
add_robot(stage)
add_camera(stage)
SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
simulation_app.update()

rclpy.init()
app_utils.play()
simulation_app.update()
bridge = Bridge(Articulation("/World/fr3"))
while simulation_app.is_running():
    simulation_app.update()
    rclpy.spin_once(bridge, timeout_sec=0.0)
    bridge.tick()
bridge.destroy_node()
simulation_app.close()
