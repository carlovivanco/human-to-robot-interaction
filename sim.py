import time

import numpy as np

from panda_class import Panda
from vision_sim.vision_server.main import VisionClass
from vision_sim.cli.combine import SpeechRecognition


panda = Panda()
print('panda')
command = SpeechRecognition()
print("commands")
detector = VisionClass(
    yolo_model_path='vision_sim/models/tools-100-epochs.pt',
    gesture_model_path='vision_sim/models/gesture_recognizer.task',
    show_camera=True
)
print('detector')

SLOT_Y = {"Hammer": -0.3, "Screwdriver": -0.1, "Wrench": 0.1, "Pliers": 0.3}  # world.sdf: tools in a row at x=0

panda.move_to_pose([0.2, 0, 0.7])
print("first move ready")

command.start()
detector.start()

start = input("press any key to start")
while not hasattr(detector, "predicted_classes"):  # the vision thread has not finished its first frame
    time.sleep(0.1)
print("Say: 'panda, pass me the hammer'")

while panda.get_status() == 'idle':
    time.sleep(0.1)
    tool = command.tool
    action = command.action
    if tool is None:
        continue
    command.tool = command.action = None

    obj = next((obj for obj in detector.predicted_classes if obj['class'] == tool), None)
    if obj is None:
        print(f"I do not see a {tool}")
        continue
    if action not in ('pass', 'take'):
        print(f"'{action}' is not supported yet")
        continue

    # centroid of the tool's depth pixels; the box centre if there are too few of them
    found = panda.tool_position(obj['box'], detector.img.depth_image)
    object_pos, yaw = found if found else (panda.cam_to_world(obj['center']), 0.0)
    print(object_pos, f"yaw {np.degrees(yaw):.0f} deg")
    panda.pick_at(object_pos, yaw=yaw)  # the gripper turns to close across the tool, however it lies

    if action == 'take':  # the tool was in the human's hand: put it back in its slot on the table
        panda.release_at([0.0, SLOT_Y[tool], 0.25])
        panda.return_to_home()
        print("done")
        continue

    print("Waiting for your hand to be open to receive the tool")
    t0 = time.time()
    while time.time() - t0 < 30:
        hand = detector.center_hand
        if hand is not None and detector.predicted_gesture['gesture'] == 'Open_Palm':
            x, y = hand
            target_pos = panda.cam_to_world([x, y, detector.img.get_depth(x, y)])
            target_pos[2] += 0.05  # the depth is the palm surface: the tool (1 cm under the TCP) stops 1 cm above it
            print(target_pos)
            panda.release_at(target_pos)
            break
        time.sleep(0.1)
    else:
        print("No open hand in 30 s, I keep the tool")

    panda.return_to_home()
    print("done")

panda.shutdown()
