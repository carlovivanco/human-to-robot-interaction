import rospy
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

panda.move_to_pose([0.2, 0, 0.7])
print("first move ready")

command.start()
detector.start()

start = input("press any key to start")

while panda.get_status() == 'idle':
    tool = command.tool
    action = command.action
    if not command.tool == None:

        object_positions = {}

        object_center = next((obj["center"] for obj in detector.predicted_classes if obj['class'] == tool), None)
        if object_center == None:
            object_center = panda.home_position[0]
        object_pos = panda.cam_to_world(object_center)  # input center of first object detected
        print(object_pos)

        (x, y) = detector.center_hand
        target_center = [x, y, detector.img.get_depth(x, y)]
        print(target_center)
        target_pos = panda.cam_to_world(target_center)  # input center of first object detected
        print(target_pos)

        if action == 'pass':
            if detector.predicted_gesture['gesture'] == 'Open_Palm':
                panda.pick_at(object_pos)
                panda.release_at(target_pos)
                object_positions[tool] = object_pos
            else:
                panda.pick_at(object_pos)
                panda.move_to_pose([0.2, 0, 0.7])
                print("Waiting for your hand to be open to receive the tool")
        elif action == 'take':
            action = command.action
            panda.pick_at(object_pos)
            target_pos_N = object_positions[tool]
            panda.release_at(target_pos_N)
        else:
            print("Command error")

        panda.return_to_home()
        panda.grasp()
        command.tool = command.action = None

        print("done")

rospy.signal_shutdown("program ending")
