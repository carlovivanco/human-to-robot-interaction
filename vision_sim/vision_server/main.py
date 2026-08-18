import cv2
import mediapipe as mp
from mediapipe.tasks import python
from ultralytics import YOLO
import time
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from vision_sim.vision_server.img_sub import ImageSubscriber
import threading


class VisionClass(threading.Thread):
    """
    A threading-based class that processes frames for gesture recognition and object detection in real-time.

    Attributes:
        yolo_model (YOLO): The YOLO model used for object detection.
        gesture_model_path (str): Path to the gesture model file.
        gesture_recognizer: The initialized gesture recognition object.
        show_camera (bool): Whether to display the camera output in a window.
        img (ImageSubscriber): An image subscriber that provides frames.
        processed_frame (ndarray): The latest processed frame.
    """

    def __init__(self, yolo_model_path, gesture_model_path, show_camera):
        super().__init__()
        # Load the YOLO model using the provided model path
        self.yolo_model = YOLO(yolo_model_path)
        # Path for gesture recognition model
        self.gesture_model_path = gesture_model_path
        # Initialize the gesture recognizer
        self.gesture_recognizer = self.initialize_gesture_recognizer()
        # Flag to show or hide the camera output
        self.show_camera = show_camera
        self.img = ImageSubscriber()
        self.processed_frame = self.img

    def initialize_gesture_recognizer(self):
        """
        Initialize and configure the gesture recognizer using the provided model path.
        Returns:
            gesture_recognizer: A mediapipe gesture recognition object.
        """
        base_options = python.BaseOptions(
            model_asset_path=self.gesture_model_path
        )
        options = python.vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=python.vision.RunningMode.VIDEO,
        )
        return python.vision.GestureRecognizer.create_from_options(options)

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """
        Draw lines and circles representing hand landmarks on the given frame.
        """
        LANDMARK_COLOR = (0, 255, 0)        # Green
        CONNECTION_COLOR = (255, 255, 255)  # White
        THICKNESS = 2
        RADIUS = 4

        height, width, _ = frame.shape

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12), (5, 9),
            (13, 14), (14, 15), (15, 16), (9, 13),
            (17, 18), (18, 19), (19, 20), (13, 17),
            (0, 17), (5, 0)
        ]

        for connection in connections:
            start_idx, end_idx = connection
            start_point = (
                int(hand_landmarks[start_idx].x * width),
                int(hand_landmarks[start_idx].y * height)
            )
            end_point = (
                int(hand_landmarks[end_idx].x * width),
                int(hand_landmarks[end_idx].y * height)
            )
            cv2.line(frame, start_point, end_point, CONNECTION_COLOR, THICKNESS)

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), RADIUS, LANDMARK_COLOR, -1)

    def get_index_finger_vector(self, hand_landmarks, frame_shape):
        """
        Computes the vector of the index finger from base to tip.
        Returns:
            (tip_x, tip_y): Coordinates of the fingertip.
            direction: Normalized direction vector of the index finger.
        """
        height, width, _ = frame_shape
        tip = hand_landmarks[8]
        base = hand_landmarks[5]
        tip_x, tip_y = int(tip.x * width), int(tip.y * height)
        base_x, base_y = int(base.x * width), int(base.y * height)
        direction = np.array([tip_x - base_x, tip_y - base_y])
        direction = direction / np.linalg.norm(direction)
        return (tip_x, tip_y), direction

    def get_centroid_hand(self, hand_landmarks, frame_shape):
        """
        Computes a simple centroid for the hand using the wrist, index base, and pinky base.
        Returns:
            (centroid_x, centroid_y): Coordinates of the hand's centroid.
        """
        height, width, _ = frame_shape
        wrist = hand_landmarks[0]
        index_base = hand_landmarks[5]
        pinky_base = hand_landmarks[17]

        wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
        index_base_x, index_base_y = int(index_base.x * width), int(index_base.y * height)
        pinky_base_x, pinky_base_y = int(pinky_base.x * width), int(pinky_base.y * height)

        centroid_x = (wrist_x + index_base_x + pinky_base_x) // 3
        centroid_y = (wrist_y + index_base_y + pinky_base_y) // 3
        return (centroid_x, centroid_y)

    def find_pointed_object(self, predicted_classes, tip, direction):
        """
        Examines the objects predicted by the YOLO model to find which one is being pointed at.
        Returns:
            pointed_object: The class name of the object currently pointed at, or None.
        """
        min_distance = float('inf')
        pointed_object = None
        for obj in predicted_classes:
            x1, y1, x2, y2 = obj["box"]
            obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            vector_to_obj = obj_center - np.array(tip)
            projection_length = np.dot(vector_to_obj, direction)
            if projection_length > 0:
                projected_point = np.array(tip) + projection_length * direction
                distance_to_center = np.linalg.norm(projected_point - obj_center)
                if distance_to_center < min_distance:
                    min_distance = distance_to_center
                    pointed_object = obj["class"]
        return pointed_object

    def draw_axis(self, img, p_, q_, color, scale):
        """
        Draws an axis (or arrow) between two points on the image, used here to visualize orientations.
        """
        p = list(p_)
        q = list(q_)

        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3)

        # create arrow hooks
        for i in [-1, 1]:
            px = q[0] + 9 * cos(angle + i * pi / 4)
            py = q[1] + 9 * sin(angle + i * pi / 4)
            cv2.line(img, (int(px), int(py)), (int(q[0]), int(q[1])), color, 3)

    def get_orientation(self, pts, img):
        """
        Uses PCA on contour points to determine the orientation angle of the object in the image.
        Returns:
            angle_deg (int): The orientation angle in degrees.
        """
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=np.empty((0)))

        # Store the center of the object
        cntr = (int(mean[0, 0]), int(mean[0, 1]))

        # Draw the principal axes
        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
              cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
              cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

        self.draw_axis(img, cntr, p1, (255, 255, 0), 1)
        self.draw_axis(img, cntr, p2, (0, 0, 255), 5)

        angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
        angle_deg = int(np.rad2deg(angle))

        return angle_deg

    def process_frame(self, frame, recognizer, yolo_model, timestamp_ms):
        """
        Processes a single frame for both gesture and object detection.
        Returns:
            frame: The annotated frame.
            predicted_classes (list): List of detected objects with bounding boxes and angles.
            index_tip (tuple): Coordinates of the index finger tip.
            center_hand (tuple): Coordinates of the hand's centroid.
            predicted_gesture (dict): The recognized gesture and its confidence.
            pointed_object: The object class currently pointed at (if any).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)
        predicted_gesture = {"gesture": None, "confidence": 0}

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            predicted_gesture = {'gesture': top_gesture.category_name, 'confidence': top_gesture.score}

        index_tip, index_direction = None, None
        center_hand = None

        if recognition_result.hand_landmarks:
            for hand_landmarks in recognition_result.hand_landmarks:
                index_tip, index_direction = self.get_index_finger_vector(hand_landmarks, frame.shape)
                center_hand = self.get_centroid_hand(hand_landmarks, frame.shape)
                self.draw_hand_landmarks(frame, hand_landmarks)

        results = yolo_model.predict(source=frame, stream=True, verbose=False)
        predicted_classes = []

        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()
                if confidence >= 0.75:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = yolo_model.names[int(box.cls[0])]
                    confidence = box.conf[0].item()

                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width // 2
                    center_y = y1 + height // 2
                    center_z = self.img.get_depth(center_x, center_y)

                    roi = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        rotation_angle = self.get_orientation(max_contour, roi)
                    else:
                        rotation_angle = None

                    predicted_classes.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2],
                        'width': width,
                        'height': height,
                        'center': [center_x, center_y, center_z],
                        'rotation_angle': rotation_angle
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)
                    cv2.putText(frame, f"{class_name} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, f"{width}x{height}",
                                (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if rotation_angle is not None:
                        cv2.putText(frame, f"Angle: {rotation_angle} deg",
                                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Determine the pointed object
        pointed_object = None
        if index_tip is not None and index_direction is not None:
            pointed_object = self.find_pointed_object(predicted_classes, index_tip, index_direction)

        return frame, predicted_classes, index_tip, center_hand, predicted_gesture, pointed_object

    def stop(self):
        """
        Sets the event that stops this thread's execution.
        """
        self._stop_event.set()

    def run(self):
        """
        Continuously capture and process frames until the thread is stopped or 'q' is pressed.
        """
        while not self._stop_event.is_set():
            timestamp_ms = int(time.time() * 1000)
            self.processed_frame, self.predicted_classes, self.index_tip, self.center_hand, \
                self.predicted_gesture, self.pointed_object = self.process_frame(
                    self.img.frame, self.gesture_recognizer, self.yolo_model, timestamp_ms)

            if self.show_camera:
                cv2.imshow("Camera Feed", self.processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = VisionClass(
        yolo_model_path='vision_sim/models/tools-100-epochs.pt',
        gesture_model_path='vision_sim/models/gesture_recognizer.task',
        show_camera=True
    )
    detector.run()
