# !pip install transformers
# !pip install tf-keras
# !pip install SpeechRecognition pyaudio
import speech_recognition as sr
from transformers import pipeline
import threading


class SpeechRecognition(threading.Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self.tools = ["Hammer", "Screwdriver", "Wrench", "Pliers"]
        self.actions = ["pass", "take"]  # , "use", "drop"]
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.tool = None
        self.action = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        while True:
            self.text = self.recognize_speech()
            if not self.text == "quit":
                self.tool, self.action = self.extract_tool_and_action(self.text)
                print(f"Detected Tool: {self.tool}")
                print(f"Detected Action: {self.action}")
            else:
                self.tool = None
                self.action = None

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
            print("Listening for 'panda'...")
            while True:
                try:
                    audio = recognizer.listen(source)  # Listen continuously
                    text = recognizer.recognize_google(audio)  # Convert speech to text
                    print("You said:", text)
                    if "panda" in text.lower():
                        print("Trigger detected! Processing command...")
                        return text
                    elif "stop" in text.lower():
                        print("Work stopped")
                        return "quit"
                except sr.UnknownValueError:
                    print('Unrecognized speech')
                except sr.RequestError:
                    print("Could not request results, check your internet connection.")

    def extract_tool_and_action(self, input_text):
        tool_result = self.classifier(input_text, candidate_labels=self.tools)
        detected_tool = tool_result['labels'][0]

        action_result = self.classifier(input_text, candidate_labels=self.actions)
        detected_action = action_result['labels'][0]

        return detected_tool, detected_action


if __name__ == '__main__':
    command = SpeechRecognition()
    command.run()
