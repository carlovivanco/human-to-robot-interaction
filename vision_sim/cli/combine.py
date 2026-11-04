# !pip install transformers
# !pip install tf-keras
# !pip install SpeechRecognition pyaudio faster-whisper
import ctypes
import glob
import site

import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from transformers import pipeline
import threading

# ctranslate2 needs libcublas.so.12; the pip package nvidia-cublas-cu12 ships it outside the loader path
for _dir in site.getsitepackages():
    for _lib in glob.glob(f"{_dir}/nvidia/cublas/lib/libcublas*.so.12"):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

# biases Whisper towards the wake word and the tool names
PROMPT = "Panda, pass me the hammer. Panda, hand me the wrench, the screwdriver or the pliers."


def load_stt(name):
    """GPU if it really works (ctranslate2 reports a device even when the CUDA libraries are missing), else CPU."""
    for device, compute in (("cuda", "float16"), ("cpu", "int8")):
        try:
            model = WhisperModel(name, device=device, compute_type=compute)
            list(model.transcribe(np.zeros(16000, np.float32), language="en")[0])  # raises here if CUDA is broken
            return model
        except Exception as e:
            print(f"Whisper on {device} failed ({str(e)[:80]})")
    raise RuntimeError("could not load the Whisper model")


class SpeechRecognition(threading.Thread):
    def __init__(self, stt_model="large-v3-turbo", language="en"):
        super().__init__()
        self.language = language
        self.stt = load_stt(stt_model)
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
                audio = recognizer.listen(source)  # Listen continuously
                text = self.transcribe(audio)  # local Whisper, no internet needed
                if not text:
                    continue
                print("You said:", text)
                if "panda" in text.lower():
                    print("Trigger detected! Processing command...")
                    return text
                elif "stop" in text.lower():
                    print("Work stopped")
                    return "quit"

    def transcribe(self, audio):
        pcm = np.frombuffer(audio.get_raw_data(convert_rate=16000, convert_width=2), np.int16)
        segments, _ = self.stt.transcribe(
            pcm.astype(np.float32) / 32768, language=self.language, beam_size=5, vad_filter=True,
            condition_on_previous_text=False, initial_prompt=PROMPT)
        return " ".join(seg.text for seg in segments).strip()

    def extract_tool_and_action(self, input_text):
        tool_result = self.classifier(input_text, candidate_labels=self.tools)
        detected_tool = tool_result['labels'][0]

        action_result = self.classifier(input_text, candidate_labels=self.actions)
        detected_action = action_result['labels'][0]

        return detected_tool, detected_action


if __name__ == '__main__':
    command = SpeechRecognition()
    command.run()
