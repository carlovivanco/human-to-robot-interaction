# !pip install SpeechRecognition pyaudio
import speech_recognition as sr


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
        audio = recognizer.listen(source)  # Listen for speech

    try:
        text = recognizer.recognize_google(audio)  # Convert speech to text
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Could not request results, check your internet connection.")


if __name__ == "__main__":
    recognize_speech()
