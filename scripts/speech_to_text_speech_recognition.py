import speech_recognition as sr

def transcribe_recording(file_name):
	RECORDING_DIR = 'recordings/'

	r = sr.Recognizer()

	recording = sr.AudioFile(RECORDING_DIR + file_name)

	with recording as source:
		# r.adjust_for_ambient_noise(source)
		audio = r.record(source)

	try:
		return r.recognize_google(audio)
	except sr.RequestError:
		# Could not reach API
		return 'API unavailable'
	except sr.UnknownValueError:
		# Could not make sense of speech
		return 'Recording unintelligible'

if __name__ == "__main__":
	recording1 = 'harvard.wav'
	text = transcribe_recording(recording1)
	print(text)