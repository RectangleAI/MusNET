from webbrowser import get
from model import model,translate_sentence, SRC,TRG,device
from input_data import extract_note
import sys
from wav_note_wav import highpass,samplerate,get_song_data
import numpy as np

# input_file = "13.wav"

# note = "['c♯11', 'g5', 'd#-1']"
def predict(input_file):
    voice_note = extract_note(input_file.split('.')[0])
    predg_note = []
    # print(voice_note)
    print()
    for note in voice_note:
        translation, attention = translate_sentence(note, SRC, TRG, model, device)
        predg_note.append(translation)
    return predg_note
# translation, attention = translate_sentence(note, SRC, TRG, model, device)
# print("Welcome!")

file_in = sys.argv[1]
file_out = sys.argv[2]

predicted_note = predict(file_in)

data = get_song_data(predicted_note)
data = data * (16300/np.max(data)) # Adjusting the Amplitude (Optional)
# data = highpass(data,50,samplerate,order=12)
data = data[:,1:2]
from scipy.io.wavfile import write
write(file_out, samplerate, data.astype(np.int16))