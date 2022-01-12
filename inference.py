from model import model,translate_sentence, SRC,TRG,device
from input_data import extract_note
import sys

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

predicted_note = predict(file_in)