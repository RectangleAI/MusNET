from audio2midi import run
import librosa
import numpy as np
from mido import MidiFile
import sys
import csv

# song = "13.wav"
# run(song,'13.mid')
def extract_note(input_file):
    run(f"{input_file}.wav",f"{input_file}.mid")
    midi = MidiFile(f"{input_file}.mid")
    music_notes = []
    for i, track in enumerate(midi.tracks):
        print('Track {}: {}'.format(i, track.name))
        for line in track:
            if len(line.bytes()) != 3:
                continue
            note = librosa.midi_to_note(line.bytes())
            music_notes.append(note)
    return music_notes

if __name__ =="__main__":
    file_in = sys.argv[1]
    file_in = file_in.split('.')[0]
    notes = extract_note(file_in)
    with open(f"{file_in}.csv", "w", newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(notes)
