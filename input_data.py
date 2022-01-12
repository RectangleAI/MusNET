from audio2midi import run
import librosa
import numpy as np
from mido import MidiFile


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
