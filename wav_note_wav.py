from pydub import AudioSegment
import os
from audio_to_midi.audio2midi import run
from mido import MidiFile
import librosa
import numpy as np
# song = "music/twinkle-twinkle.wav"
# run(song,'twinkle.mid')
midi = MidiFile("twinkle.mid")
music_notes = []
for i, track in enumerate(midi.tracks):
    print('Track {}: {}'.format(i, track.name))
    for line in track:
        note = librosa.midi_to_note(line.bytes())
        music_notes.append(note)
        print(line.bytes(), note)

# sound = AudioSegment.from_file(song,format="wav")
# sound.export(os.path.join(song.split('.')[0]+'.wav'), 
#                format="wav")
samplerate = 44100 #Frequecy in Hz



def get_wave(freq, duration=1):
    '''
    Function takes the "frequecy" and "time_duration" for a wave 
    as the input and returns a "numpy array" of values at all points 
    in time
    '''
    
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq*t[:,np.newaxis])
    
    return wave


def get_song_data(music_notes):
    '''
    Function to concatenate all the waves (notes)
    '''
    print(len(music_notes))
    print(type(music_notes))
    note_freqs = librosa.note_to_hz(music_notes) # Function that we made earlier
    print(len(note_freqs))
    print(type(note_freqs))
    song = [get_wave(fre) for fre in note_freqs]
    print(type(song[0]))
    song = np.concatenate(song)
    return song

data = get_song_data(music_notes[1:])

data = data * (16300/np.max(data)) # Adjusting the Amplitude (Optional)


from scipy.io.wavfile import write
write('twinkle-twinkle2.wav', samplerate, data.astype(np.int16))