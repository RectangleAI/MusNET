    import librosa
    import numpy as np
    import torch
    import os
    from torch.utils.data import Dataset
    import pandas as pd

    class VoiceToGuitarDataset(Dataset):
        def __init__(self, data_dir,voice_paths,guitar_paths tokenizer, num_mels, hop_length):
            super(VoiceToGuitarDataset, self).__init__()
            self.data_dir = data_dir
            # self.annotations = pd.read_csv(csv_file)
            self.voice_paths = list(voice_paths)
            self.guitar_paths = list(guitar_paths)

            self.num_mels = num_mels
            self.hop_length = 512

        def _parse_voice_audio(self, audio_path):
            y, _ = librosa.load(audio_path, sr=16_000)
            S = np.abs(librosa.stft(y, hop_length=self.hop_length, n_fft=self.hop_length*2))
            mel_spec = librosa.feature.melspectrogram(S=S, sr=16_000, n_mels=self.num_mels, hop_length=self.hop_length)


            return torch.FloatTensor(mel_spec).transpose(0, 1)

        def _parse_guitar_audio(self, audio_path):
            y, _ = librosa.load(audio_path, sr=16_000)
            S = np.abs(librosa.stft(y, hop_length=self.hop_length, n_fft=self.hop_length*2))
            mel_spec = librosa.feature.melspectrogram(S=S, sr=16_000, n_mels=self.num_mels, hop_length=self.hop_length)


            return torch.FloatTensor(mel_spec).transpose(0, 1)

        def __getitem__(self, idx):
            voice = self._parse_voice_audio(self.voice_paths[idx])
            guitar = self._parse_guitar_audio(self.guitar_paths[idx])
            return voice, guitar

        def __len__(self):
            return len(self.guitar_paths)