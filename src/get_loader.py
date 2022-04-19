from logging import root
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
# import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from glob import glob
import torchaudio

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
# spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
        return [tok.lower() for tok in text.split(',')]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:            
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class Voice2NoteDataset(Dataset):
    def __init__(self, root_dir, captions_file, transformation,target_sample_rate, 
                 num_samples, device, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        # self.df = pd.read_csv(captions_file)
        self.transform = transform
        with open(captions_file,encoding='utf-8',mode='r') as gf:
            self.captions = gf.readlines()
        # Get img, caption columns
        # self.imgs = self.df["image"]
        self.aud = os.listdir(root_dir)
        # self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        # self.vocab.build_vocabulary(self.captions.tolist())
        self.vocab.build_vocabulary(self.captions)

        self.device = device
        self.transformation = transformation#.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        

    def __len__(self):
        return len(self.aud)


    def __getitem__(self, index) :
        audio_sample_path = self._get_audio_sample_path(index)
        caption = self.captions[index]
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = signal.to(self.device)
        signal = self.transformation(signal)
        return signal, torch.tensor(numericalized_caption)

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples ]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.root_dir, self.aud[index])
        return path
    

    


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    transform,
    target_sample_rate,
    num_samples,
    device,
    batch_size=1,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    

):
    dataset = Voice2NoteDataset(root_folder, annotation_file, transform,
                                target_sample_rate, num_samples, device, freq_threshold=5)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 131071

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels=299)

    loader, dataset = get_loader(
        "./data/voice_wav/", "./data/guitarp.txt", transform=mel_spectogram, target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES, device=device
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
        break