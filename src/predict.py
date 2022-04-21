import pickle
from utils import audio_to_melspec, load_checkpoint
from cnn_rnn import CNNtoRNN
import torch.optim as optim
import torch



check_point = "model/my_checkpoint.pth.tar"
with open('data/vocab', 'rb') as pickle_file:
    vocab = pickle.load(pickle_file)

embed_size = 256
hidden_size = 256
vocab_size = len(vocab)
num_layers = 1
learning_rate = 3e-4
device = 'cpu'

# initialize model, loss etc
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to('cpu')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

audio_path = "data/voice_wav/5.wav"
sample_rate=22500
n_fft = 1024
hop_length=512
n_mels=299
num_samples=262144
checkpoint = torch.load(check_point,map_location=torch.device('cpu'))
print(checkpoint.keys())
print("=> Loading checkpoint")
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])
step = checkpoint["step"]

spectogram = audio_to_melspec(audio_path=audio_path,    
                                    target_sample_rate=sample_rate, num_samples=num_samples)
print(model)
print(spectogram.shape)
vvv = " ".join(model.caption_image(spectogram.to(device).unsqueeze(0), vocab))
print(vvv)