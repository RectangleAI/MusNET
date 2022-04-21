import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from cnn_rnn import CNNtoRNN
import torchaudio
import pickle

def train():

    spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=22500,
        n_fft = 1024,
        hop_length=512,
        n_mels=299
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, dataset = get_loader(
        root_folder="/content/drive/MyDrive/DataSetVoice/voice_wav",
        annotation_file= "./data/guitarp.txt",
        transform=spectogram,
        target_sample_rate=22500,
        num_samples=262144,
        device=device,
        num_workers=2,
    )
    
    with open('data/vocab','wb') as f:
        pickle.dump(dataset.vocab,f)
        
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 2
    # for tensorboard
    writer = SummaryWriter("runs/voice2n")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.model.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
