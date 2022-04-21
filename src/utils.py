import torch
import torchvision.transforms as transforms
from PIL import Image
import torchaudio


def audio_to_melspec(audio_path,target_sample_rate,num_samples,):
    def resample_if_necessary(signal, sr):
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def mix_down_if_necessary(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def cut_if_necessary( signal):
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > num_samples:
            signal = signal[:,:num_samples ]
        return signal
    
    def right_pad_if_necessary(signal):
        length_signal = signal.shape[1]
        if length_signal < num_samples:
            num_missing_samples = num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=22500,
        n_fft = 1024,
        hop_length=512,
        n_mels=299)

    signal, sr = torchaudio.load(audio_path)
    signal = resample_if_necessary(signal, sr)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal)
    signal = right_pad_if_necessary(signal)
    signal = transformation(signal)

    return signal

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    with open("./data/guitarp.txt",encoding='utf-8',mode='r') as gf:
        captions = gf.readlines()

    model.eval()
    test_aud1 = transform(audio_to_melspec("test_examples/12.wav"))
    print("Example 1 CORRECT: {captions[12]} ")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_aud1.to(device), dataset.vocab))
    )
    test_aud2 = transform(audio_to_melspec("test_examples/19.wav"))
    print("Example 2 CORRECT: {captions[19]} ")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_aud2.to(device), dataset.vocab))
    )
    test_aud3 = transform(audio_to_melspec("test_examples/20.wav"))
    print("Example 3 CORRECT: {captions[20]} ")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_aud3.to(device), dataset.vocab))
    )
    test_aud4 = transform(audio_to_melspec("test_examples/30.wav"))
    print("Example 4 CORRECT: {captions[30]} ")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_aud4.to(device), dataset.vocab))
    )
    test_aud5 = transform(audio_to_melspec("test_examples/1.wav"))
    print("Example 5 CORRECT: {captions[1]} ")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_aud5.to(device), dataset.vocab))
    )
    model.train()

# def print_examples(model, device, dataset):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((299, 299)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )

#     model.eval()
#     test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(
#         0
#     )
#     print("Example 1 CORRECT: Dog on a beach by the ocean")
#     print(
#         "Example 1 OUTPUT: "
#         + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
#     )
#     test_img2 = transform(
#         Image.open("test_examples/child.jpg").convert("RGB")
#     ).unsqueeze(0)
#     print("Example 2 CORRECT: Child holding red frisbee outdoors")
#     print(
#         "Example 2 OUTPUT: "
#         + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
#     )
#     test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(
#         0
#     )
#     print("Example 3 CORRECT: Bus driving by parked cars")
#     print(
#         "Example 3 OUTPUT: "
#         + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
#     )
#     test_img4 = transform(
#         Image.open("test_examples/boat.png").convert("RGB")
#     ).unsqueeze(0)
#     print("Example 4 CORRECT: A small boat in the ocean")
#     print(
#         "Example 4 OUTPUT: "
#         + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
#     )
#     test_img5 = transform(
#         Image.open("test_examples/horse.png").convert("RGB")
#     ).unsqueeze(0)
#     print("Example 5 CORRECT: A cowboy riding a horse in the desert")
#     print(
#         "Example 5 OUTPUT: "
#         + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
#     )
#     model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step