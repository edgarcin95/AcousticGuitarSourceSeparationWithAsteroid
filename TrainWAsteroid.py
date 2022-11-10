# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import cuda, device
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from asteroid.models import SuDORMRFImprovedNet, DCUNet, DPTNet
from asteroid.losses import PITLossWrapper, pairwise_mse, pairwise_neg_sisdr
from asteroid.losses import singlesrc_mse, singlesrc_neg_sisdr, singlesrc_neg_sdsdr, singlesrc_neg_snr
# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
from torch.utils.data import Dataset, DataLoader
import torch, torchaudio, argparse, random
import sys, os, json, yaml, tqdm
from pathlib import Path
from typing import Optional, List

#Función usada en la clase AGlignedDS
def load_info(path: str) -> dict:
	"""Load audio metadata
	this is a backend_independent wrapper around torchaudio.info
	Args:
		path: Path of filename
	Returns:
		Dict: Metadata with
		`samplerate`, `samples` and `duration` in seconds
	"""
	# get length of file in samples
	if torchaudio.get_audio_backend() == "sox":
		raise RuntimeError("Deprecated backend is not supported")

	info = {}
	si = torchaudio.info(str(path))
	info["samplerate"] = si.sample_rate
	info["samples"] = si.num_frames
	info["channels"] = si.num_channels
	info["duration"] = info["samples"] / info["samplerate"]
	return info

#Función usada en la clase AGlignedDS
def load_audio(
	path: str,
	start: float = 0.0,
	dur: Optional[float] = None,
	info: Optional[dict] = None,
):
	"""Load audio file
	Args:
		path: Path of audio file
		start: start position in seconds, defaults on the beginning.
		dur: end position in seconds, defaults to `None` (full file).
		info: metadata object as called from `load_info`.
	Returns:
		Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
	"""
	# loads the full track duration
	if dur is None:
		# we ignore the case where start!=0 and dur=None
		# since we have to deal with fixed length audio
		sig, rate = torchaudio.load(path)
		return sig, rate
	else:
		if info is None:
			info = load_info(path)
		num_frames = int(dur * info["samplerate"])
		frame_offset = int(start * info["samplerate"])
		sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset, normalize=True)
		return sig, rate

#Clase para cargar Dataset, misma clase usada en OpenUnmix
class AGlignedDS(Dataset):
	def __init__(
		self,
		root: str,
		split: str = "train",
		input_file: str = "Mix.wav",
		output_file: str = "AG.wav",
		seq_duration: Optional[float] = None,
		random_chunks: bool = False,
		sample_rate: float = 44100.0) -> None:
		"""A dataset of that assumes multiple track folders
		where each track includes and input and an output file
		which directly corresponds to the the input and the
		output of the model. This dataset is the most basic of
		all datasets provided here, due to the least amount of
		preprocessing, it is also the fastest option, however,
		it lacks any kind of source augmentations or custum mixing.
		Typical use cases:
		* Source Separation (Mixture -> Target)
		* Denoising (Noisy -> Clean)
		* Bandwidth Extension (Low Bandwidth -> High Bandwidth)
		Example
		=======
		data/train/01/mixture.wav --> input
		data/train/01/vocals.wav ---> output
		"""
		self.root = Path(root).expanduser()
		self.split = split
		self.sample_rate = sample_rate
		self.seq_duration = seq_duration
		self.random_chunks = random_chunks
		# set the input and output files (accept glob)
		self.input_file = input_file
		self.output_file = output_file
		self.tuple_paths = list(self._get_paths())
		if not self.tuple_paths:
			raise RuntimeError("Dataset is empty, please check parameters")

	def __getitem__(self, index):
		input_path, output_path = self.tuple_paths[index]

		if self.random_chunks:
			input_info = load_info(input_path)
			output_info = load_info(output_path)
			duration = min(input_info["duration"], output_info["duration"])
			start = random.uniform(0, duration - self.seq_duration)
		else:
			start = 0

		X_audio, srX = load_audio(input_path, start=start, dur=self.seq_duration)
		Y_audio, srY = load_audio(output_path, start=start, dur=self.seq_duration)
		X_audio = self.resampleIfNecessary(X_audio, srX)
		Y_audio = self.resampleIfNecessary(Y_audio, srY)
		X_audio = X_audio[0, :]
		Y_audio = Y_audio[0, :]
		#X_audio = torch.unsqueeze(X_audio, 0)
		Y_audio = torch.unsqueeze(Y_audio, 0)
		# return torch tensors
		return X_audio, Y_audio

	def __len__(self):
		return len(self.tuple_paths)

	def _get_paths(self):
		"""Loads input and output tracks"""
		p = Path(self.root, self.split)
		for track_path in tqdm.tqdm(p.iterdir()):
			if track_path.is_dir():
				input_path = list(track_path.glob(self.input_file))
				output_path = list(track_path.glob(self.output_file))
				if input_path and output_path:
					if self.seq_duration is not None:
						input_info = load_info(input_path[0])
						output_info = load_info(output_path[0])
						min_duration = min(input_info["duration"], output_info["duration"])
						# check if both targets are available in the subfolder
						if min_duration > self.seq_duration:
							yield input_path[0], output_path[0]
					else:
						yield input_path[0], output_path[0]
	
	def resampleIfNecessary(self, signal, sr):
		if sr != self.sample_rate:
			resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
			signal = resampler(signal)
		return signal
'''
#Mi clase creada para cargar mi dataset
class AGDS(Dataset):

	def __init__(self, audio_dir, split, segsDur, sample_rate):
		self.audio_dir = audio_dir
		self.sample_rate = sample_rate
		self.segs = segsDur
		self.split = split
		self.inputFile = 'Mix.wav'
		self.outputFile ='AG.wav'
		self.pathsX = []
		self.pathsY = []
		self.root = os.path.join(self.audio_dir, self.split)
		for songName in os.listdir(self.root):
			songFullPth = os.path.join(self.root, songName)
			self.pathsX.append(os.path.join(songFullPth, self.inputFile))
			self.pathsY.append(os.path.join(songFullPth, self.outputFile))

	def __len__(self):
		return len(os.listdir(self.root))

	def __getitem__(self, i):
		pathX = self.pathsX[i]
		pathY = self.pathsY[i]
		samples = int(self.segs * self.sample_rate)
		signalX, sr = torchaudio.load(pathX, num_frames=samples)
		signalY, sr = torchaudio.load(pathY, num_frames=samples)
		signalX = signalX[0, :]
		#signalY = signalY[0, :]
		signalX = torch.squeeze(signalX)
		#signalY = torch.squeeze(signalY)
		return signalX, signalY
'''

#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--segs', type=float, default=3.0,
					help='Sequence duration in seconds to take from the dataset songs')
parser.add_argument('--sampRate', type=int, default=44100, help='Sample Rate in Hertz')
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--nWorkers', type=int, default=2, help='Number of workers for dataloader')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patienc', type=int, default=140, help='Max num. of train epochs before early stopping')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--modlNam', type=str, default='defaultFolder',
					help="Name of the model, also output path folder name")
args = parser.parse_args()

#Set seeds
torch.manual_seed(42)
random.seed(42)

#Create directory to save
os.makedirs(args.modlNam, exist_ok=True)

#Save Arguments
with open(Path(Path(args.modlNam), 'Arguments' + ".json"), "w") as outfile:
    outfile.write(json.dumps(vars(args), indent=4, sort_keys=True))

# Datasets y DataLoaders
trDs = AGlignedDS(root='../Datos/DB44-1kHz24bitsAlignd/', split='train',
					seq_duration=args.segs, random_chunks=True, sample_rate=args.sampRate)
valDs = AGlignedDS(root='../Datos/DB44-1kHz24bitsAlignd/', split='valid',
					seq_duration=args.segs, random_chunks=True, sample_rate=args.sampRate)
trainLoader = DataLoader(trDs, batch_size=args.batchSize, shuffle=True,
							num_workers=args.nWorkers, drop_last=True)
validLoader = DataLoader(valDs, batch_size=args.batchSize, shuffle=False,
							num_workers=args.nWorkers, drop_last=True)
'''
print(f"There are {len(trDs)} samples in the training dataset.")
print(f"There are {len(valDs)} samples in the validation dataset.")
x, y = next(iter(trainLoader))
print(f'x shape={x.shape} dtype={x.dtype}')
print(f'y shape={y.shape} dtype={y.dtype}')
x, y = next(iter(trainLoader))
print(f'x shape={x.shape} dtype={x.dtype}')
print(f'y shape={y.shape} dtype={y.dtype}')
'''

#Model
model = SuDORMRFImprovedNet(n_src=1, bn_chan=128, num_blocks=8, upsampling_depth=4,
							mask_act='relu', in_chan=None, fb_name='free', kernel_size=21,
							n_filters=1024, stride=10, sample_rate=args.sampRate)
'''
model = DPTNet(n_src=1, n_heads=8, ff_hid=256, chunk_size=100, hop_size=None, n_repeats=4,
				norm_type='gLN', ff_activation='relu', encoder_activation='relu',
				mask_act='relu', bidirectional=True, dropout=0, in_chan=None, fb_name='free',
				kernel_size=16, n_filters=64, stride=8, sample_rate=args.sampRate)#, out_chan=150)
model = DCUNet(n_src=1, architecture='DCUNet-20', stft_n_filters=2048, stft_kernel_size=1024,
				stft_stride=256, sample_rate=args.sampRate, fix_length_mode='pad')
'''

#Optimizer
optimzr = Adam(model.parameters(), lr=args.lr)

# Define scheduler
scheduler = ReduceLROnPlateau(optimizer=optimzr, factor=0.5, patience=25, verbose=True, eps=1.0e-10)
											#Probar 0.1 o 0.5, Probar 80 a 10

#Loss Function. PITLossWrapper works with any loss function.
lossFunc = PITLossWrapper(singlesrc_mse, pit_from="pw_pt")#"pw_mtx")

#System object from Asteroid for Pytorch Lighting
system = System(model=model, optimizer=optimzr, loss_func=lossFunc,
				train_loader=trainLoader, val_loader=validLoader, scheduler=scheduler)

# Define callbacks
callbacks = []
checkpoint_dir = os.path.join(args.modlNam, "checkpoints/")
checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_loss", mode="min",
								save_top_k=4, verbose=True, save_last=True)
callbacks.append(checkpoint)
callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=args.patienc, verbose=True))

#Accelerator backend
distributdBacknd = "dp" if cuda.is_available() else None

#Train
trainer = Trainer(max_epochs=args.epochs, callbacks=callbacks, default_root_dir=args.modlNam,
					gpus=2, accelerator=distributdBacknd)#, gradient_clip_val=5.0)
trainer.fit(system)

#Save best top models
best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
with open(os.path.join(args.modlNam, "best_k_models.json"), "w") as f:
	json.dump(best_k, f, indent=0)

state_dict = torch.load(checkpoint.best_model_path)
system.load_state_dict(state_dict=state_dict["state_dict"])
system.cpu()

#Save best model
to_save = system.model.serialize()
torch.save(to_save, os.path.join(args.modlNam, "best_model.pth"))


