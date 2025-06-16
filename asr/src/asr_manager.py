"""Manages the ASR model."""

from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch
import io
import soundfile as sf

class ASRManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        
        self.model = EncDecRNNTBPEModel.restore_from("model/default-1.1b.nemo")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f'!!!!!!!{self.device}')

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """

        audio_file = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_file)
        
        transcription = self.model.transcribe([waveform])[0].text
        
        return transcription
