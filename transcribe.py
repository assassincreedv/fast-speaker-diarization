import threading
import time
import os

import torch
from pyannote.audio import Pipeline
import re


MODEL_PATH = os.getenv("MODEL_PATH", "./models/speaker-diarization-3.1/config.yaml")


# Device selection for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_time(seconds: float) -> float:
    """Format seconds to a float with two decimal places."""
    return round(seconds, 2)


def increment_speaker(speaker: str) -> str:
    match = re.match(r"SPEAKER_(\d+)", speaker)
    if match:
        num = int(match.group(1))
        return f"SPEAKER_{num + 1:02d}"
    return speaker  # fallback


class Transcribe:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline = Pipeline.from_pretrained("./models/pyannote_diarization_config.yaml")
        self._pipeline.to(device)

    # @staticmethod
    # def wav_to_np(file_obj):
    #     file_obj.read(44)
    #     raw_data = file_obj.read()
    #     if len(raw_data) % 2 != 0:
    #         raw_data = raw_data[:-1]
    #     samples = np.frombuffer(raw_data, dtype=np.int16)
    #     audio_array = samples.astype(np.float32) / np.iinfo(np.int16).max
    #     return audio_array
    #
    # def transcribe(self, file: BytesIO):
    #     with self._lock:
    #         output = self._whisper.transcribe(self.wav_to_np(file), temperature=0.2, temperature_inc=0.2,
    #                                           translate=False, language="auto")
    #     return "\n".join([seg.text for seg in output])
    #
    # def transcribe_with_time(self, file: BytesIO):
    #     with self._lock:
    #         output = self._whisper.transcribe(self.wav_to_np(file), temperature=0.2, temperature_inc=0.2,
    #                                           translate=False, language="auto", print_realtime=True, split_on_word=True)
    #     return output

    import time

    def process_task(self, file_path: str) -> list[dict[str, float | str]]:
        start_time = time.time()
        with self._lock:
            diarization = self._pipeline(file_path)
            data = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                new_speaker = increment_speaker(speaker)
                data.append({
                    "startTime": format_time(turn.start),
                    "endTime": format_time(turn.end),
                    "speaker": new_speaker
                })
        end_time = time.time()
        duration = end_time - start_time
        print(f"处理 {file_path} 用时：{duration:.2f} 秒")
        return data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python transcribe_module.py /path/to/audio.wav")
        exit(1)
    audio_file = sys.argv[1]
    transcribe = Transcribe()
    result = transcribe.process_task(audio_file)
    print(result)
