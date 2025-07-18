# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
#from audiolab import Reader, Writer
import soundfile as sf
#from audiolab.av import AudioGraph, aformat
from tqdm import tqdm
from scipy.signal import resample


from .rnnoise import FRAME_SIZE, FRAME_SIZE_MS, SAMPLE_RATE, create, destroy, process_frame


class RNNoise:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.channels = None
        self.denoise_states = None


    def __del__(self):
        if self.denoise_states is not None:
            for denoise_state in self.denoise_states:
                destroy(denoise_state)

    def reset(self):
        self.denoise_states = None

    # def process_frame(self, frame: np.ndarray, partial: bool = False):
    #     if self.denoise_states is None:
    #         self.denoise_states = [create() for _ in range(self.channels)]
    #     denoised_frame, speech_probs = process_frame(self.denoise_states, frame)
    #     if self.sample_rate != SAMPLE_RATE:
    #         self.out_graph.push(denoised_frame)
    #         denoised_frame = np.concatenate([frame for frame, _ in self.out_graph.pull(partial)], axis=1)
    #     return speech_probs, denoised_frame

    # def process_chunk(self, chunk: np.ndarray, partial: bool = False):
    #     chunk = np.atleast_2d(chunk)
    #     # [num_channels, num_samples]
    #     self.channels = chunk.shape[0]
    #     self.dtype = chunk.dtype
    #     #self.in_graph.push(chunk)
    #     frames = [frame for frame, _ in self.in_graph.pull(partial)]
    #     for idx, frame in enumerate(frames):
    #         yield self.process_frame(frame, partial and (idx == len(frames) - 1))

    # def process_wav(self, in_path, out_path):
    #     reader = Reader(in_path, dtype=np.int16, frame_size_ms=FRAME_SIZE_MS)
    #     writer = Writer(out_path, reader.rate, reader.codec, layout=reader.layout)
    #     for idx, (frame, _) in tqdm(enumerate(reader), desc="Denoising", total=reader.num_frames, unit="frames"):
    #         partial = idx == reader.num_frames - 1
    #         for speech_prob, frame in self.process_chunk(frame, partial):
    #             writer.write(frame)
    #             yield speech_prob
    #     writer.close()

    def process_frame(self, frame: np.ndarray, partial: bool = False):
        if self.denoise_states is None:
            self.denoise_states = [create() for _ in range(self.channels)]

        # Resample to 48kHz if needed
        if self.sample_rate != SAMPLE_RATE:
            num_samples = int(FRAME_SIZE * self.sample_rate / SAMPLE_RATE)
            frame = resample(frame, num_samples, axis=1)

        denoised_frame, speech_probs = process_frame(self.denoise_states, frame)

        # Resample back to original rate if needed
        if self.sample_rate != SAMPLE_RATE:
            denoised_frame = resample(denoised_frame, frame.shape[1], axis=1)

        return speech_probs, denoised_frame

    def process_chunk(self, chunk: np.ndarray, partial: bool = False):
        chunk = np.atleast_2d(chunk)
        self.channels = chunk.shape[0]
        total_samples = chunk.shape[1]

        # Frame size in samples
        frame_size = int(self.sample_rate * FRAME_SIZE_MS / 1000)

        # Split into frames
        num_frames = total_samples // frame_size
        for i in range(num_frames):
            frame = chunk[:, i * frame_size:(i + 1) * frame_size]
            yield self.process_frame(frame, partial and (i == num_frames - 1))

    def process_wav(self, in_path, out_path):
        with sf.SoundFile(in_path, mode='r') as reader:
            self.sample_rate = reader.samplerate
            self.channels = reader.channels
            subtype = reader.subtype

            frame_size = int(self.sample_rate * FRAME_SIZE_MS / 1000)

            with sf.SoundFile(out_path, mode='w', samplerate=self.sample_rate,
                              channels=self.channels, subtype=subtype) as writer:

                total_frames = len(reader) // frame_size

                for idx in tqdm(range(total_frames), desc="Denoising", unit="frames"):
                    data = reader.read(frames=frame_size, dtype='int16', always_2d=True).T
                    if data.shape[1] < frame_size:
                        # Pad last frame
                        pad_width = frame_size - data.shape[1]
                        data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')

                    partial = idx == total_frames - 1
                    for speech_prob, processed in self.process_chunk(data, partial):
                        writer.write(processed.T)
                        yield speech_prob
    
    def process_wav_to_array(self, in_path):
        with sf.SoundFile(in_path, mode='r') as reader:
            self.sample_rate = reader.samplerate
            self.channels = reader.channels
            dtype = 'int16'

            frame_size = int(self.sample_rate * FRAME_SIZE_MS / 1000)
            total_samples = len(reader)
            total_frames = total_samples // frame_size

            output_frames = []

            for idx in tqdm(range(total_frames), desc="Denoising", unit="frames"):
                data = reader.read(frames=frame_size, dtype=dtype, always_2d=True).T  # [channels, frame]
                if data.shape[1] < frame_size:
                    pad_width = frame_size - data.shape[1]
                    data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')

                partial = idx == total_frames - 1
                for _, processed in self.process_chunk(data, partial):
                    output_frames.append(processed)

            # Concatenate along time axis
            output_audio = np.concatenate(output_frames, axis=1)  # shape: [channels, samples]

            return output_audio.T.astype(np.int16)  # return as [samples, channels]
    
    def process_array(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
        """
        Denoise an in-memory audio array using RNNoise.

        Args:
            audio (np.ndarray): Input audio, shape (samples,) for mono or (samples, channels) for stereo.
            sample_rate (int): Sample rate of the input audio.

        Returns:
            np.ndarray: Denoised audio, same shape as input.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # [1, samples]
        elif audio.ndim == 2:
            audio = audio.T  # [channels, samples]
        else:
            raise ValueError("Input audio must be 1D or 2D numpy array")

        self.sample_rate = sample_rate
        self.channels = audio.shape[0]

        frame_size = int(sample_rate * FRAME_SIZE_MS / 1000)
        total_samples = audio.shape[1]
        num_frames = total_samples // frame_size

        self.reset()  # Ensure fresh state
        output_frames = []

        for i in range(num_frames):
            frame = audio[:, i * frame_size:(i + 1) * frame_size]
            if frame.shape[1] < frame_size:
                pad_width = frame_size - frame.shape[1]
                frame = np.pad(frame, ((0, 0), (0, pad_width)), mode='constant')

            partial = i == num_frames - 1
            for _, processed in self.process_chunk(frame, partial):
                output_frames.append(processed)

        output = np.concatenate(output_frames, axis=1)  # [channels, samples]

        return output.T.astype(np.int16)  # [samples, channels]

