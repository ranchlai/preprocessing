# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import argparse
import multiprocessing
import os
from os import PathLike
from typing import List, Union
import time
import librosa
import numpy as np
import paddleaudio
import yaml
from paddleaudio.utils.log import Logger
import paddle
logger = Logger(__file__)


class FeatureExtractor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.transform = paddleaudio.transforms.MelSpectrogram(**kwargs)

    def process_wav(self, wav: Union[PathLike, np.ndarray]) -> np.ndarray:

        if isinstance(wav, str):
            #import pdb;pdb.set_trace()
            #wav, sr = librosa.load(wav, sr=None)
            target_sr = self.kwargs.get('sr')
            try:
                #wav, sr = paddleaudio.load(wav, sr=target_sr)
                wav, sr = librosa.load(
                    wav, sr=target_sr, res_type='kaiser_fast')
            except:
                print(f'error load {f}')
                return None
            #assert sr == target_sr, f'sr: {sr} ~= {target_sr}'

            #if wav.dtype == 'int16':
            # wav = pa.depth_convert(wav, 'float32')
        wav = paddle.to_tensor(wav).unsqueeze(0)
        x = self.transform(wav)
        x = paddle.log(paddle.clip(x, 1e-5))
        return x.numpy()


def wav_list_to_fbank(wav_list: List[PathLike],
                      key_list: List[str],
                      mel_folder: PathLike,
                      feature_extractor: FeatureExtractor) -> None:
    """Convert wave list to fbank
    """

    logger.info(f'saving to {mel_folder}')
    logger.info(f'{len(wav_list)} wav files listed')
    for f, key in zip(wav_list, key_list):
        x = feature_extractor.process_wav(f)
        if x is not None:
            dst_file = os.path.join(mel_folder,
                                    f.split('/')[-1].split('.')[0] + '.npy')
            np.save(dst_file, x)


def wav_to_fbank_mp(params):
    """Convert wave list to fbank, store into an h5 file, multiprocessing warping"""
    import paddle
    f, key, output_folder = params
    x = feature_extractor.process_wav(f)
    dst_file = os.path.join(output_folder,
                            f.split('/')[-1].split('.')[0] + '.npy')
    np.save(dst_file, x)


def read_scp(scp_file):
    scp_file = os.path.expanduser(scp_file)
    with open(scp_file) as f:
        lines = f.read().split('\n')

    names = [l.split()[0] for l in lines if len(l) > 1]
    files = [l.split()[1] for l in lines if len(l) > 1]
    return names, files


def read_list(list_file):
    list_file = os.path.expanduser(list_file)
    with open(list_file) as f:
        lines = f.read().split('\n')
    lines = [l for l in lines if len(l) > 1]
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wave2mel')
    parser.add_argument(
        '-c', '--config', type=str, required=True, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    paddle.set_device(config['device'])
    feature_extractor = FeatureExtractor(**config['fbank'])
    try:
        names, wav_files = read_scp(config['wav_scp'])
    except:
        wav_files = read_list(config['wav_scp'])
        names = [f.split('/')[-1].split('.')[0] for f in wav_files]

    t0 = time.time()
    mel_folder = config['output_folder']
    mel_folder = os.path.expanduser(mel_folder)
    os.makedirs(mel_folder, exist_ok=True)
    n_workers = config['num_works']
    logger.info(f'using {n_workers} process(es)')
    if n_workers <= 1:
        wav_list_to_fbank(wav_files, names, mel_folder, feature_extractor)
    else:
        pool = multiprocessing.Pool(n_workers)
        dst_files = []
        # Collect multi-processing parameters
        params = [(file, name, mel_folder)
                  for file, name in zip(wav_files, names)]

        pool.map(wav_to_fbank_mp, params)
        pool.close()
        pool.join()

    time_used = time.time() - t0
    logger.info('done!')
    logger.info(f'processed {len(wav_files)} files in {int(time_used)} seconds')
