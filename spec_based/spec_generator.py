import random
import numpy as np
from generator import BaseGenerator
from vggish.mel_features_np import wavfile_to_examples


class SpecGenerator(BaseGenerator):
    def make_train_generator(self):
        return SpecGenerator.SpecInnerGenerator(self, self.train_indices, True)

    def make_val_generator(self):
        return SpecGenerator.SpecInnerGenerator(self, self.val_indices, False)

    def obtain_params_descriptor(self, spec_params):
        return str(spec_params.NUM_FRAMES) + 'x' + str(spec_params.NUM_BANDS) + 'x' + \
               str(spec_params.SAMPLE_RATE) + 'x' + str(spec_params.STFT_WINDOW_LENGTH_SECONDS) + 'x' + \
               str(spec_params.STFT_HOP_LENGTH_SECONDS) + 'x' + str(spec_params.MEL_MIN_HZ) + 'x' + \
               str(spec_params.MEL_MAX_HZ) + 'x' + str(spec_params.EXAMPLE_WINDOW_SECONDS) + 'x' + \
               str(spec_params.EXAMPLE_HOP_SECONDS)

    class SpecInnerGenerator(BaseGenerator.BaseInnerGenerator):
        def preproc_audio(self, specto):
            sampling_itvs = self.outer.sampling_itvs
            spec_params = self.outer.spec_params
            res = []
            for itvl in sampling_itvs:
                left_p, right_p, l_s = itvl
                # left_sec = left * specto.shape[0] * spec_params.STFT_HOP_LENGTH_SECONDS
                # right_sec = right * specto.shape[0] * spec_params.STFT_HOP_LENGTH_SECONDS
                l_hops = int(float(l_s) / spec_params.STFT_HOP_LENGTH_SECONDS)
                left = int(left_p * specto.shape[0])
                right = int(right_p * specto.shape[0])
                if right - left < l_hops:
                    left = max(0, left - int(l_hops - (right - left) / 2.))
                    right = min(specto.shape[0], left + l_hops)

                assert right - left >= l_hops

                left = random.randint(left, left + (right - left - l_hops))
                right = left + l_hops
                res.append(specto[left:right])
                assert right - left == l_hops

            return np.concatenate(res, axis=0)

        def read_audio(self, file_name):
            audio = wavfile_to_examples(file_name, self.outer.spec_params)
            audio = np.expand_dims(audio, axis=2)
            return audio
