from train_with_caching import train_with_cache
from train_with_generator_slow import train_with_generator
from vggish.vggish_arch import create_vggish
from wave_based.wave_arch import create_wave_arch
from wave_based.wave_generator import WaveGenerator

train_with_generator(create_wave_arch, WaveGenerator)
