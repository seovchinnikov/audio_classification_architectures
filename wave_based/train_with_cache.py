from train_with_caching import train_with_cache
from vggish.vggish_arch import create_vggish
from wave_based.wave_arch import create_wave_arch
from wave_based.wave_generator import WaveGenerator

train_with_cache(create_wave_arch, WaveGenerator)
