from spec_based.spec_generator import SpecGenerator
from train_with_generator_slow import train_with_generator
from vggish.vggish_arch import create_vggish

train_with_generator(create_vggish, SpecGenerator)
