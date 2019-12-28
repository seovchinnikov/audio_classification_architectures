from spec_based.spec_generator import SpecGenerator
from train_with_caching import train_with_cache
from vggish.vggish_arch import create_vggish

train_with_cache(create_vggish, SpecGenerator)
