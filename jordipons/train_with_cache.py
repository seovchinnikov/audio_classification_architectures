from spec_based.spec_generator import SpecGenerator
from train_with_caching import train_with_cache
from jordipons.jordi_arch import create_jordi_pons

train_with_cache(create_jordi_pons, SpecGenerator)
