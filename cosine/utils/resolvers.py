import hydra
from omegaconf import OmegaConf


def get_vocab_size(vocab_cfg):
    vocab = hydra.utils.instantiate(vocab_cfg)
    return len(vocab)


def register_custom_resolvers():
    """Register all custom OmegaConf resolvers"""
    OmegaConf.register_new_resolver("get_vocab_size", get_vocab_size)
