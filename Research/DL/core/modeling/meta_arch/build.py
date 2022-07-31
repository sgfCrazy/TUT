from Research.DL.core.utils.registry import Registry


META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
    注册 meta-architectures
    通过调用obj(cfg)注册, 返回一个 nn.Module 对象
"""


def build_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)
