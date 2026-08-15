from importlib import import_module


__all__ = ["ImageProcessor", "Downloader", "ImageUploader"]

_EXPORTS = {
    "ImageProcessor": (".image_processor", "ImageProcessor"),
    "Downloader": (".downloader", "Downloader"),
    "ImageUploader": (".image_uploader", "ImageUploader"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
