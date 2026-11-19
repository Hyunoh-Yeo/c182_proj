from .common import Defense
from .defensivetoken import load_defensivetoken
from .secpo import load_secpo
from .undefended import load_undefended


def load(name: str, **kwargs) -> Defense:
    """load("undefended") | load("secpo", checkpoint=...) | load("defensivetoken", embeddings_path=...)"""
    loaders = {
        "undefended": load_undefended,
        "secpo": load_secpo,
        "defensivetoken": load_defensivetoken,
    }
    return loaders[name](**kwargs)
