from .common import Defense
from .defensivetoken import load_defensivetoken
from .delims import load_delims
from .secpo import load_secpo
from .undefended import load_undefended


def load(name: str, **kwargs) -> Defense:
    """load("undefended") | load("secpo", checkpoint=...) | load("defensivetoken", embeddings_path=...)
    | load("delims", checkpoint="outputs/.../delims_stepN.pt")"""
    loaders = {
        "undefended": load_undefended,
        "secpo": load_secpo,
        "defensivetoken": load_defensivetoken,
        "delims": load_delims,
    }
    return loaders[name](**kwargs)
