import os
import glob
from pathlib import Path
import re
import wandb # key : db3dfb28050584618b561f9040b7fb3456a5e1f9
# Appending key for api.wandb.ai to your netrc file: /opt/ml/.netrc


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]  
        i = [int(m.groups()[0]) for m in matches if m]  
        n = max(i) + 1 if i else 2
        return f"{path}{n}"