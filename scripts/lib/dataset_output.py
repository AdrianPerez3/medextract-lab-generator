"""Publish generated datasets and splits without exposing partial outputs."""
from contextlib import contextmanager
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory


@contextmanager
def staged_output(destination: Path):
    """Yield a private staging area, then atomically publish to a new or empty directory."""
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and
            (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError(f"Output must be a new or empty directory: {destination}")
    mode = S_IMODE(destination.stat().st_mode) if destination.exists() else None
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        # A normal child inherits the caller's umask/group setup. Its private
        # parent protects the staging contents until the final same-volume rename.
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        yield staging
        if mode is not None:
            staging.chmod(mode)
        staging.rename(destination)
