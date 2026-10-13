"""Publish encoded JPEG and optional EPS; never handle rendering failures here."""
import errno
import os
from pathlib import Path


RECOVERABLE_ERRNOS = frozenset((errno.EACCES, errno.ENOSPC, errno.EIO))


def write_payload(stream, payload):
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())


def eps_output_enabled():
    value = os.environ.get('ASTR_INSITU_WRITE_EPS', '1')
    if value not in ('0', '1'):
        raise ValueError('ASTR_INSITU_WRITE_EPS must be 0 or 1')
    return value == '1'


def publish_pair(picture, jpeg, eps=None):
    picture = Path(picture)
    destinations = (picture,) if eps is None else (picture, picture.with_suffix('.eps'))
    payloads = (jpeg,) if eps is None else (jpeg, eps)
    temporary = tuple(path.with_name(path.name + '.partial') for path in destinations)
    if any(path.exists() for path in (*destinations, *temporary)):
        raise FileExistsError('Refusing to overwrite an existing image or partial image')
    owned = []
    phase = ''
    try:
        for path, payload in zip(temporary, payloads):
            phase = 'stage_' + path.suffixes[-2][1:]
            # Track only files created by this writer, including a partial write.
            with path.open('xb') as stream:
                owned.append(path)
                write_payload(stream, payload)
        for path, destination in zip(temporary, destinations):
            phase = 'publish_' + destination.suffix[1:]
            os.link(path, destination)
            owned.append(destination)
        for path in temporary:
            path.unlink()
            owned.remove(path)
    except BaseException as exc:
        # A failed publication has no usable final image. Cleanup failures are fatal.
        for path in reversed(owned):
            path.unlink(missing_ok=True)
        if isinstance(exc, OSError) and exc.errno in RECOVERABLE_ERRNOS:
            return {'errno': exc.errno, 'phase': phase, 'reason': os.strerror(exc.errno)}
        raise
    return None
