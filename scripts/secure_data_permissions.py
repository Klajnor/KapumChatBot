"""Выставление прав данных через дескрипторы без перехода по ссылкам (Linux)."""

from __future__ import annotations

import argparse
import errno
import os
import stat


def repair(path: str, uid: int, gid: int, *, parent_fd: int | None = None) -> None:
    """Обработать дерево, пропуская ссылки, специальные файлы и hardlink-файлы."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags, dir_fd=parent_fd)
    except OSError as exc:
        # Ссылка, исчезнувший файл или недоступный объект не разыменовываются.
        if exc.errno in {errno.ENOENT, errno.ELOOP, errno.ENXIO}:
            return
        raise
    try:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            os.fchown(fd, uid, gid)
            os.fchmod(fd, 0o700)
            for name in os.listdir(fd):
                repair(name, uid, gid, parent_fd=fd)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            os.fchown(fd, uid, gid)
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def main() -> None:
    """Применить права к переданным путям для указанного пользователя."""
    import pwd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    user = pwd.getpwnam(args.user)
    for path in args.paths:
        repair(path, user.pw_uid, user.pw_gid)


if __name__ == "__main__":
    main()
