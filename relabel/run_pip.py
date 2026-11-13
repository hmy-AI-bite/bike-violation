"""绕开 tempfile.mkdtemp 在受限环境下的权限问题后调用 pip。

mkdtemp 建目录用 0o700，在当前文件系统策略下 pip 进程写不进去，
会报 "Permission denied: ... pip-unpack-xxxx/*.whl"。这里改用 os.makedirs 建临时目录。
"""
import os
import random
import string
import sys
import tempfile


def _mkdtemp(suffix="", prefix="tmp", dir=None):
    base = dir or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    for _ in range(10000):
        name = prefix + "".join(random.choices(string.ascii_letters + string.digits, k=10)) + suffix
        path = os.path.join(base, name)
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("无法创建临时目录")


tempfile.mkdtemp = _mkdtemp

from pip._internal.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
