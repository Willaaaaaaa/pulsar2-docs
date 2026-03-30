import re


def sanitize(s):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)
