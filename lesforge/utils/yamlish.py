"""Minimal YAML-subset parser (zero-dependency fallback).

Supports what our configs actually use:
    * nested maps (``key: value``)
    * block lists (``- item`` and ``- key: value``)
    * inline comments (``key: value  # note``)
    * blank lines

This is intentionally tiny -- it is NOT a full YAML parser. It exists only so
the package still reads its ``.yaml`` configs when PyYAML is absent.
"""


def _strip_comment(line):
    # inline comment: a ' #' preceded by whitespace (avoid stripping bare '#' values)
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _indent_of(line):
    return len(line) - len(line.lstrip(" "))


def _parse_scalar(s):
    s = s.strip()
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "~", ""):
        return None
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    # inline list [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",")]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse(text):
    """Parse YAML-subset text into nested dict/list structures.

    Stack entries are 5-tuples ``(indent, container, parent, key, pending)``:

    * ``container`` is the dict/list we are currently filling;
    * ``parent`` is the container that holds ``container`` (so we can *replace*
      a placeholder dict with a list when a block list arrives);
    * ``key`` is the key under which ``container`` sits in ``parent``;
    * ``pending`` marks a placeholder dict created by ``key:`` (empty value)
      that may still turn into either a nested map or a list.

    The two subtle rules (historically the source of "lost first list item" bugs):
    1. a block list ``- item`` under ``key:`` must replace the placeholder dict
       in its *parent* (not nest a list inside the placeholder);
    2. sibling ``- item`` lines at the same indent belong to the same list, so a
       list item only pops entries *deeper* than itself, never an equal-indent list.
    """
    root = {}
    stack = [(-1, root, None, None, False)]

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        line = _strip_comment(raw).rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        indent = _indent_of(line)

        if stripped.startswith("- "):
            item_text = stripped[2:].strip()
            # pop only *deeper* blocks; keep an equal-indent list (sibling item)
            while len(stack) > 1 and stack[-1][0] > indent:
                stack.pop()
            top_indent, container, parent, key, pending = stack[-1]

            if pending and isinstance(container, dict) and not container:
                # placeholder dict -> replace with a list in its parent
                lst = []
                if isinstance(parent, dict):
                    parent[key] = lst
                elif isinstance(parent, list):
                    parent.append(lst)
                stack.pop()
                stack.append((indent, lst, parent, key, False))
                container = lst
            elif not isinstance(container, list):
                # no list context yet; fall back to appending into a fresh list
                lst = []
                container = lst
                stack.append((indent, lst, None, None, False))

            if ":" in item_text and not item_text.startswith(("[", "{", '"', "'")):
                k, _, v = item_text.partition(":")
                d = {}
                container.append(d)
                if v.strip():
                    d[k.strip()] = _parse_scalar(v)
                else:
                    stack.append((indent, d, container, k.strip(), True))
            else:
                container.append(_parse_scalar(item_text))
        elif ":" in stripped:
            k, _, v = stripped.partition(":")
            k = k.strip()
            # sibling keys at equal/deeper indent finish the previous key's subtree
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            container = stack[-1][1]
            if v.strip():
                container[k] = _parse_scalar(v)
            else:
                d = {}
                container[k] = d
                stack.append((indent, d, container, k, True))
        else:
            if isinstance(stack[-1][1], list):
                stack[-1][1].append(_parse_scalar(stripped))

    return root
