"""Compact target format.

All reference programs have the form `def solve() -> int:` + body + `return result`. The decoder is trained
to generate only the body without comments, blank lines and the function-level indentation; the program
is restored deterministically. This shortens targets from ~72 to ~58 BPE tokens on average.
"""
import io
import tokenize

HEADER = "def solve():"
RETURN = "return result"


def _strip_comments(code):
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT]
    return tokenize.untokenize(tokens)


def to_target(code):
    """Reference program -> body of solve(); None for programs of a different shape."""
    try:
        if "#" in code:
            code = _strip_comments(code)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    lines = [line.rstrip() for line in code.split("\n") if line.strip()]
    if len(lines) < 3 or not lines[0].startswith("def solve(") or lines[-1].strip() != RETURN:
        return None
    body = lines[1:-1]
    if not all(line.startswith("    ") for line in body):
        return None
    return "\n".join(line[4:] for line in body)


def to_program(target):
    """Generated body -> runnable program."""
    body = "\n".join("    " + line for line in target.split("\n"))
    return f"{HEADER}\n{body}\n    {RETURN}\n"
