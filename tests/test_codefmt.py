import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from codefmt import to_program, to_target  # noqa: E402
from execute import answers_equal, majority_vote, run_programs  # noqa: E402

PROGRAMS = [
    "def solve() -> int:\n    total_time = 5.5 #hours\n    d = 10 # miles\n    result = d / total_time\n\n"
    "    return result\n",
    "def solve() -> int:\n    s = '#notacomment'\n    x = 0\n    for i in range(3):\n        # loop\n        x += i\n"
    "    if x > 2:\n        x = x * 2\n    result = x + len(s)\n\n    return result\n",
    "def solve() -> int:\n    import math\n    result = math.sqrt(16)\n    return result\n",
]


def test_roundtrip():
    for code in PROGRAMS:
        target = to_target(code)
        assert target is not None
        assert not target.startswith("def") and "return result" not in target
        assert "#" not in target.replace("'#notacomment'", "")
        a, b = run_programs([code, to_program(target)])
        assert a is not None and answers_equal(b, a)


def test_other_shapes_are_rejected():
    assert to_target("def solve():\n    return 3\n") is None
    assert to_target("x = 1\n") is None


def test_sandbox():
    codes = ["def solve():\n    return 10 ** 10 ** 10", "def solve():\n    while True: pass",
             "import os\ndef solve():\n    os._exit(1)", "def solve():\n    return '3.5'", "def solve(:\n"]
    assert run_programs(codes, timeout=0.5) == [None, None, None, 3.5, None]


def test_tolerance_and_vote():
    assert answers_equal(1.0000005, 1.0) and not answers_equal(1.00001, 1.0)
    assert answers_equal(1e6 + 0.5, 1e6) and not answers_equal(1e6 + 5, 1e6)
    assert majority_vote(["g", "a", "b", "c"], [1.0, 2.0, 2.0, None]) == ("a", 2.0)
    assert majority_vote(["g", "a"], [1.0, 2.0]) == ("g", 1.0)
    assert majority_vote(["g", "a"], [None, None]) == ("g", None)


if __name__ == "__main__":
    test_roundtrip()
    test_other_shapes_are_rejected()
    test_sandbox()
    test_tolerance_and_vote()
    print("ok")
