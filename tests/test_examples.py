import pytest
from glob import glob
from subprocess import run
import os
import re

USAGE_RE = re.compile(r"^# Usage: ([^\n]*)$", flags=re.MULTILINE)
EXAMPLES = glob("examples/*.py") + glob("examples/**/*.py")


@pytest.mark.parametrize("example_path", EXAMPLES)
def test_run_trainer(example_path):
    with open(example_path) as f:
        content = f.read()
    usages = USAGE_RE.findall(content)
    assert len(usages) > 0, f"No usage examples in {example_path}"
    for usage in usages:
        print("Running:", usage)
        run(usage, shell=True, check=True)
