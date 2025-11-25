from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

# Pull only the *pip* dependencies from environment.yml (as you already do)
with open(os.path.join(here, "environment.yml"), encoding="utf-8") as f:
    install_requires = []
    started = False
    for line in f:
        if started:
            dep = line.strip().replace("- ", "")
            if dep:
                install_requires.append(dep)
        if "pip:" in line:
            started = True
    if not install_requires:
        raise ValueError("Error parsing pip dependencies from environment.yml")

# With PEP 621 metadata in pyproject.toml, we only need to set the dynamic field(s):
setup(
    install_requires=install_requires,
)
