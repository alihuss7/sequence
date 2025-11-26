from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read dependencies from environment.yml
with open(os.path.join(here, 'environment.yml'), encoding='utf-8') as f:
    # We need to parse without yaml parser, since this runs before any packages are installed
    install_requires = []
    started = False
    for line in f:
        if started:
            install_requires.append(line.strip().replace('- ', ''))
        if 'pip:' in line:
            started = True
    if not install_requires:
        raise ValueError(f'Error parsing pip dependencies from environment.yml')

setup(
    name='nanomelt',
    version='1.3.0',
    description='A nanobody thermostability prediction tool',
    licence='CC BY-NC-SA 4.0',
    author='Aubin Ramon',
    author_email='ar2033@cam.ac.uk',
    long_description='See READ.me in GitLab repository https://gitlab.developers.cam.ac.uk/ch/sormanni/nanomelt.git',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nanomelt=nanomelt.__main__:main'
        ]
    },
    include_package_data=True,
    package_data={'': ["model/saved_models/NanoMelt_finalmodel/*"]},
    install_requires=install_requires
)
