#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

from distutils.core import setup
import glob
import os
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools import find_packages
import tempfile
import zipfile

MODELS_DIR = "diarizer/models"
MODELS = ["ResNet101_8kHz"]


def install_scripts(directory):
    """Call cmd commands to install extra software/repositories.

    Args:
        directory (str): path
    """
    # unpack multiple zip files into .pth file
    for model in MODELS:
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        nnet_dir = os.path.join(MODELS_DIR, model, "nnet")
        assert os.path.isdir(nnet_dir), f"{nnet_dir} does not exist."

        for zip_part in sorted(
            glob.glob(f'{os.path.join(nnet_dir, "*.pth.zip.part*")}')
        ):
            with open(zip_part, "rb") as f:
                temp_zip.write(f.read())

        with zipfile.ZipFile(temp_zip, "r") as fzip:
            fzip.printdir()
            fzip.extractall(path=nnet_dir)


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        self.execute(
            install_scripts, (self.egg_path,), msg="Running post install scripts"
        )


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        self.execute(
            install_scripts, (self.install_lib,), msg="Running post install scripts"
        )


setup(
    name="diarizer",
    version="0.1.1",
    packages=find_packages(),
    url="https://github.com/desh2608/diarizer",
    install_requires=[
        "numpy>=1.19.5",
        "scipy==1.4.1",
        "numexpr==2.6.9",
        "h5py==2.9.0",
        "fastcluster==1.2.4",
        "onnxruntime==1.4.0",
        "soundfile==0.10.2",
        "torch==1.10.0",
        "numba==0.53.0",
        "kaldi_io",
        "tabulate>=0.8.6",
        "intervaltree",
        "spy-der==0.2.0",
        "scikit-learn @ git+https://github.com/desh2608/scikit-learn.git@overlap",
        "pyannote.audio @ git+https://github.com/desh2608/pyannote-audio.git@develop",
    ],
    license="Apache License, Version 2.0",
    cmdclass={"install": PostInstallCommand, "develop": PostDevelopCommand},
)
