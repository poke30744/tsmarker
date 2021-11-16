import setuptools, os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

build_number = os.getenv('BUILD_NUMBER') or '0'

setuptools.setup(
    name="tsmarker",
    version=f"0.1.{build_number}",
    author="poke30744",
    author_email="poke30744@gmail.com",
    description="Mark MEPG TS clips by program/CM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/poke30744/tsmarker/blob/main/README.md",
    packages=setuptools.find_packages(exclude=['tests',]),
    install_requires=[
        'tscutter',
        'sklearn',
        'opencv-python',
        'pandas',
        'pysubs2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
