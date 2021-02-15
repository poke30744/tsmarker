import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsmarker",
    version="0.0.1",
    author="poke30744",
    author_email="poke30744@gmail.com",
    description="Mark MEPG TS clips by program/CM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
     install_requires=[
        'tsmarker',
        'sklearn',
        'opencv-python'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
