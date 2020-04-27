import setuptools

setuptools.setup(
    name="deeparc2normalize",
    version="0.0.1",
    author="Vision and Learning lab",
    author_email="allist@vistec.ac.th",
    description="normalize deeparc camera to plane [0,0]",
    url="https://github.com/pureexe/deeparc2normalize",
    packages=[''],
    py_modules=['deeparc2normalize'],
    install_requires=[
          'numpy',
          'torch'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
     'console_scripts': ['deeparc2normalize=deeparc2normalize:entry_point'],
    },
    python_requires='>=3.6'
)