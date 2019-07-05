# mic-toolbox
Toolbox for implementation of the Markov information criterion

This readme covers contents of the toolbox, compilation and import instructions. Documentation is provided in all classes and functions, and demonstration files are provided to illustrate how to implement the tools. For further details on the implementation of the CTW/CTM algorithms and the theoretical properties of the MIC, please refer to the following documents:
- The appendix of ["A practical, accurate, information criterion for nth order markov processes".](https://link.springer.com/article/10.1007/s10614-016-9617-9)
- Sections 2 and 3 of ["Macroeconomic simulation comparison with a multivariate extension of the Markov Information Criterion".](https://www.kent.ac.uk/economics/documents/research/papers/2019/1908.pdf)

## Contents

The toolbox contains 2 folders:
- `demo`: this contains two Jupyter notebooks demonstrating the steps required to compare two simulation models using the MIC.
- `source`: this contains the source cython files, the `__init__.py` file for the package and the `setup.py` script for compiling the toolbox and creating the wheel distribution.

## Toolbox installation and import instructions:

Thanks to @embray's work on the `setup.py` script, the toolbox is now available as a set of distributable wheels and can be installed using `pip` on the github release. To install the toolbox, run one of the following commands:

`$ pip install https://github.com/Sylvain-Barde/mic-toolbox/releases/download/v0.1.0a1/mic_toolbox-0.1.0a1-cp36-cp36m-linux_x86_64.whl`

`$ pip install https://github.com/Sylvain-Barde/mic-toolbox/releases/download/v0.1.0a1/mic_toolbox-0.1.0a1-cp36-cp36m-win_amd64.whl`

`$ pip install https://github.com/Sylvain-Barde/mic-toolbox/releases/download/v0.1.0a1/mic_toolbox-0.1.0a1-cp37-cp37m-win_amd64.whl`

Note, because the toolbox uses cython and needs to be compiled, it is platform and python version dependent, therefore you should use the link corresponding to your installation. Once the toolbox is installed, simply include the following import statement in your code:

`import mic.toolbox`

or

 `import mic.toolbox as mt`

The `demo` folder contains two examples of how to use the toolbox to get the MIC score of a model on some empirical data.

To uninstall the toolbox, run:

`$ pip uninstall mic-toolbox`

## Compilation instructions

In order to use the toolbox from the source code it must first be compiled. This requires Numpy, Cython and a C compiler, and the process is platform-dependent. More details are provided in the [Cython documentation on platform dependent installation](https://cython.readthedocs.io/en/latest/src/quickstart/install.html). To create wheel packages for the release, navigate to the source and run either:

`$ pip wheel .`

or

`$ python setup.py bdist_wheel`

The first option places the wheel in the source folder, the second in a dedicated `source/dist` subfolder.

In cases where the user is not allowed to install packages, for example on a cluster, it is possible to compile the extensions for local use. Navigate to the source folder and run:

`$ python setup.py build_ext`

This places the cython extensions in the source/mic subfolder, which can then be used locally with the same `import` commands.

Note: Compiling the toolbox will result in Cython throwing some warnings, all of which can be ignored.
- numpy/C type conversions (in Windows)
- signed/unsigned comparisons in `Tree.desc` and uninitialised variables (in Linux)
- In all cases [the standard but harmless](https://github.com/scipy/scipy/issues/5889) 'Warning: Using deprecated NumPy API' message.

### Linux

Compilation on Linux is straightforward, as `gcc` is typically included with the distribution. All that is required is running the setup file as outlined above. Compilation was successfully tested using Python 3.6 on Red Hat Enterprise Linux Server 7.6 (Maipo)

### Windows

Compiling on Windows is trickier as Windows does not come bundled with a C compiler. The Visual C++ compiler needs to be downloaded and installed. Furthermore, the correct C compiler has to be selected depending on the version of Python installed on your machine. More details are provided on the [Python wiki page on Windows C compilers](https://wiki.python.org/moin/WindowsCompilers).

Two options are available from Python 3.5 onwards. Both of these contain the required Visual C++ v14.0 compiler:
- A full installation of Visual Studio 2017.
- A more limited installation of the Build Tools for Visual Studio 2017.

Compilation was successfully tested using Python 3.6 and 3.7 on Windows 10, both using the Build Tools for Visual Studio 2017.

### MacOS

Unfortunately, the toolbox has not yet been tested on MacOS, this will be carried out in future updates.
