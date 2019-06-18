# mic-toolbox
Toolbox for implementation of the Markov information criterion

## Contents

The toolbox contains 2 folders:
- `demo`: this contains two Jupyter notebooks demonstrating the steps required to compare two simulation models using the MIC. 
- `source`: this contains the source cython files and the `__init__.py` file for the package.

## Compilation instructions 

Before first use the source code must be compiled. This requires Cython and a C compiler, and the process is platform-dependent. More details are provided in the [Cython documentation on platform dependent installation](https://cython.readthedocs.io/en/latest/src/quickstart/install.html). In both cases, the toolbox is compiled from the command line, by navigating to the source folder and simply running:

`$ python setup.py`

Note: Compiling the toolbox will result in Cython throwing some warnings, all of which can be ignored.
- numpy/C type conversions (in Windows)
- signed/unsigned comparisons in `Tree.desc` and unitialised variables (in Linux) 
- In all cases [the standard but harmless](https://github.com/scipy/scipy/issues/5889) 'Warning: Using deprecated NumPy API' message. 

### Linux

Compilation on Linux is straightforward, as `gcc` is typically included with the distribution. All that is required is running the setup file as outlined above. Compilation was succesfully tested using Python 3.6 on Red Hat Enterprise Linux Server 7.6 (Maipo)

### Windows

Compiling on Windows is trickier as Windows does not come bundled with a C compiler. The Visual C++ compiler needs to be downloaded and installed. Furthermore, the correct C compiler has to be selected depending on the version of Python installed on your machine. More details are provided on the [Python wiki page on Windows C compilers](https://wiki.python.org/moin/WindowsCompilers). 

Two options are available from Python 3.5 onwards. Both of these contain the required Visual C++ v14.0 compiler:
- A full installation of Visual Studio 2017.
- A more limited installation of the Build Tools for Visual Studio 2017.

Compilation was succesfully tested using Python 3.6 on Windows 10 with the Build Tools for Visual Studio 2017.

### MacOS

Unfortunately, the toolbox has not yet been tested on MacOS, this will be carried out in future updates.

## To Use/import the toolbox:

Because the toolbox is still experimental. it is not yet provided as a full distributable package/wheel. This is one of the main objectives going forwards. For the moment, the recommended way to use the tools is simply to include a copy of the `mic` folder any project folder that requires the tools. This can either be the `source/mic` folder (which can be compiled in place prior to use) or a pre-compiled version. Once this is done, simply include the following import statement:
  
`import mic.toolbox` 
  
 or
  
 `import mic.toolbox as mt`
