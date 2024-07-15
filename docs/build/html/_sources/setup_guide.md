Setup Guide
===========

Introduction
------------

This setup guide provides instructions for setting up and configuring the project.

Cloning the Repository
----------------------

To clone the project repository, you need to have access to the GitHub repository. The repository 
URL can be found by clicking on the GitHub logo at the top of this documentation. If you don't 
have access tothe repository, please contact one of the maintainers or administrators of the 
codebase.

To clone the repository, run the following command in the terminal:

```bash
git clone git@github.com:ajcost/duke-msec-bootcamp.git
```

If you encounter an error, make sure you have set up your SSH keys correctly. Please do not use 
HTTPS to clone.

GitHub website [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

For instructions on checking for existing keys please refer to the GitHub documentation [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys).
For instructions on generating SSH keys, refer to the GitHub documentation [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).


Environment
-------------

It is recommended to use a Conda environment for building the project. While it is possible
 to build the project in the base Python environment, this is not recommended as it can cause
 package conflicts and resolution issues. To ensure a smooth build process, it is recommended to
 install Miniconda and create a Conda environment using the `environment.yml` file located in the
 base directory of the repository.

 To build the environment run the following command from your terminal:

 ```bash
conda env create -f environment.yml
 ```

The environment can then be activated using the following conda command:
```bash
conda activate duke-msec-bootcamp
```

If the environment ever needs to be updated due to changes in the `environment.yml` configuration
file. The following command can be used to rebuild the `duke-msec-bootcamp` conda environment locally
without much issues:

```bash
conda activate duke-msec-bootcamp
conda env update --file environment.yml --prune
```

:::{note}
:class: a-tip-reference

The following guide is specific to Linux/Mac or any other Unix based operating systems. Windows
users will need to make adjustments accordingly.
:::

Windows
-------------

Windows machines can be different from Unix-based operating systems like Mac and Linux. As a result
, a few extra steps need to be taken during the setup process. It is also important to be cautious
when working with different terminals, as the Windows CMD prompt and Powershell may not recognize
the Anaconda installation of Python. This guide does not provide detailed instructions, as there
are already well-documented setup guides available. Instead, it will direct you to those resources
for further information.

#### Anaconda

Anaconda is a popular distribution of the Python programming language that comes with a package
manager and a collection of pre-installed libraries and tools. It is recommended to use Anaconda for
 managing the Python environment when working with the SWITCH-GT project.

To download Anaconda, visit the official website [here](https://www.anaconda.com/products/individual)
 and click on the "Download" button. Follow the installation instructions to set up Anaconda on your
  Windows machine.

For detailed setup guides and documentation, you can refer to the following resources:
- [Anaconda Documentation](https://docs.anaconda.com/)
- [Anaconda User Guide](https://docs.anaconda.com/anaconda/user-guide/)

For installation on Windows please reference:
- [Anaconda Windows Setup](https://docs.anaconda.com/free/anaconda/install/windows/)

These resources provide comprehensive information on how to install Anaconda, create and manage
environments, install packages, and more.


#### GitBash

GitBash is a command-line interface tool that provides a Unix-like environment on Windows machines.
It allows users to run Git commands and execute shell scripts. GitBash is commonly used by
developers who work with Git repositories on Windows.

To download GitBash, visit the official website [here](https://gitforwindows.org/) and click on the
"Download" button. Follow the installation instructions to set up GitBash on your Windows machine.

For detailed setup guides and documentation, you can refer to the following resources:
- [GitBash Documentation](https://gitforwindows.org/documentation/)
- [Git Documentation](https://git-scm.com/doc)

These resources provide comprehensive information on how to use GitBash, including command
references, configuration options, and troubleshooting guides.

Configuration
-------------

For developers and modelers one can use their `.env` file to ensure the proper configuration of
their environment. If this file is written correctly the connection, upload and download of data
should be relatively automated. The user need not worry about too much memorization and
credentialization of resources. That should be done once. The `.env` file should be located
in the base of the repository.

Your environment file should look something like this:
```bash
# Path variables - change these to match your system
REPOSITORY_PATH=/path/to/repository/duke-msec-bootcamp
LOCAL_DATA_PATH=/path/to/repository/duke-msec-bootcamp/data
JUPYTER_PATH=/path/to/repository/duke-msec-bootcamp/notebooks
SOURCE_PATH=/path/to/repository/duke-msec-bootcamp/src
```

:::{Attention}
:class: a-tip-reference

You should never commit your `.env` file to a branch or push it to a GitHub remote repository under
any circumstances. By default, the `.gitignore` file blocks this file to prevent accidental commits.
To stage changes in the `.env` file, a force add (`git add -f`) is required, but this should be
avoided since the file contains sensitive information, including access keys. The `.env` file in the
 remote repository should remain illustrative only.
:::

##### Loading the environment varibles

To load `.env` variables into your shell session on Unix-like systems (Linux, macOS), first make
sure your `.env` file is correctly formatted with `KEY=VALUE` pairs on separate lines. Open a
terminal, navigate to the directory containing the `.env` file, and execute the following commands:
`set -a` to automatically export all variables, `source .env` to load the variables, and `set +a`
to stop the automatic export. This sequence ensures that all variables in the `.env` file are loaded
 into the environment of your current shell session.

 ```bash
set -a
source .env
set +a
 ```

 To check if this has worked you can run:

 ```bash
echo $REPOSITORY_PATH
 ```

For Windows systems, Windows does not natively support the `source` command used in Unix-like
systems. Instead, create a batch script to load the variables. Open a text editor and write a script
 with the following content: `@echo off` to prevent the command prompt from displaying commands,
 and `FOR /F "tokens=*" %%G IN (.env) DO SET %%G` to parse each line in the `.env` file and set the
 environment variables accordingly. Save this script as `load_env.bat` in the same directory as your
  `.env` file. Execute `load_env.bat` in the Command Prompt to apply the environment variables to
  your session. This method adapts the process to the Windows environment by using a batch file to
  replicate the functionality of the Unix `source` command.

```bash
@echo off
FOR /F "tokens=*" %%G IN (.env) DO SET %%G
```

 To check if this has worked you can run:

 ```bash
echo %REPOSITORY_PATH%
 ```

Clearing and Doing a Reset
-------------

Sometimes, you may encounter package incompatibility issues, outdated packages, or other errors
within the conda environment. In such cases, it may be necessary to rebuild the environment.
To do this, follow these steps:

Deactivate the current conda environment by running the command:

```bash
conda deactivate
```

Remove the existing conda environment by running the command:

```bash
conda remove -n duke-msec-bootcamp --all
```

Rebuild the environment from the `environment.yml` file using the following command:

```bash
conda env create -f environment.yml
```

Building the Sphinx Documentation
-------------

Building the Sphinx documentation is straightforward. Enter the `docs` directory and
run:

```bash
make clean
make html
```
