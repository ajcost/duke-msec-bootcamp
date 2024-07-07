Setup Guide
===========

Introduction
------------

This setup guide provides instructions for setting up and configuring the SWITCH-GT project. It is intended for
developers or users who want to use the codebase or modify it.

Cloning the Repository
----------------------

To clone the project repository, you need to have access to the GitHub repository. The repository URL can be found by
clicking on the GitHub logo at the top of this documentation. If you don't have access to the repository, please contact
one of the maintainers or administrators of the codebase. Alternatively, if you don't have access to BCG GitHub, please
reach out to the BCG X Platform Team and request access.

To clone the repository, run the following command in the terminal:

```bash
git clone git@github.com:bcgx-pi-bhi-analytics/CCSS_XIWB.git
```

If you encounter an error, make sure you have set up your SSH keys correctly. Please do not use HTTPS to clone, as
this is forbidden by the BCG GitHub. In order to setup these keys correctly you can follow the instructions on the
GitHub website [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

For instructions on checking for existing keys please refer to the GitHub documentation [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys).
For instructions on generating SSH keys, refer to the GitHub documentation [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).


Environment
-------------

It is recommended to use a Conda environment for building the SWITCH-GT project. While it is possible
 to build the project in the base Python environment, this is not recommended as it can cause
 package conflicts and resolution issues. To ensure a smooth build process, it is recommended to
 install Miniconda and create a Conda environment using the `environment.yml` file located in the
 base directory of the repository.

 To build the SWITCH-GT environment run the following command from your terminal:

 ```bash
conda env create -f environment.yml
 ```

The environment can then be activated using the following conda command:
```bash
conda activate ccss_xiwb
```

If the environment ever needs to be updated due to changes in the `environment.yml` configuration
file. The following command can be used to rebuild the `ccss_xiwb` conda environment locally
without much issues:

```bash
conda activate ccss_xiwb
conda env update --file environment.yml --prune
```

:::{note}
:class: a-tip-reference

If contributions require new packages. They need to be added to the `environment.yml`,
and subsequently called out in the merge or release of a new version.
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
# Plotly Dash configuration
APP_HOST=127.0.0.1
APP_PORT=8050
APP_DEBUG=True
APP_MODE=local

# Path variables
REPOSITORY_PATH=/absolute/path/to/repository # should be the path to the CCSS SWITCH-GT repository
SRC_PATH=/absolute/path/to/source/code # path to the source code for SWITCH-GT within the base environment
LOCAL_DATA_PATH=/absolute/path/to/local/data/storage # a data path of your choosing to store temporary data files
JUPYTER_PATH=/absolute/path/to/notebooks # should be within the repository labeled as 'notebooks'

# Source code python path


# Amazon Neptune configuration
AMAZON_NEPTUNE_DB_NAME=amra-prod-bhiindustry-15221-88-neptunedb
AMAZON_NEPTUNE_CLUSTER_ID=cluster-ro-cuhbjy64fd7p
AMAZON_NEPTUNE_PORT=8182
AMAZON_NEPTUNE_REGION=us-east-1

# PostgresSQL configuration

# Amazon S3 configuration
AMAZON_S3_BUCKET_NAME=amra-prod-bhiindustry-15221-88
AMAZON_S3_REGION=us-east-1

# Amazon credentials configuration
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/CCSS_XIWB.log
```

To manage the Plotly Dash configuration smoothly between development and production environments,
you can leverage environment-specific settings that adjust critical parameters like `DEBUG` mode and
`APP_PORT`. For instance, in development, you might enable `DEBUG` mode and use a local port, while
in production, you would disable `DEBUG` mode and switch to a secure port with HTTPS enabled.

The `REPOSITORY_PATH` variable defines the absolute path to the CCSS SWITCH-GT repository, ensuring
the repository's location is clearly specified. Similarly, `SRC_PATH` indicates the path to the
source code for SWITCH-GT within the base environment. `LOCAL_DATA_PATH` is designated for storing
temporary data files at a user-specified location. Lastly, `JUPYTER_PATH` identifies the path to the
notebooks directory, which must be located within the repository.

The Amazon Neptune variables ensure the proper configuration of a connection to the our
Amazon Neptune graph database. Similarly the Amazon S3 configuration ensure the establishment
of a connection to the S3 resource.

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

Graph Notebook Setup
-------------

The following is pulled from the AWS Graph notebook setup guide with a few modifications from the
ease enabled by conda environments and the configuration files.

You can installing graph-notebook and its prerequisites using the following command. However, this
is not needed if you have built the `ccss_xiwb` conda environment.

```bash
pip install graph-notebook
```

##### Jupyter Classic Notebook

For the Jupyter Classic Notebook, or Jupyter Notebooks, the following steps are required to enable
some extensions that we have installed. The first command enables the `graph_notebook.widgets`
extension and the second command installs the necessary static resources and notebook extensions.

```bash
jupyter nbextension enable  --py --sys-prefix graph_notebook.widgets
python -m graph_notebook.static_resources.install
python -m graph_notebook.nbextensions.install
```

We then have to set up the notebooks tree structure and configuration file. It is recommended to use
 the `JUPYTER_PATH` configured in your `.env` file for this purpose. This ensures that you don't
 have random notebook folders floating around on your local computer. Keeping it within the
 repository enables easy location and easy commits to the codebase. Although the Style Guide
 suggests that Jupyter Notebooks should not be committed to the repository, it can be done under
 limited circumstances.

To do this, and copy some premade starter notebooks to your workspace run the following command:

```
python -m graph_notebook.notebooks.install --destination $JUPYTER_PATH
```

:::{tip}
:class: a-tip-reference

This command has assumed that you have gone through the steps to load the environment varibles
into your terminal session. If you wish to use a path you specify, merely enter that path
where `$JUPYTER_PATH` is listed.
:::

:::{note}
:class: a-tip-reference

The above command is for unix systems specifically if you are working on Windows the environment
variables are referenced slightly differently. You can use the following command to accomplish
the same.

```bash
python -m graph_notebook.notebooks.install --destination %JUPYTER_PATH%
```
:::

This following code snippet creates a `nbconfig` file and directory tree if they do not already exist.
The `nbconfig` file is used to configure the behavior of Jupyter notebooks. First, the code checks
if the `~/.jupyter/nbconfig` directory exists. If it doesn't, the `mkdir` command is used to create
the directory.

Next, the code checks if the `~/.jupyter/nbconfig/notebook.json` file exists. If it doesn't, the
`touch` command is used to create an empty `notebook.json` file inside the `nbconfig` directory.

By running this code, you ensure that the necessary directory and file structure is in place for
configuring Jupyter notebooks. The `~` symbol represents the user's home directory, so the
`nbconfig` directory and `notebook.json` file will be created in the user's home directory.

```bash
mkdir ~/.jupyter/nbconfig
touch ~/.jupyter/nbconfig/notebook.json
```

:::{note}
:class: a-tip-reference

On windows the equivalent commands here are:
```bash
mkdir %USERPROFILE%\.jupyter\nbconfig
echo {} > %USERPROFILE%\.jupyter\nbconfig\notebook.json
```
:::

To finally start the Jupyter notebook one just runs the following command from their terminal:

```bash
python -m graph_notebook.start_notebook --notebooks-dir $JUPYTER_PATH
```

##### JupyterLab

If you wish to use JupyterLab, you have the option to do so as it is enabled by the `conda`
environment build. JupyterLab is already installed, so you can follow similar steps as mentioned
above. `conda` manages the proper versioning issues and clashes, so you should not need to worry
about compatibility issues that often come with JupyterLab.

To start JupyterLab, you can use the following command:

```bash
jupyter lab
```

After installation, you can copy premade starter notebooks to the destination
directory and start JupyterLab using the provided commands. If you encounter an error related to
the magic extensions in JupyterLab, you can run the command to configure the IPython profile and
reload the magic extensions. Alternatively, you can manually reload the magic extensions for a
single notebook by running the command in an empty cell.

```bash
python -m graph_notebook.start_jupyterlab --jupyter-dir $JUPYTER_PATH
```
:::{note}
:class: a-tip-reference

Again the commands provided are generally for Unix-based systems. The equivalent for
Windows is provided.

```bash
python -m graph_notebook.start_jupyterlab --jupyter-dir %JUPYTER_PATH%
```
:::

When working with JupyterLab, specifically when trying to use line or cell magics in a new notebook,
you might encounter the following error:

```bash
UsageError: Cell magic `%%graph_notebook_config` not found."
```

This issue typically occurs because the necessary extensions or libraries that support these magics
aren't loaded into the Jupyter environment. Magics are special commands prefixed with `%` or `%%`
that perform specific functions in a notebook, such as setting configuration parameters or
formatting output. To resolve this issue, you need to ensure that the extension or library providing
 the `%%graph_notebook_config` magic is properly installed and recognized by JupyterLab.

To fix this, run the following command, then restart JupyterLab.

```bash
python -m graph_notebook.ipython_profile.configure_ipython_profile
```

Alternatively, the magic extensions can be manually reloaded for a single notebook by running the
following command in any empty cell.

```python
%load_ext graph_notebook.magics
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
conda remove -n ccss_xiwb --all
```

Rebuild the environment from the `environment.yml` file using the following command:

```bash
conda env create -f environment.yml
```
