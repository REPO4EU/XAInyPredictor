# Setting up a new project

## Repo structure

````bash
.
├── docs
│   └── Doxyfile
├── src/mymodule
│   └── (project-specific files and folders)
├── test
│   └── (project-testing-specific files and folders)
├── .dockerignore
├── .gitignore
├── .gitlab-ci.yml
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── Dockerfile
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.md
└── sonar-project.properties
````

- `Dockerfile`: The Docker configuration file used to build the Docker image for your project. This file defines the base image, dependencies, build steps, and other configurations required to run your project in a Docker container. You may need to update this file as your project evolves.

- `.dockerignore`: The .dockerignore file is used to specify files and folders that should be excluded from the Docker build context, which is sent to the Docker daemon when building an image.

- `.gitignore`: A file that specifies which files and folders should not be tracked by Git. This typically includes generated files, build artifacts, log files, and local configuration files.

- `.gitlab-ci.yml`: The GitLab CI/CD pipeline configuration file. This file defines the build, test, and deployment stages for your project. You may need to update this file as your project evolves to accommodate new dependencies, frameworks, or features.

- `.pre-commit-config.yaml`: Pre-commit hook definition. It includes some reasonable hooks to help you when commiting your work. They will check for syntax errorsin your files, standardize your python syntax and keep everything clean and sorted.

- `pyproject.toml`: Python project definition. It contains the requirements for building the project

- `README.md`: A documentation file that provides an overview of the project, its purpose, and how to use or contribute to it. This file should be updated as your project evolves to reflect the current state of the project.

- `requirements.txt`: The file containing all the package dependencies and their source if needed.

- `setup.md`: A documentation file that guides users through the process of setting up the project, including configuring the development environment, installing dependencies, and initializing the project structure. This file should be updated as your project evolves to reflect the current setup process.

- `sonar-project.properties`: The SonarQube config file to automatically run SonarQube during CI/CD.

- `src/mymodule`: The main folder where your project-specific code and configuration files will reside. This folder should contain a subfolder for each module that you will develop during your project.

- `test`: The main test folder where your project-specific code tests will reside. This folder contains a data folder for small files used during testing.


## Overview

This guide is designed to help you set up a new python project using a template directory structure.

The `src` folder is where your code will reside. For example, you could create a sub-folder in `src` for each submodule, and each submodule could have its own Dockerfile, GitLab CI pipeline, and so on. Similarly, for a monolithic architecture, you could have all the code for the project in a `mymodule` folder, along with any required configuration files, Dockerfile, GitLab CI pipeline, and so on.

The instructions provided in this guide are meant to be a starting point for configuring your project. You may need to modify these instructions to fit your specific use case or technology stack. In addition, some steps may require additional setup or configuration depending on your environment.

## Initial Project Setup

Follow these steps to set up this template for your Stalicla python project:

### Switching from the Template Repo to Your Own Project Repo

1. Clone the 'python' branch of the template repo into your local machine with the name of your project:

````bash
git clone git@gitlab.local.stalicladds.com:dds/templates/stalicla-development-templates.git --depth 1 --branch python --single-branch <your-project-name>
````

2. Navigate to the project folder:

````bash
cd <your-project-name>
````

3. Create an orphan branch to start a fresh commit history

```bash
git checkout --orphan temp
git commit -am "Initial commit"
```

After cloning the template repository, you should disconnect it from the template repo and connect it to your own project repository. This ensures that you don't accidentally commit changes to the template repository. Follow these steps:

4. Remove the existing remote connection to the template repository:

````bash
git remote remove origin
````

5. Create a new repository for your project on Gitlab (choose the same name used for cloning the template in step 1) and take note of its url.

6. Connect your local project to the newly created remote repository:

````bash
git remote add origin <your-new-repository-url>
````

7. Create and push the 'main' branch:

````bash
git checkout -b main
git push -u origin main
````

8. Create push the 'development' branch:

````bash
git checkout -b development
git push -u origin development
````

9. Delete the original branch:

````bash
git branch -D <original-template-branch>
````

10. Create a feature branch to start developing
```bash
git checkout -b 'feat_<feature_name>'
```

Now your project is connected to your own repository, and you can safely commit and push changes without affecting the template repository.

The content of both main and development branches is the same as the "python" branch of the original template repo.

### Installing initial dependencies and using Commitizen

1. Make sure you have python3, pip and conda installed in your system. You can check this by running:

````bash
python3 --version
python3 -m pip --version
conda --version
````
It's advised to build a conda environment specific for a project
and update the pyproject.toml as needed during development.


4. Install the project dependencies, including Commitizen:
2. Install [Commitizen](https://commitizen-tools.github.io/commitizen/) and [Pre-commit](https://pre-commit.com/#install) following the instructions.


### Linter and Pre-commit hooks

This template repository includes a default lintr and other standard prepare-commit hooks.

To set up specific hooks, exceptions or other configurations for your project, modify the `.pre-commit-config.yaml` file accordingly.

### 1. Install and test the pre-commit hooks

Install the hooks with:
````bash
pre-commit install
````

And then test them with:
````bash
pre-commit run --all-files
````
In order to update hooks to latest version, use:
````bash
pre-commit autoupdate
````

If the autoupdater updates some hooks commit the new version of the pre-commit config file
````
git add .pre-commit-config.yaml
cz c
````

When you want to make a commit, do NOT use the classic git commit.

To commit use the following command:
````bash
cz c
````
This command will trigger a console that prompts you with a series of questions to generate a well-formatted commit message.

This command will trigger a hook that runs the linter and prompts you with a series of questions to generate a well-formatted commit message.

### 2. Checks at commit time

When a commit is triggered, a series of checks will be performed as such:
````
Check: Python syntax errors?.............................................Passed
Check Trailing whitespace in the code?...................................Passed
Check: Wrong end-of-files?...............................................Passed
Check: Json syntax errors?...............................................Passed
Check: toml syntax errors?...............................................Passed
Check: yaml syntax errors?...............................................Passed
Check: Large files commited?.............................................Passed
Check: private keys commited?............................................Passed
Check: AWS credentials commited?.........................................Passed
Check: requirements.txt file sorted?.....................................Passed
Check: Commiting onto protected branches?................................Passed
Python code style?.......................................................Passed
Python import sorted?....................................................Passed
Check: Unlinted Dockerfile?..............................................Passed
gitlab ci correct?.......................................................Passed
````
In case of errors, some will be automatically fixed, while others will require manual correction, for example:
````
Check: Python syntax errors?.............................................Passed
Check Trailing whitespace in the code?...................................Passed
Check: Wrong end-of-files?...............................................Passed
Fixing src/mymodule/mymodule.py

Check: Wrong end-of-files?...............................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook

Fixing src/mymodule/mymodule.py

Check: Json syntax errors?...............................................Passed
Check: toml syntax errors?...............................................Passed
Check: yaml syntax errors?...............................................Failed
- hook id: check-yaml
- exit code: 1

while parsing a block mapping
  in ".gitlab-ci.yml", line 1, column 1
did not find expected key
  in ".gitlab-ci.yml", line 9, column 3


Check: Large files commited?.............................................Passed
Check: private keys commited?............................................Passed
Check: AWS credentials commited?.........................................Passed
Check: requirements.txt file sorted?.....................................Passed
Check: Commiting onto protected branches?................................Passed
Python code style?...................................(no files to check)Skipped
Python import sorted?....................................................Passed
Check: Unlinted Dockerfile?..............................................Passed
gitlab ci correct?.......................................................Passed

````
If it fails you can retry the failed commit with

````bash
cz c --retry
````

## Start developing TODO FROM HERE

Once you've set up commitizen and the pre-commit hooks, you can start developing your project in the `src` folder

As you develop your project and add new dependencies, or features, you might need to update the GitLab CI/CD configuration and the Docker setup accordingly. Here are some general guidelines:

### Follow the git-flow SOP
In the git-flow we are supposed to create a branch when implementing a new feature. Remember to create new branches and do NOT push to main or development branches. When your feature is ready merge your feature branch first in the development branch and then in main.

### Update GitLab CI/CD Configuration

The `.gitlab-ci.yml` file defines the build, test, and deployment stages for your project using GitLab CI/CD. The provided template contains a basic pipeline that covers checking, building, testing, and pushing Docker images and generating releases from tags. As you adapt the template to fit the needs of your specific project, you may need to modify this file to accommodate new dependencies, or features.

For example, you might need to update the build and test steps to include additional tasks specific to your project, such as installing new dependencies, running a different testing framework, or generating reports. Additionally, you may need to adjust the deployment stages to target specific environments or platforms, depending on your project's requirements.

Refer to Stalicla's CI/CD SOP for further information on how to manage the pipeline.

### Update Docker Configuration

The Docker structure consists of a `Dockerfile` and/or `docker-compose.yml` file. The `Dockerfile` defines how the Docker image for your project is built, including the base image, dependencies, build steps, and other configurations required to run your project in a Docker container. As you adapt the template to your specific project, you might need to modify the `Dockerfile` to include new dependencies or change the base image to one that better suits your project's language or framework.

The `docker-compose.yml` file is provided as a starting point for orchestrating multi-container applications. While it is not integrated into the pipeline by default, you can modify your pipeline stages to utilize docker-compose for building, testing, and deploying your application if needed. If this project is meant to be a continuosly deployed service, it is recommended to provide a Docker-compose file that defines the conditions of deployment instead of a `docker run` command that could be easily lost in the command history.

Refer to Stalicla's Docker SOP for further information on how to containerize your project.
