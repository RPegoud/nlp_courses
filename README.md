# NLP Class Repository

Welcome to the NLP Class Repository! This repository contains the code and materials for our Natural Language Processing class. To get started, follow the instructions below to set up the project environment and install the necessary dependencies using Poetry.

## Table of Contents

- [NLP Class Repository](#nlp-class-repository)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10  or higher installed on your system.
- Poetry installed. If you haven't installed Poetry yet, you can do so by following the instructions [here](https://python-poetry.org/docs/) or simply running:

    ```bash
    pip install poetry
    ```

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/RPegoud/nlp_courses.git
   ```

2. Navigate to the project directory:

    ```bash
    cd nlp_courses
    ```

3. Use Poetry to create a virtual environment and install project dependencies (This command will create a virtual environment specifically for this project and install all the required dependencies defined in the `pyproject.toml` file).

    ```bash
    poetry install
    ```

4. Activate the virtual environment. You are now in the project's virtual environment, where you can run Python scripts and use the installed dependencies.:

    ```bash
    poetry shell
    ```

5. Restart VS Code and you should be able to select the `nlp_course` kernel
