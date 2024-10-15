# Phishing URL Detection

This project aims to detect phishing URLs based on various URL features using machine learning techniques.

#### Dataset : [url of dataset]( https://www.kaggle.com/datasets/simaanjali/tes-upload ) 

## Getting Started

To get started with the project, follow the instructions below.

### Prerequisites

- Ensure you have `make` installed on your machine.
- The project uses `Poetry` for package management. If you do not have `Poetry` installed, refer to the [Poetry installation guide](https://python-poetry.org/docs/#installation).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dhruv-88/Phishing_url_detection.git
   cd Phishing_url_detection
2. **Create the Conda environment: Use the following command to create the environment**
   ```bash
   make create-env
3. **Activate Conda Enviorment**
   ```bash
   conda activate group_project
4. **Check Poetry installation: Ensure Poetry is correctly installed using** :
   ```bash
   make check-poetry
5. **Install dependencies: Install all required dependencies via Poetry**:
   ```bash
   make install-deps-poetry

### Adding New Files with Poetry

If you need to add new files or install additional dependencies, follow these steps:

1. **To add a new package**:
   Use Poetry to add a package to the project:
   ```bash
   poetry add <package-name>

### Contributing 
If you'd like to contribute to the project, feel free to fork the repository and create a pull request. Ensure all new dependencies are managed through Poetry.
