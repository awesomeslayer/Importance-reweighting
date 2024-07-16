import nox

locations = "source", "tests", "plots", "noxfile.py"

@nox.session(python=["3.11"])
def lint(session):
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)

@nox.session(tags=["style", "fix"])
def black(session):
    session.install("black")
    session.run("black", ".")

@nox.session(tags=["style", "fix"])
def isort(session):
    session.install("isort")
    session.run("isort", ".")

@nox.session(tags=["style", "fix"])
def pylint(session):
    session.install("pylint")
    session.run("pylint", *locations)

@nox.session(python=["3.11"])
def install_packages(session):
    packages = [
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "tqdm",
    ]
    session.install(*packages)

all_packages = [
    "numpy",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "tqdm",
]

@nox.session(name="setup-env")
def setup_env(session):
    session.install(*all_packages)

@nox.session(python=["3.11"])
def install_poetry(session):
    session.run("pip", "install", "poetry")
    session.run("poetry", "install")

