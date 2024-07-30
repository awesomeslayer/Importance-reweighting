import nox

locations = "source", "tests", "plots", "noxfile.py"

PYTHON_VERSIONS = ["3.11.9"]


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)


@nox.session(tags=["style", "fix"])
def black(session):
    session.install("black")
    session.run("black", ".")


@nox.session(python=PYTHON_VERSIONS, tags=["style", "fix"])
def isort(session):
    session.install("isort")
    session.run("isort", ".")


@nox.session(python=PYTHON_VERSIONS, tags=["style", "fix"])
def pylint(session):
    session.install("pylint")
    session.run("pylint", *locations)


@nox.session(python=PYTHON_VERSIONS)
def poetry_run(session):
    session.run("poetry", "install", external=True)
    session.run(
        "poetry", "run", "python3.11", "-m", "plots.max_cov_plot", external=True
    )
