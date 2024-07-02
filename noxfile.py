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
