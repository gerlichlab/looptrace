# `looptrace` development

## Running tests
This project mixes a few languages, and there are tests exist in Python and in Scala.

### _Scala tests_
Scala tests are written primarily with a mix of `ScalaTest` for execution and `ScalaCheck` for random test case generation towards an approximation of universal or qualified quantification of properties to which various components of the code should adhere. These can be run through something like `sbt test` when in the project's development Nix shell.

### _Python_ tests
Python tests are written with [pytest](https://docs.pytest.org/en/7.4.x/contents.html). To establish a marking and opt-in protocol for slow tests (e.g. integration tests which use relatively large data), we additionally use [pytest-skip-slow](). This accomplishes two things:
    * Adds a `@pytest.mark.slow` marker with which to tag tests as slow (open to interpretation)
    * Adds `--slow` as opt-in hook to pass through command-like to execute tests tagged as such, e.g. `pytest --slow`.

These can be run through normal `pytest` syntax when in the project's development Nix shell.

