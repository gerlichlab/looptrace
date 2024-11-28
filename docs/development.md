# `looptrace` development

## Building with Docker
1. Clone the repo: `git clone ... && cd looptrace`
1. Checkout the appropriate commit: `git checkout ...`
1. Start Nix shell: `nix-shell`. Note that you may need to turn off various arguments (`--arg <name> false`), depending on the machine you're building on and the current state of the Nix shell definition file.
1. Build the JAR: `sbt assembly`
1. Create (and tag!) new image: `docker build -t looptrace:<new-tag> .`

## Running tests
This project mixes a few languages, and there are tests exist in Python and in Scala.

### _Scala tests_
Scala tests are written primarily with a mix of `ScalaTest` for execution and `ScalaCheck` for random test case generation towards an approximation of universal or qualified quantification of properties to which various components of the code should adhere. These can be run through, e.g., `sbt test` when in the project's development Nix shell.

### _Python_ tests
Python tests are written with [pytest](https://docs.pytest.org/en/7.4.x/contents.html).
These can be run through normal `pytest` syntax when in the project's development Nix shell.

## Updating documentation
If anything about pipeline stages is changed (e.g., a stage name is changed, stages are re-ordered, or a stage is added or removed), please run the [generate_execution_control_document.py](../bin/cli/generate_execution_control_document.py) to regenerate the [restart and execution guide](./pipeline-execution-control-and-rerun.md).
