# How to run the main `looptrace` processing pipeline
This document describes the essentials needed to run the main `looptrace` processing pipeline. This is primarily for end-users and does not describe much about the software design or packaging.


## Minimal requirements
To be able to run this pipeline on the lab machine, these are the basic requirements:
* __Have an account on the machine__: you should be able to authenticate with your normal username/password combination used for most authentication within the institute.
* __Be in the `docker` group__: If you've not run something with `docker` on this machine previously, you're most likely not in this group. Ask Vince or Chris to add you.


## Data layout and organisation
* __Main experiment folder__ (`CURR_EXP_HOME` environment variable): On the cluster and on the lab machine, this is often something like `/path/to/experiments/folder/Experiments_00XXXX/00XXXX`, but it could be anything so long as the substructure matches what's expected / defined in the config file.
* __Images subfolder__ (created on lab machine or cluster): something like `images_all`, but just needs to match the value you'll give with the `--images-folder` argument when running the pipeline.
    * _Raw nuclei images_ subfolder: something like `nuc_images_raw`, though just must match the corresponding key in the config file
    * _FISH images_ subfolder: something like `seq_images_raw`, though just must match the corresponding key in the config file
* __Pypiper subfolder__ (created on lab machine or on cluster): something like `pypiper_output`, where pipeline logs and checkpoints are written; this will be passed by you to the pipeline runner through the `--pypiper-folder` argument when you run the pipeline.
* __Analysis subfolder__ (created on lab machine or on cluster): something like `2023-08-10_Analysis01`, though can be anything and just must match the name of the subfolder you supply in the config file, as the leaf of the path in the `analysis_path` value


## _Imaging rounds_ configuration file
This is a file that declares the imaging rounds executed over the course of an experiment.
These data should take the form of a mapping, stored as a JSON object; the sections are described in the separate document for the [imaging rounds configuration file](./imaging-rounds-configuration-file.md).
An example is available in the [test data](../src/test/resources/TestImagingRoundsConfiguration/example_imaging_round_configuration.json).


## _Parameters_ configuration file
Looptrace uses a configuration file to define values for processing parameters and pointers to places on disk from which to read and write data. 
The path to the configuration file is a required parameter in order to [run the pipeline](#general-workflow), and it should be created (or copied and edited) before running anything.
For requirements and suggestions on settings, refer to the separate documentation for the [parameters configuration file](./parameters-configuration-file.md).


## General workflow
Once you have the [minimal requirements](#minimal-requirements), this will be the typical workflow for running the pipeline:
1. __Login__ to the machine: something like `ssh username@ask-Vince-or-Chris-for-the-machine-domain`
1. __Path creation__: Assure that the necessary filepaths exist; particularly easy to forget are the path to the folder in which analysis output will be placed (the value of `analysis_path` in the parameters config file), and the path to the folder in which the pipeline will place its own files (`--pypiper-folder` argument at the command-line). See the [data layout section](#data-layout-and-organisation).
1. `tmux`: attach to an existing `tmux` session, or start a new one. See the [tmux section](#tmux) for more info.
1. __Docker__: Start the relevant Docker container, using just `docker` rather than `nvidia-docker` if you _don't_ want to run deconvolution (setting `decon_iter` to $0$, see [the parameters configuration file](./parameters-configuration-file.md))
    ```shell
    nvidia-docker run --rm -it -u root -v '/groups/gerlich/experiments/.../00XXXX':/home/experiment looptrace:2024-04-05b bash
    ```
1. __Run pipeline__: Once in the Docker container, run the pipeline, replacing the file and folder names as needed / desired:
    ```shell
    python /looptrace/bin/cli/run_processing_pipeline.py \
        --rounds-config /home/experiment/looptrace_00XXXX.rounds.json \
        --params-config /home/experiment/looptrace_00XXXX.params.yaml  \
        --images-folder /home/experiment/images_all \
        --pypiper-folder /home/experiment/pypiper_output
    ```
1. __Detach__: `Ctrl+b d` -- for more, see the [tmux section](#tmux).

NB: Remember to place single quotes around the filepath (experiment folder) you're making available as a volume (`-v`) to the Docker container. 
While often not necessary, this will protect you if your filepath contains spaces.

## `tmux`
In this context, for running the pipeline, think of the _terminal multiplexer_ (`tmux`) as a way to start a long-running process and be assured that an interruption in Internet connectivity (e.g., computer sleep or network failure) won't also be an interruption in the long-running process. If your connection to the remote machine is interrupted, but you've started your long-running process in `tmux`, that process won't be interrupted.

### Basic usage
* Start a session: `tmux`
* Detach from the active session: `Ctrl+b d` (i.e., press `Ctrl` and `b` at the same time, and then `d` afterward.)
* List sessions: `tmux list-sessions`
* Attach to a session: `tmux attach -t <session-number>`
* Detach again: `Ctrl+b d`
* (Eventually) stop a session: `tmux kill-session -t <session-number>`

For more, search for "tmux key bindings" or similar, or refer to [this helpful Gist](https://gist.github.com/mloskot/4285396)/
