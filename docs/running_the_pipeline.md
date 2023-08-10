# How to run the main `looptrace` processing pipeline
This document describes the essentials needed to run the main `looptrace` processing pipeline. This is primarily for end-users and does not describe much about the software design or packaging.


## Minimal requirements
To be able to run this pipeline on the lab machine, these are the basic requirements:
* __Have an account on the machine__: you should be able to authenticate with your normal username/password combination used for most authentication within the institute.
* __Be in the `docker` group__: If you've not run something with `docker` on this machine previously, you're most likely not in this group. Ask Vince or Chris to add you.


## General workflow
Once you have the [minimal requirements](#minimal-requirements), this will be the typical workflow for running the pipeline:
1. __Login__ to the machine: something like `ssh username@ask-Vince-or-Chris-for-the-machine-domain`
1. __Path creation__: Assure that the necessary filepaths exist; particularly easy to forget are the path to the folder in which analysis output will be placed (the value of `analysis_path` in the config file), and the path to the folder in which the pipeline will place its own files (`-O / --output-folder` argument at the command-line).
1. `tmux`: attach to an existing `tmux` session, or start a new one. See the [tmux section](#tmux) for more info.
1. __Docker__: Start the relevant Docker container:
```shell

```
<a name="run-pipeline"></a>
1. __Run pipeline__: Once in the Docker container, run the pipeline, replacing the file and folder names as needed / desired:
```shell
python /looptrace/bin/cli/run_processing_pipeline.py -C /home/experiment/current_experiment_number_looptrace.yaml -I /home/experiment/images_all -O /home/experiment/pypiper_output
```
1. __Detach__: `Ctrl+b d` -- for more, see the [tmux section](#tmux).


## tmux
In this context, for running the pipeline, think of the _terminal multiplexer_ (`tmux`) as a way to start a long-running process and be assured that an interruption in Internet connectivity (e.g., computer sleep or network failure) won't also be an interruption in the long-running process. If your connection to the remote machine is interrupted, but you've started your long-running process in `tmux`, that process won't be interrupted.

### Basic usage
1. Start a session: `tmux`
1. Detach from the active session: `Ctrl+b d` (i.e., press `Ctrl` and `b` at the same time, and then `d` afterward.)
1. List sessions: `tmux list-sessions`
1. Attach to a session: `tmux attach -t <session-number>`
1. Detach again: `Ctrl+b d`
1. (Eventually) stop a session: `tmux kill-session -t <session-number>`

For more, search for "tmux key bindings" or similar, or refer to [this helpful Gist](https://gist.github.com/mloskot/4285396)/


## Data: folder layout / organisation
* __Main experiment folder__ (`CURR_EXP_HOME` environment variable): On the cluster and on the lab machine, this is often something like `/path/to/experiments/folder/Experiments_00XXXX/00XXXX`, but it could be anything so long as the substructure matches what's expected / defined in the config file.
* __Images subfolder__ (created on lab machine or cluster): something like `images_all`, but just needs to match the value you'll give with the `-I / --images-folder` argument when [running the pipeline](#run-pipeline)
    * _Raw nuclei images_ subfolder: something like `nuc_images_raw`, though just must match the corresponding key in the config file
    * _FISH images_ subfolder: something like `seq_images_raw`, though just must match the corresponding key in the config file
* __Pypiper subfolder__ (created on lab machine or on cluster): something like `pypiper_output`, where pipeline logs and checkpoints are written; this will be passed by you to the pipeline runner through the `-O / --output-folder` argument when you [run the pipeline](#run-pipeline).
* __Analysis subfolder__ (created on lab machine or on cluster): something like `2023-08-10_Analysis01`, though can be anything and just must match the name of the subfolder you supply in the config file, as the leaf of the path in the `analysis_path` value

