# Chromatin fiber tracing and analysis.

`looptrace` is a software suite for chromatin tracing image and data analysis as described in https://doi.org/10.1101/2021.04.12.439407

## Using (and perhaps installing) `looptrace`
We expect that in most cases `looptrace` will be used as a __processing pipeline__, with perhaps some post-hoc analysis of the resulting, processed data. 
For that, we have a [Docker image](./Dockerfile) and documentation on how to [run the pipeline](./docs/running-the-pipeline.md).

To use `looptrace` in more of a package / library fashion, you may work with various components of the [Python package](./looptrace) and/or the [Scala library](./src).

In either case, we recommend to [install Nix](https://nixos.org/download) and then use this project's [Nix shell](./shell.nix) with whatever `--arg <argname> true` values you need to pull in the dependency group(s) needed for your use case.

## Authors
Written by Kai Sandvold Beckwith (kai.beckwith@embl.de), [Ellenberg group](https://www-ellenberg.embl.de/), CBB, EMBL Heidelberg.

Extended and maintained by Vincent Reuter, [Gerlich group](https://www.oeaw.ac.at/imba/research/daniel-gerlich/), Institute for Molecular Biotechnology (IMBA) in Vienna, Austria.
See https://www.oeaw.ac.at/imba/research/daniel-gerlich/

## Citation
Please cite this paper: https://www.biorxiv.org/content/10.1101/2021.04.12.439407
