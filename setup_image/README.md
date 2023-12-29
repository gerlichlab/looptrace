# `setup_image`: additional setup scripts for Docker image
The scripts here perform additional setup for building the `looptrace` Docker image.

At time of writing (2023-12-29), we need Java > 17 in order to avoid a known issue in `os-lib` whereby we can encounter `java.lang.ExceptionInInitializerError` by way of `IllegalArgumentError` with the message `requirement failed: ? is not an absolute path`. See more here from the [discussion on `os-lib`](https://github.com/com-lihaoyi/os-lib/issues/242).

[Java 21 rather than 19](https://github.com/gerlichlab/looptrace/issues/156) was chosen since 21 is a LTS version while 19 is not. 
Once Java 21 becomes available as a `stable` package rather than just `proposed`, these scripts and their [corresponding steps](https://github.com/vreuter/looptrace/commit/4d929bb0249425c6140308cf21b974f26ce9a4b5) in the [Dockerfile](../Dockerfile) can likely be removed.