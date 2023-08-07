{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "nixos-23.05";
    rev = "4ecab3273592f27479a583fb6d975d4aba3486fe";
  }) {}, 
  pipeline ? false,
  analysis ? false, 
  pydev ? true, 
  absolutelyOnlyR ? false,
}:
let baseBuildInputs = with pkgs; [ poetry stdenv.cc.cc.lib zlib ];
    py310 = pkgs.python310.withPackages (ps: with ps; [ numpy pandas ]);
    R-analysis = pkgs.rWrapper.override{ packages = with pkgs.rPackages; [ argparse data_table ggplot2 reshape2 ]; };
    poetryExtras = [] ++ 
      (if pipeline then [ "pipeline" ] else []) ++
      (if analysis then [ "analysis" ] else []) ++ 
      (if pydev then ["dev"] else []);
    poetryInstallExtras = (
      if poetryExtras == [] then ""
      else pkgs.lib.concatStrings [ " -E " (pkgs.lib.concatStringsSep " -E " poetryExtras) ]
      );
in
pkgs.mkShell {
  name = "looptrace-env";
  buildInputs = if absolutelyOnlyR then [ R-analysis ] else baseBuildInputs ++ 
    (if pipeline then [
      py310
      pkgs.zlib
      pkgs.stdenv.cc.cc.lib
    ] else []) ++ 
    (if analysis then [ R-analysis ] else []);
  shellHook = if absolutelyOnlyR then "" else ''
    # To get this working on the lab machine, we need to modify Poetry's keyring interaction:
    # https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection
    # https://github.com/python-poetry/poetry/issues/1917
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    poetry env use "${py310}/bin/python"
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib"
    poetry install -vv --sync${poetryInstallExtras}
    source "$(poetry env info --path)/bin/activate"
  '';
}
