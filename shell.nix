{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "nixos-23.11";
    rev = "0a7b3c727a888ac4d1409c46fa2bffb25cb0ee90";
  }) {}, 
  pipeline ? false,
  analysis ? false, 
  interactive ? false,
  rDev ? true,
  pyDev ? true, 
  scalaDev ? true, 
  absolutelyOnlyR ? false,
  jdk ? "jdk19_headless",
}:
let baseBuildInputs = with pkgs; [ poetry stdenv.cc.cc.lib zlib ] ++ [ pkgs.${jdk} ];
    py310 = pkgs.python310.withPackages (ps: with ps; [ numpy pandas ]);
    myR = pkgs.rWrapper.override{ 
      packages = with pkgs.rPackages; [ argparse data_table ggplot2 reshape2 stringi ] ++ 
        (if rDev then [ pkgs.rPackages.languageserver ] else [ ]);
    };
    poetryExtras = [] ++ 
      (if pipeline then [ "pipeline" ] else []) ++
      (if analysis then [ "analysis" ] else []) ++ 
      (if interactive then [ "interactive" ] else []) ++
      (if pyDev then ["dev"] else []);
    poetryInstallExtras = (
      if poetryExtras == [] then ""
      else pkgs.lib.concatStrings [ " -E " (pkgs.lib.concatStringsSep " -E " poetryExtras) ]
      );
    scalaDevTools = with pkgs; [ ammonite coursier sbt-with-scala-native ];
in
pkgs.mkShell {
  name = "looptrace-env";
  buildInputs = [ myR ] ++ 
    (if absolutelyOnlyR then [ ] else baseBuildInputs ++ 
        (if pipeline then [
          py310
          pkgs.zlib
          pkgs.stdenv.cc.cc.lib
        ] else [ ]) ++ 
        (if scalaDev then scalaDevTools else [ ]));
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
