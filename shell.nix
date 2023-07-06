{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "nixos-23.05";
    rev = "4ecab3273592f27479a583fb6d975d4aba3486fe";
  }) {}, 
  pipeline ? false,
  analysis ? false, 
  pydev ? false,
}:
let baseBuildInputs = with pkgs; [ poetry stdenv.cc.cc.lib zlib ];
    py310 = pkgs.python310.withPackages (ps: with ps; [ numpy pandas ]);
    py311 = pkgs.python311.withPackages (ps: with ps; [ numpy pandas ]);
    R-analysis = pkgs.rWrapper.override{ packages = with pkgs.rPackages; [ data_table ggplot2 ]; };
    poetryGroups = [] ++ 
      (if pipeline then ["pipeline"] else []) ++
      (if analysis then ["analysis"] else []) ++ 
      (if pydev then ["dev"] else []);
    poetryInstallExtras = (
      if poetryGroups == [] then ""
      else " --with " ++ pkgs.lib.concatStringsSep "," poetryGroups
      );
in
pkgs.mkShell {
  name = "looptrace-env";
  buildInputs = baseBuildInputs ++ 
    (if pipeline then [
      py310
      py311
      pkgs.zlib
      pkgs.stdenv.cc.cc.lib
    ] else []) ++ 
    (if analysis then [ R-analysis ] else []);
  shellHook = ''
    poetry env use "${py310}/bin/python"
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib"
    poetry install --sync${poetryInstallExtras}
    source "$(poetry env info --path)/bin/activate"
  '';
}
