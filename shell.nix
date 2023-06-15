{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "nixos-23.05";
    rev = "4ecab3273592f27479a583fb6d975d4aba3486fe";
  }) {}
}:
let py310 = pkgs.python310.withPackages (ps: with ps; [ numpy pandas ]);
    py311 = pkgs.python311.withPackages (ps: with ps; [ numpy pandas ]);
in
pkgs.mkShell {
  name = "looptrace-env";
  buildInputs = [
    pkgs.poetry
    py310
    py311
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];
  shellHook = ''
    poetry env use "${py310}/bin/python"
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib"
    poetry install --sync --with=dev
    source "$(poetry env info --path)/bin/activate"
  '';
}
