{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "release-24.05";
    rev = "a417c003326c9a0cdebb62157466603313ffd47e";
  }) { overlays = [ (self: super: {
    pkginfo = super.python311Packages.pkginfo.overrideAttrs (oldAttrs: {
      version = "1.12.0";
      src = super.fetchPypi {
        pname = "pkginfo";
        version = "1.12.0";
        sha256 = "sha256-itkaBEWgNngrk2bvi4wsUCkfg6VTR4uoWAxz0yFXAM8";
      };
    });

    poetry-core = super.python311Packages.poetry-core.overrideAttrs (oldAttrs: {
      version = "1.9.1";
      src = super.fetchFromGitHub {
        owner = "python-poetry";
        repo = "poetry-core";
        rev = "1.9.1";
        sha256 = "sha256-L8lR9sUdRYqjkDCQ0XHXZm5X6xD40t1gxlGiovvb/+8";
      };
    });

    virtualenv = super.virtualenv.overrideAttrs (oldAttrs: {
      version = "20.26.6";
      src = super.fetchPypi {
        pname = "virtualenv";
        version = "20.26.6";
        sha256 = "sha256-KArt4JoqXDF+QJoAEC5wd8ZDLFo48O+TjmQ4BaetLEg";
      };
    });

    poetry = super.poetry.overrideAttrs (oldAttrs: {
      version = "1.8.5";
      src = super.fetchFromGitHub {
        owner = "python-poetry";
        repo = "poetry";
        rev = "1.8.5";
        sha256 = "sha256-YR0IgDhmpbe8TyTMP1cjUxGRnrfV8CNHkPlZrNcnof0";
      };
      buildInputs = oldAttrs.buildInputs or [] ++ [ self.poetry-core self.virtualenv self.pkginfo ];
    });
  })];
  }, 
  pipeline ? false,
  test ? true,
  deconvolution ? false,
  analysis ? false, 
  interactive-visualisation ? true,
  new-mac-napari ? true,
  rDev ? true,
  pyDev ? true, 
  scalaDev ? true, 
  absolutelyOnlyR ? false,
  jdk ? "jdk21_headless",
}:
let baseBuildInputs = with pkgs; [ poetry stdenv.cc.cc.lib zlib ] ++ [ pkgs.${jdk} ];
    py311 = pkgs.python311.withPackages (ps: with ps; [ numpy pandas ]);
    myR = pkgs.rWrapper.override{ 
      packages = with pkgs.rPackages; [ argparse data_table ggplot2 stringi ] ++ 
        (if rDev then [ pkgs.rPackages.languageserver ] else [ ]);
    };
    poetryExtras = [] ++ (
      (if pipeline then [ "pipeline" ] else []) ++
      (if test then [ "test" "pipeline" ] else []) ++ 
      (if deconvolution then [ "deconvolution" ] else []) ++
      (if analysis then [ "analysis" ] else []) ++ 
      (if interactive-visualisation then [ "interactive-visualisation" ] else []) ++
      (if new-mac-napari then [ "new-mac-napari" ] else []) ++
      (if pyDev then [ "dev" "test" ] else [])
    );
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
        (if pipeline then [ py311 ] else [ ]) ++ 
        (if scalaDev then scalaDevTools else [ ]));
  shellHook = if absolutelyOnlyR then "" else ''
    # To get this working on the lab machine, we need to modify Poetry's keyring interaction:
    # https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection
    # https://github.com/python-poetry/poetry/issues/1917
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    poetry env use "${py311}/bin/python"
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib"
    installcmd="poetry install -vv --sync${poetryInstallExtras}"
    echo "Running installation command: $installcmd"
    eval "$installcmd"
    source "$(poetry env info --path)/bin/activate"
  '';
}
