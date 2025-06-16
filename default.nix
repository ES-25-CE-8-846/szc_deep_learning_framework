# save this as shell.nix or default.nix
{
  pkgs ?
    import (fetchTarball {
      # This URL should point to the tarball of the nixpkgs 24.11 commit
      url = "https://github.com/NixOS/nixpkgs/archive/refs/heads/nixos-24.11.tar.gz";
    }) {},
}: let
  pythonPackages = pkgs.python311Packages;

  pyroomacoustics = let
    pname = "pyroomacoustics";
    version = "0.8.3";
  in
    pythonPackages.buildPythonPackage {
      inherit pname version;
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "5619bcab3d03ffa4221383cf1ebaa2f078335c08d45601e10864612c8cbbc7d3";
      };
      propagatedBuildInputs = with pythonPackages; [
        cython
        numpy
        scipy
        pybind11
      ];
      doCheck = false;
    };
  pyflac = let
    pname = "pyflac";
    version = "3.0.0";
  in
    pythonPackages.buildPythonPackage {
      inherit pname version;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/2a/83/4f3e184618b1847d3b9905adc134ce016d32712151cc2c478d36f09727ef/pyFLAC-3.0.0.tar.gz";
        sha256 = "825d920e696f61493249afa2f7fd2fb42d7e7d2884a7b3e8b6ad1d76b5998119";
      };
      propagatedBuildInputs = with pythonPackages; [
        cffi
        numpy
        soundfile
      ];
      doCheck = false;
    };
  pesq = let
    pname = "pesq";
    version = "0.0.4";
  in
    pythonPackages.buildPythonPackage {
      inherit pname version;
      src = pkgs.fetchurl {
        url = "https://github.com/ludlows/python-pesq/archive/master.zip";
        sha256 = "sha256-brCuQmRhS8X7gSpMVSkb+9HxAy3hbtmYJjgmb+gmXsI=";
      };
      propagatedBuildInputs = with pythonPackages; [
        cffi
        numpy
        cython
      ];
      doCheck = false;
    };
  pystoi = let
    pname = "pystoi";
    version = "0.4.1";
  in
    pythonPackages.buildPythonPackage {
      inherit pname version;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/bf/3d/1ae8bdb686c6aaaeef474aa6b790abbe38f42b61188b57a974dd9320e521/pystoi-0.4.1.tar.gz";
        sha256 = "sha256-HG9Q1vv+5GsAySJFjNvScijZgwyoHOp4j9YA/C995uQ=";
      };
      propagatedBuildInputs = with pythonPackages; [
        cffi
        numpy
        cython
      ];
      doCheck = false;
    };
in
  pkgs.mkShell {
    buildInputs = with pythonPackages;
      [pyroomacoustics pyflac pesq pystoi]
      ++ (with pkgs; [
        python311
      ])
      ++ (with pythonPackages; [
        numpy
        scipy
        matplotlib
        jupyter-core
        ipykernel
        jupyterlab
        sounddevice
        torch
        torchaudio
        torchinfo
        soundfile
        wandb
        pandas
      ]);
  }
