{
  description = "A flake for Python development with PyTorch and NumPy";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3;

      pythonPackages = ps: with ps; [
        pip
        numpy
        torch
        torchvision
        torchaudio
        torchinfo
        scipy
        black
        flake8
        pylint
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "python-dev-shell";

        buildInputs = [
          (python.withPackages pythonPackages)
        ];

        shellHook = ''
          echo "Python dev environment with NumPy + PyTorch is ready."
        '';
      };
    };
}
