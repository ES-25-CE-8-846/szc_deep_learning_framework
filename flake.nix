{
  description = "A flake for pythonification with PyTorch in a venv";

  # Define inputs (the sources we need to import)
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable"; # Use a recent version of nixpkgs
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs: let
    system = "x86_64-linux"; # NixOS system architecture
    pkgs = import nixpkgs {inherit system;}; # Import Nixpkgs

    pythonPackages = pkgs.python3Packages; # Python 3.x packages
    venvDir = "./env"; # Virtual environment directory

    # Global tools like pylint, flake8, black (installed via Nix, but not part of the project itself)
    globalPackages = with pkgs; [
      pythonPackages.pylint
      pythonPackages.flake8
      pythonPackages.black
    ];

    # Custom shell hook to set up the virtual environment if it doesn't exist
    postShellHook = ''
      if [ ! -d ${venvDir} ]; then
        echo "Creating virtual environment..."
        ${pythonPackages.virtualenv}/bin/virtualenv ${venvDir}  # Create venv
        # Activate the virtual environment and install required Python packages
        source ${venvDir}/bin/activate
        echo "Installing required Python packages (including PyTorch)..."
        pip install --no-cache-dir -r requirements.txt  # Install from requirements.txt if exists
        pip install --no-cache-dir torch torchvision torchaudio  # Install PyTorch and related packages
      fi
      # Ensure the virtual environment is activated in the shell
      source ${venvDir}/bin/activate
      export PYTHONPATH=\$PWD/\${venvDir}/lib/python3.*/site-packages:\$PYTHONPATH
    '';
  in {
    # This is the "run" shell where you would run your scripts
    runShell = pkgs.mkShell {
      name = "pythonify-run";
      buildInputs = with pkgs;
        [
          pythonPackages.python # Ensure Python is available
          pythonPackages.virtualenv # Ensure virtualenv is available
        ]
        ++ globalPackages; # Add global packages like pylint, flake8, etc.
      shellHook = postShellHook; # Add the postShellHook to set up the venv
    };

    # This is the development shell, similar to "runShell" but may include additional dev tools
    devShells.x86_64-linux.default = pkgs.mkShell {
      name = "pythonify-dev";
      buildInputs = with pkgs;
        [
          pythonPackages.python # Ensure Python is available
          pythonPackages.virtualenv # Ensure virtualenv is available
        ]
        ++ globalPackages; # Add global packages like pylint, flake8, etc.
      shellHook = postShellHook; # Add the postShellHook to set up the venv
    };
  };
}
