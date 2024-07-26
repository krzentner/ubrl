{
  description = "Unbound Reinforcement Learning";

  inputs = {
    # nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }: let
    pythonVersion = "python310";
  in
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell rec {
          buildInputs = [
            pkgs.python310
            pkgs.zlib
            pkgs.stdenv.cc.cc
            # pkgs.python310Packages.tensorboard
            pkgs.quarto
          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
          QUARTO_PYTHON = "${toString ./.}/.venv/bin/python";
        };
      }
    );
}
