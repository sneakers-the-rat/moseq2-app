#!/bin/bash
option='0'
one='1'
two='2'
echo "Welcome to the MoSeq2-App Installer!"
echo "Enter your desired installation method."
echo "  1. Create a new conda environment named moseq2-app."
echo "  2. Install the latest MoSeq2 tools in your current conda environment."
echo "  Else to exit."
read option
if [ $option == $one ]; then
{
  echo 'Installing conda environment named moseq2-app'
  export CC="$(which gcc-7)"
  export CXX="$(which g++-7)"
} ||
{
  echo "gcc-7 not found. Trying default gcc version..."
  export CC="$(which gcc)"
  export CXX="$(which g++)"
} &&
{
  conda env create -f moseq2-env.yaml
  conda init moseq2-app
  conda activate moseq2-app
}
fi

if [ $option == $two ]; then
{
  echo 'Installing latest version of moseq2'
  export CC="$(which gcc-7)"
  export CXX="$(which g++-7)"
  pip install --upgrade jupyter
  conda install -c conda-forge ffmpeg -y
} ||
{
  echo "gcc-7 not found. Trying default gcc version..."
  export CC="$(which gcc)"
  export CXX="$(which g++)"
  pip install --upgrade jupyter
  conda install -c conda-forge ffmpeg -y
} &&
{
  pip install git+https://github.com/dattalab/moseq2-extract.git@release
  pip install git+https://github.com/dattalab/moseq2-pca.git@release
  pip install git+https://github.com/dattalab/moseq2-viz.git@release
  pip install git+https://github.com/dattalab/moseq2-model.git@release
}
fi
