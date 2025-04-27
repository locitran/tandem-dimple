#!/bin/bash
echo $CONDA_PY
if [ -z $CONDA_PY ]; then
    echo plain build
    # so that packages are installed in order specified
    cat requirements.txt | grep -v '#' | xargs -n1 pip install
else
    echo conda build
    if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
        brew install md5sha1sum
    fi
    source devtools/travis-ci/install_miniconda.sh
    conda install --yes --file requirements.txt
fi
