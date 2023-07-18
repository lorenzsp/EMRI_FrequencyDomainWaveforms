git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms/
git checkout fd

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "This system is macOS."

    machine=$(uname -m)
    if [[ "$machine" == "arm64" ]]; then
        echo "This is an M1 Mac."
        conda create -n fd_env -c conda-forge -y wget gsl hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.9 openblas lapack liblapacke
    else
        echo "This is not an M1 Mac."
        conda create -n fd_env -c conda-forge -y clangxx_osx-64 clang_osx-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.10
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "This system is Unix/Linux."
    conda create -n fd_env -c conda-forge -y gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.10
else
    echo "Unsupported operating system."
fi

conda activate fd_env
pip install tqdm corner
machine=$(uname -m)

if [[ "$machine" == "arm64" ]]; then
    python setup.py install --no_omp --ccbin /usr/bin/
else
    python setup.py install --no_omp
fi

python setup.py install --no_omp
python -m unittest few/tests/test_fd.py
