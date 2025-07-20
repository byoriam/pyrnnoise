#! /bin/bash -e

echo "🔄 Cleaning build artifacts..."
rm -rf ./build
rm -rf ./__pycache__
rm -rf ./pyrnnoise.egg-info
rm -rf ./pyrnnoise/build
rm -rf ./pyrnnoise/__pycache__
rm -f  ./pyrnnoise/librnnoise.so

echo "Prepare RNNoise repo (git submodule) (TODO: only run if folder does not exist)"
#git submodule update --init
#git checkout v0.2

echo "🛠️ Running CMake (configure)..."
cmake -B pyrnnoise/build -DCMAKE_BUILD_TYPE=Release

echo "🔨 Running CMake (build + install target)..."
cmake --build pyrnnoise/build --target install

echo "🐍 Building Python package..."
python setup.py build

echo "✅ Build completed successfully."
