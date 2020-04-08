git --version 2>&1 >/dev/null
GIT_IS_AVAILABLE=$?
if [ $GIT_IS_AVAILABLE -ne 0 ]; then #...
	if [[ "$OSTYPE" == "linux-gnu" ]]; then
		sudo apt-get install git
	elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
		brew install git
	else
		echo "The OS '$OSTYPE' is not supported (Only Linux and MacOS is). " \
			 "Feel free to contribute the install for a new OS."
		exit 1
	fi
fi
export PYTHONPATH=/home/travis/build/datamllab/PyODDS
git clone https://github.com/yli96/TDengine.git
cd TDengine/
mkdir build && cd build
cmake .. && cmake --build .
make install
cd ..
pip install src/connector/python/linux/python3
sudo taosd & cd ..
pip install -e .

