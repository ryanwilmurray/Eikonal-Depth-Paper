# Eikonal-Depth-Paper 
Code repository for the paper *Eikonal depth: an optimal control approach to statistical depths* by

- Martin Molina-Fructuoso
- Ryan Murray 


## Requirements

The main specific requirements are

-  graphlearning 0.0.2
- annoy 1.17.0

1. An environment to reproduce our results can be created using the file ``requirements.txt`` in the following way:
```
python3 -m venv .learning_env
source .learning_env/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
```

2. Alternatively, on a distribution based on Ubuntu (tested on Linux Mint 21.3), an environment to reproduce our results can be created as follows:
```
python3 -m venv .learning_env
source .learning_env/bin/activate
pip install scikit-learn
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install graphlearning==0.0.2
pip install notebook
pip install annoy==1.17.0
``` 
The installation of the following packages might be necessary before creating the environment:
```
apt install python3.10-venv
sudo apt-get install g++
sudo apt-get install python3-dev
```

## License
[cc-by-sa](http://creativecommons.org/licenses/by-sa/4.0/)
