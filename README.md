# wishes-generator
Neural network that generates Christmas messages

Language : python 2.7
Dependencies:
- theano 0.8.1 OR tensorflow >= 1.2.0
- Keras 2.1.2

Dependencies can be installed as:

sudo pip install theano==0.8.1
sudo pip install tensorflow==1.2.0
sudo pip install keras==2.1.2

Run generator as:

python generate.py --seed "your string seed"

Parameters of script are:

--seed : any string that will be a beginning of your Chrismtas message. Length of string must be >= 100 symbols

Optional parameters:

--length : how many symbols to generate ( can be from 1 to inf). By default 1000 symbols

--div : diversity (can vary from 0.2 to 1.0). Parameter that specifies how concervative neural network will be in it's generation. More diversity - more interesting messages but at the same time more spelling errors. By default 0.5

