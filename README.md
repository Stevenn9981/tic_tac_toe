## RL for Tic-Tac-Toe Variant

All codes, including both algorithms and tests, can be found in `codes` folder.

Codes for part1 and part2 can be found in `codes/part1/` and `codes/part2/` folders, respectively.

### Requirements

```shell
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg freeglut3-dev
pip install tensorflow
pip install tf-agents[reverb]
pip install imageio
pip install imageio-ffmpeg
pip install pyvirtualdisplay
pip install pyglet
pip install pygame
pip install tabulate
pip install IPython
```

Please make sure the above dependencies are installed before running codes.

You can use the following commands to train the agents and evaluate (for part1 and part2, respectively):

```shell
python codes/part1/src/main.py 
python codes/part2/src/main.py
```

If you want to skip the training process and use the pre-trained model to conduct evaluation, use the following scripts:

```shell
python codes/part1/src/main.py use_pretrain
python codes/part2/src/main.py use_pretrain
```

To run the tests, use the following scripts:

```shell
python -m unittest codes/part1/test_part1.py 
python -m unittest codes/part2/test_part2.py
```


Besides, for your convenience, a Jupyter Notebook version is also provided so that you can also conduct model training and evaluation in Google Colab.

Part1: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg "Open in Colab")](https://colab.research.google.com/drive/1ix4_b3dvhbbdneNgnmJnbEb2epv6KD1M?usp=sharing)

Part2: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg "Open in Colab")](https://colab.research.google.com/drive/1t2xRSSG9HqOaWbu9kXdAL1TBMusjjzve?usp=sharing)
