<left><img src="https://github.com/seolhokim/BipedalWalker-BranchingDQN/blob/master/assets/video.gif" width="250" height="200"></left>
# BipedalWalker
Branching DQN implemetation with pytorch in BipedalWalker environment. 

## RUN

~~~
python main.py
~~~

  - if you want to change hyper-parameters, you can check "python main.py --help"
  - you just train and test basic model using main.py

  - '--train', type=bool, default=True, help="(default: True)"
  - '--render', type=bool, default=False, help="(default: False)"
  - '--epochs', type=int, default=1000, help='number of epochs, (default: 1000)'
  - '--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)'
  - '--lr_rate', type=float, default=0.0001, help='learning rate (default : 0.0001)'
  - '--batch_size', type=int, default=64, help='batch size(default : 64)'
  - '--gamma', type=float, default=0.99, help='gamma (default : 0.99)'
  - '--action_scale', type=int, default=6, help='action scale between -1 ~ +1'
  - "--load", type=str, default = 'no', help = 'load network name in ./model_weights'
  - "--save_interval", type=int, default = 100, help = 'save interval(default: 100)'
  - "--print_interval", type=int, default = 1, help = 'print interval(default : 1)'
## install BipedalWalker

### Ubuntu
```
conda install swig # needed to build Box2D in the pip install
pip install box2d-py # a repackaged version of pybox2d
```

https://stackoverflow.com/questions/44198228/install-pybox2d-for-python-3-6-with-conda-4-3-21

### Windows10

https://mclearninglab.tistory.com/136

