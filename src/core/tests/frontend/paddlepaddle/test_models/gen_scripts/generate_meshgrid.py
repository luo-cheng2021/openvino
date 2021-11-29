# ref: https://www.paddlepaddle.org.cn/tutorials/projectdetail/1998893#anchor-2

import os
import sys

import numpy as np
import paddle

from save_model import exportModel, saveModel

x = paddle.randint(low=0, high=100, shape=[10])
y = paddle.randint(low=0, high=100, shape=[20])
print(x)
print(y)

def meshgrid():
    grid_x, grid_y = paddle.meshgrid(x, y)

    print(grid_x.shape)
    # print(grid_y.shape)

    #the shape of res_1 is (100, 200)
    #the shape of res_2 is (100, 200)

    print(grid_x)
    # print(grid_y)

def tile():
    reshape_x = paddle.reshape(x, [-1, 1])
    print(reshape_x.shape)
    tile_x = paddle.tile(reshape_x, repeat_times=[1, 20])
    print(tile_x)

if __name__ == "__main__":
    meshgrid()
    tile()


