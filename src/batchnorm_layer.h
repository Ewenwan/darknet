// 批 归一化层 层初始化函数  cpu/gpu 前向传播 / 反向传播函数 声明
#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c);// 批 归一化层 层初始化函数

// cpu  前向传播 / 反向传播函数 声明
void forward_batchnorm_layer(layer l, network_state state);
void backward_batchnorm_layer(layer l, network_state state);

// gpu 前向传播 / 反向传播函数 声明
#ifdef GPU
void forward_batchnorm_layer_gpu(layer l, network_state state);
void backward_batchnorm_layer_gpu(layer l, network_state state);
void pull_batchnorm_layer(layer l);
void push_batchnorm_layer(layer l);
#endif

#endif
