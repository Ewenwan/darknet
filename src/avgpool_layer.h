// 均值池化层 函数声明 创建初始化函数 前向传播/反向传播 cpu/gpu 函数声明
#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer; // typedef 类型重命名

image get_avgpool_image(avgpool_layer l);
// 创建均值池化层===申请内存等
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
// 变形
void resize_avgpool_layer(avgpool_layer *l, int w, int h);

// cpu实现=======
// 均值池化层前向传播
void forward_avgpool_layer(const avgpool_layer l, network_state state);
// 均值池化层反向传播
void backward_avgpool_layer(const avgpool_layer l, network_state state);

// gpu实现=====
#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network_state state); // 前向传播
void backward_avgpool_layer_gpu(avgpool_layer l, network_state state);// 反向传播
#endif

#endif

