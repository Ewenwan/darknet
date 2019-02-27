// 激活层CPU/GPU初始化 激活层 正向/方向传播 CPU/GPU 函数 接口
#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 创建激活层====层初始化===网络初始化函数=====
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*)); // 输出内存
    l.delta = calloc(batch*inputs, sizeof(float*));  // 梯度内存

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); // cuda gpu 内存
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation; // 激活类型
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs); // 打印信息
    return l;
}

// cpu 激活层 前向反向传播==============
// 激活层前向传播
void forward_activation_layer(layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation); // 执行激活函数
}
// 激活层反向传播
void backward_activation_layer(layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);// 激活梯度反向传播
    copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

// GPU 激活层 前向反向传播==============
#ifdef GPU
void forward_activation_layer_gpu(layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}
void backward_activation_layer_gpu(layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
