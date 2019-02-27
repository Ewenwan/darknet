// 均值池化层cpu实现  创建初始化函数 前向传播/反向传播  
#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

// 均值池化层cpu实现  创建初始化函数
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    // 图片数量 * 宽度 * 高度 * 通道数量
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    
    // 所有w*h个元素的均值输出为一个 
    l.out_w = 1;
    l.out_h = 1;
    
    l.out_c = c;
    l.outputs = l.out_c; // 输出数量
    l.inputs = h*w*c;    // 输入数量
    
    int output_size = l.outputs * batch; // 所有批次，输出总数量
    
    l.output =  calloc(output_size, sizeof(float)); // 输出 变量 内存  cpu
    l.delta =   calloc(output_size, sizeof(float)); // 对应 梯度 内存
    
    l.forward = forward_avgpool_layer;   // 对应的函数指针
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size); // gpu 内存
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

// 指定 均值池化层 的 尺寸
void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

// 均值池化前向传播层==============================================
void forward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){ // 每张图片
        for(k = 0; k < l.c; ++k){ // 每个通道 ====输出后变成1个
            int out_index = k + b*l.c; // 输出 索引
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){ // 该通道内的所有变量
                int in_index = i + l.h*l.w*(k + b*l.c);// 该元素的索引
                l.output[out_index] += state.input[in_index];// 求和
            }
            l.output[out_index] /= l.h*l.w; // 求均值
        }
    }
}

// 均值池化层 反向传播
void backward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){ // 每张图片
        for(k = 0; k < l.c; ++k){// 每个通道 
            int out_index = k + b*l.c;// 输出 索引
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                state.delta[in_index] += l.delta[out_index] / (l.h*l.w); // 梯度反向传播
            }
        }
    }
}

