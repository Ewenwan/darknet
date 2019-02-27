// 激活函数类型枚举变量 各种激活函数/反激活函数的cpu实现 CPU/Gpu激活函数接口声明
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "cuda.h"
#include "math.h"

// 激活函数类型枚举变量  enum
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

// 从网络参数文件 的字符串中 解析出 激活函数的 枚举变量类型
ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a); // 写网络参数时会用到，枚举变量转字符串
float activate(float x, ACTIVATION a); // 单个元素 激活函数
float gradient(float x, ACTIVATION a); // 单个元素 反激活函数 激活函数 导数 梯度 反向传播

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);// 对应与数组多个元素的 激活函数
void activate_array(float *x, const int n, const ACTIVATION a);// 梯度反向传播

#ifdef GPU
void activate_array_ongpu(float *x, int n, ACTIVATION a);// gpu 中 对应与数组多个元素的 激活函数
void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta);// 梯度反向传播
#endif


// 各种激活函数  的  cpu实现===========static inline================
/*
说如果inline函数在两个不同的文件中出现，也就是说
一个.h被两个不同的文件包含，则会出现重名，链接失败
所以static inline 的用法就能很好的解决这个问题，
使用static修饰符，函数仅在文件内部可见，不会污染命名空间。
可以理解为一个inline在不同的.C里面生成了不同的实例，而且名字是完全相同的.
*/
static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}

// 返回线性矫正单元（ReLU）非线性激活函数关于输入x的导数  x*(x>0)
static inline float relu_activate(float x){return x*(x>0);}

static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}

// leaky ReLU非线性激活函数
// 0.1*x (,0] ;
// x     (0,)
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}

static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}

// 各种 反激活函数 导数梯度反向传播函数 的cpu实现
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}
static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

#endif

