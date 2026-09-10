#include "PID.h"
#include <stddef.h>

PID_TYPE PID_pitch , PID_roll , PID_yaw;
PID_TYPE PID_plant1 , PID_plant2;
PID_TYPE PID_depth; 

float32_t absf(float32_t x)
{
    return (x > 0) ? x : -x;
}

float32_t constrain(float32_t x, float32_t max)
{
    if (x < -max)
        return -max;
    else if (x > max)
        return max;
    else
        return x;
}

void PID_Init(PID_TYPE *pid, float32_t P, float32_t I, float32_t D, float32_t OutMax, float32_t OutMin)
{
    if (pid == NULL) return;
    if (OutMax < OutMin) return; // 防呆

    pid->P = P;
    pid->I = I;
    pid->D = D;

    pid->Error = 0.0f; // 本次误差
    pid->PreError = 0.0f; // 上次误差
    pid->Differ = 0.0f; // 本次误差与上次误差的差值
    pid->Integral = 0.0f; // 积分值

    pid->Ilimit = 10.0f; // 单次积分限幅
    pid->Ilimit_flag = 1.0f; // 积分限幅标志
    pid->Irang = 100.0f; // 积分范围

    pid->Pout = 0.0f;
    pid->Iout = 0.0f;
    pid->Dout = 0.0f;

    pid->PIDout = 0.0f;
    pid->OutMax = OutMax;
    pid->OutMin = OutMin;
}

void PID_Calculate(PID_TYPE *pid, float32_t Target, float32_t measured_value)
{
    if (pid == NULL) return;

    pid->Error = Target - measured_value; // 计算误差
    pid->Differ = pid->Error - pid->PreError; // 计算误差变化量
    pid->Integral += pid->Error; // 积分累加

    // 积分限幅
    if ( absf(pid -> Error) > pid -> Ilimit )
    {
        pid -> Ilimit_flag = 0.0f; // 禁止积分
    }
    else
    {
        pid -> Ilimit_flag = 1.0f; // 允许积分
    }

    // 积分范围限制
    pid->Integral = constrain(pid->Integral, pid->Irang);

    // PID输出计算
    pid->Pout = pid->P * pid->Error;
    pid->Iout = pid->I * pid->Integral * pid->Ilimit_flag;
    pid->Dout = pid->D * pid->Differ;

    // 总输出
    pid->PIDout = pid->Pout + pid->Iout + pid->Dout;

    // 输出限幅
    pid->PIDout = constrain(pid->PIDout, pid->OutMax);

    // 更新上次误差
    pid->PreError = pid->Error;
}
