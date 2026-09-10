#ifndef PID_H
#define PID_H

#include <stdint.h>
#include <stdbool.h>

typedef float float32_t;

typedef struct
{
    float32_t P;
    float32_t I;
    float32_t D;

    float32_t Error;
    float32_t PreError;
    float32_t Differ;
    float32_t Integral;

    float32_t Ilimit;
    float32_t Ilimit_flag;
    float32_t Irang;

    float32_t Pout;
    float32_t Iout;
    float32_t Dout;

    float32_t PIDout;
    float32_t OutMax;
    float32_t OutMin;
    
}PID_TYPE;

extern PID_TYPE PID_plant1 , PID_plant2 , PID_roll , PID_depth;

float32_t constrain(float32_t value, float32_t limit);
float32_t absf(float32_t value);
void PID_Init(PID_TYPE *pid, float32_t P, float32_t I, float32_t D, float32_t OutMax, float32_t OutMin);
void PID_Calculate(PID_TYPE *pid, float32_t Target, float32_t measured_value);


#endif // PID_H
