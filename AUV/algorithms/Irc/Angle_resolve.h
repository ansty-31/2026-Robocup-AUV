#ifndef __ANGLE_RESOLVE_H__
#define __ANGLE_RESOLVE_H__

#include <stdint.h>
#include <stdbool.h>

typedef float float32_t;

typedef struct
{
    float32_t Roll;
    float32_t Pitch;
    float32_t Yaw;

    float32_t Roll_rate;
    float32_t Pitch_rate;
    float32_t Yaw_rate;

    float32_t Roll_ori;
    float32_t Pitch_ori;
    float32_t Yaw_ori;

    float32_t plant1_Angle;
    float32_t plant2_Angle;

    float32_t plant1_Rate;
    float32_t plant2_Rate;

    /* IMU 原始/相对零位加速度，单位 m/s^2 */
    float32_t Accel_x;
    float32_t Accel_y;
    float32_t Accel_z;

}ANGLE_TYPE;


#define SQRT3  1.7320508f
#define INV_SQRT3  0.5773503f

void Angle_Init(ANGLE_TYPE *Angle , float32_t Pitch_ORI , float32_t Yaw_ORI , float32_t Roll_ORI);
void Angle_Resolve(ANGLE_TYPE *Angle);

/* Angle_resolve.c 中定义的全局姿态数据对象 */
extern ANGLE_TYPE Angle;

#endif
