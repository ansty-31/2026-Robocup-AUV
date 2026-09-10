#include "Angle_resolve.h"
#include <stddef.h>

ANGLE_TYPE Angle;

float32_t Angle_wrap_deg(float32_t angle)
{
    if (angle > 180.0f) {
        angle -= 360.0f;
    } 
    else if (angle <= -180.0f)
    {
        angle += 360.0f;
    }
    return angle;
}

void Angle_Init(ANGLE_TYPE *Angle , float32_t Pitch_ORI , float32_t Yaw_ORI , float32_t Roll_ORI)
{
    Angle -> Roll = 0.0f;
    Angle -> Pitch = 0.0f;
    Angle -> Yaw = 0.0f;

    Angle -> Roll_rate = 0.0f;
    Angle -> Pitch_rate = 0.0f;
    Angle -> Yaw_rate = 0.0f;

    Angle -> Roll_ori = Roll_ORI;
    Angle -> Pitch_ori = Pitch_ORI;
    Angle -> Yaw_ori = Yaw_ORI;

    Angle -> plant1_Angle = 0.0f;
    Angle -> plant2_Angle = 0.0f;
}

void Angle_Resolve(ANGLE_TYPE *Angle)
{
    float32_t d_roll  = Angle_wrap_deg(Angle -> Roll  - Angle -> Roll_ori );
    float32_t d_pitch = Angle_wrap_deg(Angle -> Pitch - Angle -> Pitch_ori);
    float32_t d_yaw   = Angle_wrap_deg(Angle -> Yaw   - Angle -> Yaw_ori  );

    Angle->plant1_Angle = -d_pitch + d_yaw * INV_SQRT3;
    Angle->plant2_Angle = -d_pitch - d_yaw * INV_SQRT3;

    Angle -> plant1_Rate = Angle -> Pitch_rate + Angle -> Yaw_rate * INV_SQRT3;
    Angle -> plant2_Rate = Angle -> Pitch_rate - Angle -> Yaw_rate * INV_SQRT3;
}

