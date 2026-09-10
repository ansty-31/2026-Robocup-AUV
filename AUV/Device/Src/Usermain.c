#include "Angle_resolve.h"
#include "PID.h"

#include "Motor.h"
#include "Imu.h"
#include "RC.h"
#include "MS5837.h"

#include "usart.h"
#include "iwdg.h"
#include "tim.h"

#include <math.h>
#include <stdint.h>

#include "Usermain.h"

uint8_t PID_Cal_Flag = 0;
uint8_t RC_Cal_Flag  = 0;

float MAX_DEV = 400.0f;
float plant1_xishu = 1;
float plant2_xishu = 1;
float rc_xishu = 1;
float depth_xishu = 40;


float pid_dev[8] , rc_dev[8];
uint16_t t[8];

const MS5837_Data_t *MS5837_Data ;
uint8_t Send[16];

float target_depth = 0.45f;



uint16_t constrain_motor_power( uint16_t power , uint16_t max , uint16_t min)
{
    if ( power >= max ){return max;}
    if ( power <= min ){return min;}
    return power;
}

static int16_t to_i16_scaled(float value, float scale)
{
    float v = value * scale;
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    return (int16_t)v;
}

static void put_i16_le(uint8_t *buf, uint8_t index, int16_t value)
{
    uint16_t v = (uint16_t)value;
    buf[index] = (uint8_t)(v & 0xFF);
    buf[index + 1] = (uint8_t)((v >> 8) & 0xFF);
}

static void Telemetry_Send(float current_depth)
{
    static uint32_t last_send_ms = 0;
    uint32_t now = HAL_GetTick();

    if ((now - last_send_ms) < 100U)
    {
        return;
    }
    last_send_ms = now;

    Send[0] = 0xAA;
    Send[1] = 0x55;
    Send[2] = 10;

    put_i16_le(Send, 3,  to_i16_scaled(current_depth, 100.0f));
    put_i16_le(Send, 5,  to_i16_scaled(target_depth, 100.0f));
    put_i16_le(Send, 7,  to_i16_scaled(Angle.Roll, 100.0f));
    put_i16_le(Send, 9,  to_i16_scaled(Angle.Pitch, 100.0f));
    put_i16_le(Send, 11, to_i16_scaled(Angle.Yaw, 100.0f));

    uint8_t checksum = 0;
    for (uint8_t i = 0; i < 13; i++)
    {
        checksum += Send[i];
    }
    Send[13] = checksum;

    if (huart2.gState == HAL_UART_STATE_READY)
    {
        HAL_UART_Transmit_DMA(&huart2, Send, 14);
    }
}

void AUV_Init(void)
{
    MS5837_Init();
    HAL_TIM_Base_Start_IT(&htim12);
	PID_Init(&PID_roll , 8.0f , 0.0f , 0.1f , 300.0f , -300.0f);
	PID_Init(&PID_depth , 20.0f , 0.01f , 0.1f , 300.0f , -300.0f);

    PID_Init(&PID_plant1 , 8.0f , 0.0f , 0.1f , 300.0f , -300.0f);
    PID_Init(&PID_plant2 , 8.0f , 0.0f , 0.1f , 300.0f , -300.0f);
    Angle_Init(&Angle , 0.0f , 0.0f , 0.0f);
	
	
//	MS5837_SetSurfacePressure(MS5837_Data->pressure_01mbar);

    h30_configure();

    HAL_TIM_PWM_Start(&htim2 , TIM_CHANNEL_1 );//M1上右后
    HAL_TIM_PWM_Start(&htim2 , TIM_CHANNEL_2 );//M2下右后
    HAL_TIM_PWM_Start(&htim2 , TIM_CHANNEL_3 );//M3上左前
    HAL_TIM_PWM_Start(&htim2 , TIM_CHANNEL_4 );//M4下左前
	
    HAL_TIM_PWM_Start(&htim3 , TIM_CHANNEL_1 );//M5上左后
    HAL_TIM_PWM_Start(&htim3 , TIM_CHANNEL_2 );//M6下右前
    HAL_TIM_PWM_Start(&htim3 , TIM_CHANNEL_3 );//M7上右前
    HAL_TIM_PWM_Start(&htim3 , TIM_CHANNEL_4 );//M8下左后

    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_1 , 1500);
    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_2 , 1500);
    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_3 , 1500);
    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_4 , 1500);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_1 , 1500);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_2 , 1500);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_3 , 1500);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_4 , 1500);
	
}

void AUV_Task(void)
{
    HAL_IWDG_Refresh(&hiwdg);

    PID_Cal_Flag = 1;
    PID_Calculate(&PID_plant1 , 0.0f , Angle.plant1_Angle);
    PID_Calculate(&PID_plant2 , 0.0f , Angle.plant2_Angle);
	
	PID_Calculate(&PID_roll , 0.0f , Angle.Roll);
    PID_Cal_Flag = 0;

    RC_Cal_Flag = 1;
    RCPower_Calc(&RCPower , MyRCKey);
    RC_Cal_Flag = 0;
	
	static uint8_t depth_hold_enabled = 0;
	static uint8_t last_heave_active = 0;
	
    MS5837_Process();
    MS5837_Data = MS5837_GetData();
	
	float current_depth = MS5837_Data->depth_m;
	uint8_t heave_active = ((MyRCKey[5] + MyRCKey[6]) > StopValue);

	/* 手动上浮/下沉时不启用定深；松开升降通道时锁定当前深度。 */
	if (last_heave_active && !heave_active)
	{
		target_depth = current_depth;
		if (target_depth < 0.0f)
		{
			target_depth = 0.0f;
		}

		PID_depth.Error = 0.0f;
		PID_depth.PreError = 0.0f;
		PID_depth.Differ = 0.0f;
		PID_depth.Integral = 0.0f;
		PID_depth.Pout = 0.0f;
		PID_depth.Iout = 0.0f;
		PID_depth.Dout = 0.0f;
		PID_depth.PIDout = 0.0f;
		depth_hold_enabled = 1U;
	}

	last_heave_active = heave_active;

	if (heave_active || !depth_hold_enabled)
	{
		PID_depth.PIDout = 0.0f;
	}
	else
	{
		PID_Calculate(&PID_depth , target_depth , current_depth);
	}
    
    pid_dev[0] = - plant2_xishu * PID_plant2.PIDout - PID_roll.PIDout - depth_xishu * PID_depth.PIDout;	//1
    pid_dev[1] = - plant1_xishu * PID_plant1.PIDout - PID_roll.PIDout - depth_xishu * PID_depth.PIDout;	//2

    pid_dev[2] = - plant1_xishu * PID_plant1.PIDout - PID_roll.PIDout + depth_xishu * PID_depth.PIDout;	//3
    pid_dev[3] = - plant2_xishu * PID_plant2.PIDout - PID_roll.PIDout + depth_xishu * PID_depth.PIDout;	//4

    pid_dev[4] = - plant1_xishu * PID_plant1.PIDout + PID_roll.PIDout - depth_xishu * PID_depth.PIDout;	//5
    pid_dev[5] = - plant1_xishu * PID_plant1.PIDout + PID_roll.PIDout + depth_xishu * PID_depth.PIDout;	//6

    pid_dev[6] = + plant2_xishu * PID_plant2.PIDout - PID_roll.PIDout - depth_xishu * PID_depth.PIDout;	//7
    pid_dev[7] = - plant2_xishu * PID_plant2.PIDout + PID_roll.PIDout - depth_xishu * PID_depth.PIDout;	//8
	
//M1上右后
//M2下右后
//M3上左前
//M4下左前
//M5上左后
//M6下右前
//M7上右前
//M8下左后

    rc_dev[0] = + rc_xishu * RCPower.MotorPow_1 ;
    rc_dev[1] = + rc_xishu * RCPower.MotorPow_2 ;
    rc_dev[2] = + rc_xishu * RCPower.MotorPow_3 ;
    rc_dev[3] = + rc_xishu * RCPower.MotorPow_4 ;
    rc_dev[4] = + rc_xishu * RCPower.MotorPow_5 ;
    rc_dev[5] = + rc_xishu * RCPower.MotorPow_6 ;
    rc_dev[6] = + rc_xishu * RCPower.MotorPow_7 ;
    rc_dev[7] = + rc_xishu * RCPower.MotorPow_8 ;

    //比例缩放遥控器推力，避免与 PID 输出叠加后导致 PWM 溢出
    float rc_scale = 1.0f;
    for(int i = 0; i < 8; i++) {
        float pid_abs = fabsf(pid_dev[i]);
        if(pid_abs >= MAX_DEV)
        {
            // 如果 PID 为了稳住姿态就已经要满载了，直接切断人为遥控推力
            rc_scale = 0.0f; 
        } 
        else 
        {
            // 只有当 PID调整方向和遥控推力方向一致，才有可能导致PWM溢出
            if((pid_dev[i] >= 0 && rc_dev[i] > 0) || (pid_dev[i] <= 0 && rc_dev[i] < 0)) 
            {
                if(pid_abs + fabsf(rc_dev[i]) > MAX_DEV) 
                {
                    // 算出能塞下遥控器推力的剩余空间，并求出最小缩小比例
                    float s = (MAX_DEV - pid_abs) / fabsf(rc_dev[i]);
                     if(s < rc_scale) rc_scale = s; 
                }
            }
        }
    }

    t[0] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[0] + rc_scale * rc_dev[0]) , 1900 , 1100 );
    t[1] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[1] + rc_scale * rc_dev[1]) , 1900 , 1100 );
    t[2] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[2] + rc_scale * rc_dev[2]) , 1900 , 1100 );
    t[3] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[3] + rc_scale * rc_dev[3]) , 1900 , 1100 );
    t[4] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[4] + rc_scale * rc_dev[4]) , 1900 , 1100 );
    t[5] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[5] + rc_scale * rc_dev[5]) , 1900 , 1100 );
    t[6] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[6] + rc_scale * rc_dev[6]) , 1900 , 1100 );
    t[7] = constrain_motor_power( (uint16_t)(midvalue + pid_dev[7] + rc_scale * rc_dev[7]) , 1900 , 1100 );


    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_1 , t[0]);
    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_2 , t[1]);
    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_3 , t[2]);
    __HAL_TIM_SET_COMPARE(&htim2 , TIM_CHANNEL_4 , t[3]);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_1 , t[4]);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_2 , t[5]);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_3 , t[6]);
    __HAL_TIM_SET_COMPARE(&htim3 , TIM_CHANNEL_4 , t[7]);
	
	Telemetry_Send(current_depth);

}

