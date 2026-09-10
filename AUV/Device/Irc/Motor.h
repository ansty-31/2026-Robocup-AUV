#ifndef __MOTOR_H
#define __MOTOR_H

#include "tim.h"
#include "PID.h"
#include "gpio.h"

#define Motor_1Polarity  -1
#define Motor_2Polarity  1
#define Motor_3Polarity  1
#define Motor_4Polarity  -1

#define Motor_5Polarity   -1
#define Motor_6Polarity   1
#define Motor_7Polarity   -1
#define Motor_8Polarity   -1
#define deadzone 10


#define RCStepA  490.0f / 255.0f//750��PWM�ķ�Χ��1500-3000��2250����ֵ

typedef struct
{
	float	MotorPow_1;
	float	MotorPow_2;
	float	MotorPow_3;
	float	MotorPow_4;
		
	float	MotorPow_5;
	float	MotorPow_6;
	float	MotorPow_7;
	float	MotorPow_8;
	
}MotorPower;

extern MotorPower RCPower;

void RCPower_Calc(MotorPower* POWER, uint8_t *RC);
void RCServo_Calc(uint8_t *RC);
int Servo_Limit(int a);




#endif
