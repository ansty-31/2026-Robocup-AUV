#include "Motor.h"
#include "math.h"
#include "RC.h"
#include <stdint.h>

MotorPower RCPower;
float RCStep,RaiseStep;
int L_Servo;
/**
 * @brief  计算各电机的遥控期望推力（不含姿态 PID 修正）
 * @param  POWER 输出电机功率结构体指针
 * @param  RC    遥控关键通道数组指针（通常为 MyRCKey）
 * @note   RC[1]~[8] 约定：
 *           1/2：左旋/右旋正反向
 *           3/4：前进/后退正反向
 *           5/6：上升/下降正反向
 *           7/8：左右平移正反向
 *         本函数根据模式通道 RC[SC] 动态调整 RCStep，实现三档灵敏度。
 */
static const float allocation_matrix[8][4] = {//升降，左右，前后，偏航
    {  -2.50f,   -1.24f,  -1.67f,   -1.76f},  // Motor 1
    {  2.50f,  -1.24f,  -1.67f,  -1.76f},  // Motor 2
    {  -2.50f,   1.24f,   1.67f,  -1.76f},  // Motor 3
    {  2.50f,  1.24f,   1.67f,   -1.76f},  // Motor 4
    { -2.50f,   1.24f,  -1.67f,   1.76f},  // Motor 5
    { -2.50f,  1.24f,  -1.67f,  -1.76f},  // Motor 6
    { -2.50f,   -1.24f,   1.67f, 1.76f},  // Motor 7
    { -2.50f,  -1.24f,   1.67f,   -1.76f},  // Motor 8
};

//M1上右后
//M2下右后
//M3上左前
//M4下左前
//M5上左后
//M6下右前
//M7上右前
//M8下左后

void RCPower_Calc(MotorPower* POWER, uint8_t *RC)
{

	if(RC[SC] == 0)
	{
		RCStep = RCStepA * 0.20f;	
	}

	if(RC[SC] == 1)
	{
		RCStep = RCStepA * 0.20f *1.20f;	
	}
	
	if(RC[SC] == 2)
	{
		RCStep = RCStepA * 0.20f *3.0f;	
	}
	

	float command_up_down = (float)(RC[5] - RC[6]);
	float command_left_right = (float)(RC[7] - RC[8]);
	float command_forward_back = (float)(RC[3] - RC[4]);
	float command_yaw = (float)(RC[2] - RC[1]);

	POWER->MotorPow_1 = (allocation_matrix[0][0] * command_up_down + 
	                     allocation_matrix[0][1] * command_left_right + 
	                     allocation_matrix[0][2] * command_forward_back + 
	                     allocation_matrix[0][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_2 = (allocation_matrix[1][0] * command_up_down + 
	                     allocation_matrix[1][1] * command_left_right + 
	                     allocation_matrix[1][2] * command_forward_back + 
	                     allocation_matrix[1][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_3 = (allocation_matrix[2][0] * command_up_down + 
	                     allocation_matrix[2][1] * command_left_right + 
	                     allocation_matrix[2][2] * command_forward_back + 
	                     allocation_matrix[2][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_4 = (allocation_matrix[3][0] * command_up_down + 
	                     allocation_matrix[3][1] * command_left_right + 
	                     allocation_matrix[3][2] * command_forward_back + 
	                     allocation_matrix[3][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_5 = (allocation_matrix[4][0] * command_up_down + 
	                     allocation_matrix[4][1] * command_left_right + 
	                     allocation_matrix[4][2] * command_forward_back + 
	                     allocation_matrix[4][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_6 = (allocation_matrix[5][0] * command_up_down + 
	                     allocation_matrix[5][1] * command_left_right + 
	                     allocation_matrix[5][2] * command_forward_back + 
	                     allocation_matrix[5][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_7 = (allocation_matrix[6][0] * command_up_down + 
	                     allocation_matrix[6][1] * command_left_right + 
	                     allocation_matrix[6][2] * command_forward_back + 
	                     allocation_matrix[6][3] * command_yaw) * RCStep;
	
	POWER->MotorPow_8 = (allocation_matrix[7][0] * command_up_down + 
	                     allocation_matrix[7][1] * command_left_right + 
	                     allocation_matrix[7][2] * command_forward_back + 
	                     allocation_matrix[7][3] * command_yaw) * RCStep;		
	
	POWER->MotorPow_1 = POWER->MotorPow_1 * Motor_1Polarity;
	POWER->MotorPow_2 = POWER->MotorPow_2 * Motor_2Polarity;
	POWER->MotorPow_3 = POWER->MotorPow_3 * Motor_3Polarity;
	POWER->MotorPow_4 = POWER->MotorPow_4 * Motor_4Polarity;
	POWER->MotorPow_5 = POWER->MotorPow_5 * Motor_5Polarity;
	POWER->MotorPow_6 = POWER->MotorPow_6 * Motor_6Polarity;
	POWER->MotorPow_7 = POWER->MotorPow_7 * Motor_7Polarity;
	POWER->MotorPow_8 = POWER->MotorPow_8 * Motor_8Polarity;
	
}


/*
void RCServo_Calc(uint8_t *RC){	
	int Servo_CCR,Servo = RC[9];	
	if(RC_WhetherSE_IN_JustNow() == FirstTime){
		L_Servo = Servo;
		RC_SI_Out = MyRCKey[9];
  }
	Servo = Servo_Limit(L_Servo + MyRCKey[9] - RC_SI_Out);
	if (RC[SB] == 0){ 
		Servo_CCR = 600+(float)Servo/255.0f*(2400.0f-600.0f);
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_2, Servo_CCR);
	}
	if (RC[SB] == 1){
		Servo_CCR = 600+(float)Servo/255.0f*(2400.0f-600.0f);
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, Servo_CCR);
	}	
	if (RC[SB] == 2){
		Servo_CCR = 600+(float)Servo/255.0f*(2400.0f-600.0f);
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, Servo_CCR);
	}		
}
*/

int Servo_Limit(int a)
{
	if(a >= 255)
	{
		return 255;
	}
	else if(a <= 0)
	{
		return 0;
	}
	return a;
}
