#include "RC.h"
#include <stdint.h>


uint8_t YawFirstTimeStopFlag = FirstTime;//1
uint8_t YawFirstTimeStopFlagflag = 0;
uint8_t YawStopFlag = 0;
uint8_t RollStopFlag = 0;

uint8_t MyRCKey[MyRcLength];
uint8_t LastMyRCKey[MyRcLength];

uint32_t RC_Disconnecttimes = 0;
uint8_t RCflag = 1;  

uint8_t RcData[RcMulti];
volatile uint8_t Serial_RcFlag = 0;
uint8_t RC_SI_In = 0, RC_SI_Out = 0;
static uint16_t rc_frame_pos = 0U;

void RC_InputByte(uint8_t byte)
{
	/* 等待帧头；错误或丢字节后可自动重新同步。 */
	if (rc_frame_pos == 0U)
	{
		if (byte != RcKey)
		{
			return;
		}
	}

	RcData[rc_frame_pos++] = byte;

	if (rc_frame_pos >= RcMulti)
	{
		rc_frame_pos = 0U;
		RCflag = 1U;
		RC_Translate(RcData);
		RC_Disconnecttimes = 0U;
		/* 原始帧和 MyRCKey 均更新完成后，再通知主循环。 */
		Serial_RcFlag = 1U;
	}
}

void RC_Translate(uint8_t *RcData)
{
	for(int i = 1;i < MyRcLength;i++)
	{
		LastMyRCKey[i] = MyRCKey[i];
	}
	
	if(RcData[1] > 127)
	{
		MyRCKey[2] = 0;
		MyRCKey[1] = RC_Matching(Rc2MyRcKey(RcData[1]));
	}
	if(RcData[1] <= 127)
	{
		MyRCKey[1] = 0;
		MyRCKey[2] = RC_Matching(Rc2MyRcKey(RcData[1]));
	}
	
	if(RcData[2] > 127)
	{
		MyRCKey[4] = 0;
		MyRCKey[3] = RC_Matching(Rc2MyRcKey(RcData[2]));
	}
	if(RcData[2] <= 127)
	{
		MyRCKey[3] = 0;
		MyRCKey[4] = RC_Matching(Rc2MyRcKey(RcData[2]));
	}
	
	if(RcData[3] > 127)
	{
		MyRCKey[6] = 0;
		MyRCKey[5] = RC_Matching(Rc2MyRcKey(RcData[3]));
	}
	if(RcData[3] <= 127)
	{
		MyRCKey[6] = RC_Matching(Rc2MyRcKey(RcData[3]));
		MyRCKey[5] = 0;
	}
	
	if(RcData[4] > 127)
	{
		MyRCKey[8] = 0;
		MyRCKey[7] = RC_Matching_Spin(Rc2MyRcKey(RcData[4]));
	}
	if(RcData[4] <= 127)
	{
		MyRCKey[7] = 0;
		MyRCKey[8] = RC_Matching_Spin(Rc2MyRcKey(RcData[4]));
	}
	
	MyRCKey[9] = RcData[5];//������ϽǶ�
	MyRCKey[10] = RcData[6];//�����
	MyRCKey[11] = RcData[7];//�ٶȵ�
	/* 不再清除 RcData[0]，使 RcData 始终保留最近一帧原始数据；
	 * Serial_RcFlag 用作“收到新帧”标志。 */
	if(YawFirstTimeStopFlagflag == 0)
	{
		YawFirstTimeStopFlag = RC_WhetherYawStopJustNow();
		if(YawFirstTimeStopFlag == FirstTime)
		{
			YawFirstTimeStopFlagflag = 1;
		}
	}
	if(RC_WhetherSE_IN_JustNow() == FirstTime)
	{
		RC_SI_In = MyRCKey[14];
	}
	if(RC_WhetherSE_OUT_JustNow() == FirstTime)
	{
		RC_SI_Out = MyRCKey[14];
	}
	YawStopFlag = RC_WhetherYawStop();
	RollStopFlag = RC_WhetherRollStop();
}

uint8_t RC_WhetherYawStopJustNow(void)
{
	if(MyRCKey[7] + MyRCKey[8] <= StopValue && LastMyRCKey[7] + LastMyRCKey[8] > StopValue)
	{
		return FirstTime;
	}
	else
	{
		return NotFirstTime;
	}
}

uint8_t RC_WhetherSE_IN_JustNow(void)
{
	if(MyRCKey[13] == 1 && LastMyRCKey[13] == 0)
	{
		return FirstTime;
	}
	else
	{
		return NotFirstTime;
	}
}

uint8_t RC_WhetherSE_OUT_JustNow(void)
{
	if(MyRCKey[13] == 0 && LastMyRCKey[13] == 1)
	{
		return FirstTime;
	}
	else
	{
		return NotFirstTime;
	}
}

uint8_t RC_WhetherYawStop(void)
{
	if(MyRCKey[7] + MyRCKey[8] <= StopValue)
	{
		return Stop;
	}
	else
	{
		return Running;
	}
}

uint8_t RC_WhetherRollStop(void)
{
	if(MyRCKey[9] + MyRCKey[10] <= StopValue)
	{
		return Stop;
	}
	else
	{
		return Running;
	}
}

uint8_t Rc2MyRcKey(uint8_t RcNum)
{
	if(RcNum == 127)
	{
		return 0;
	}
	if(RcNum > 127)
	{
		return (uint8_t)((float)(RcNum - 127) / 128.0f * 255.0f);
	}
	if(RcNum < 127)
	{
		return (uint8_t)((float)(127 - RcNum) / 127.0f * 255.0f);
	}
	return 0;
}

uint8_t RC_Matching(uint8_t X)
{
	if(X<=35)
	{
		return 0;
	}
	else
	{
		return 0.9583*X;
	}
}

uint8_t RC_Matching_F(uint8_t X)
{
	if(X<=35)
	{
		return 0;
	}
	else if(X < 150 && X>35)
	{
		return 0.555*X-3.349f;
	}
	else
	{
		return 1.666*X-170.06f;
	}
}

uint8_t RC_Matching_Spin(uint8_t X)
{
	if(X<=35)
	{
		return 0;
	}
	else if(X < 150 && X>35)
	{
		return 0.555*X-3.349f;
	}
	else
	{
		return 1.666*X-170.06f;
	}
}

