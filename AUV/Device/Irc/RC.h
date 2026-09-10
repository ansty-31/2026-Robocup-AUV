#ifndef __RC_H
#define __RC_H

#include "main.h"

#define SA 9
#define SB 11
#define SC 10
#define SD 12
#define SE 13
#define SI 14

#define RcMulti 11  
#define RcKey    0xA5

#define FirstTime  1
#define NotFirstTime 0
#define Stop  1
#define Running  0

#define MyRcLength 15 
#define StopValue  5 

extern uint8_t MyRCKey[];
extern uint8_t LastMyRCKey[];
extern uint8_t RcData[RcMulti];
extern volatile uint8_t Serial_RcFlag;
extern uint8_t RC_SI_In, RC_SI_Out;

void RC_Translate(uint8_t *RcData);
/* USART2 遥控器 DMA 接收接口 */
void RC_InputByte(uint8_t byte);
	
uint8_t RC_WhetherYawStopJustNow(void);
uint8_t RC_WhetherYawStop(void);
uint8_t RC_WhetherRollStop(void);
uint8_t RC_WhetherSE_IN_JustNow(void);
uint8_t RC_WhetherSE_OUT_JustNow(void);

uint8_t Rc2MyRcKey(uint8_t RcNum);
uint8_t RC_Matching_F(uint8_t X);
uint8_t RC_Matching(uint8_t X);
uint8_t RC_Matching_Spin(uint8_t X);
#endif
