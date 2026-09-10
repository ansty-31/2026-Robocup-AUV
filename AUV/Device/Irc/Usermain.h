#ifndef __USERMAIN_H
#define __USERMAIN_H

#define midvalue 1488

extern uint8_t PID_Cal_Flag;
extern uint8_t RC_Cal_Flag;

void AUV_Init(void);
void AUV_Task(void);


#endif
