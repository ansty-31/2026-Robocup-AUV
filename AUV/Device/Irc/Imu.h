#ifndef __IMU_H
#define __IMU_H

#include "main.h"
#include "Angle_resolve.h"

typedef struct { float pit; float rol; float yaw; } FLOAT_Angle;
typedef struct { float wx; float wy; float wz; } FLOAT_Gyro;
typedef struct { float ax; float ay; float az; } FLOAT_Accel;
typedef struct {
    FLOAT_Angle angle;
    FLOAT_Gyro gyro;
    FLOAT_Accel accel;
} IMU_Data;

extern IMU_Data imu_data;
extern FLOAT_Angle imu_offset;
extern uint8_t imu_offset_calibrated;
extern uint8_t imu_data_ready;


void imu_reset_offset(void);
void h30_configure(void);
uint8_t h30_data_callback(uint8_t byte);
void h30_parse_data(uint8_t *data, uint16_t len);
void USART1_Receive_DMA_Init(void);

#endif
