#ifndef __MS5837_H
#define __MS5837_H

#include "main.h"
#include <stdint.h>
#include <stdbool.h>

#define MS5837_ADDR             (0x76U << 1)

#define MS5837_CMD_RESET        0x1EU
#define MS5837_CMD_ADC_READ     0x00U
#define MS5837_CMD_D1_8192      0x4AU
#define MS5837_CMD_D2_8192      0x5AU

#define MS5837_D1_PER_D2        9U

typedef enum
{
    MS5837_CONV_D1 = 0,
    MS5837_CONV_D2
} MS5837_ConvType_t;

typedef enum
{
    MS5837_STATE_IDLE = 0,
    MS5837_STATE_CONV_TX,       // 正在发送 D1/D2 转换命令
    MS5837_STATE_CONVERTING,    // 命令已发送，传感器正在转换
    MS5837_STATE_ADC_CMD_TX,    // 正在发送 0x00
    MS5837_STATE_ADC_DMA        // 正在 DMA 接收 ADC 三字节
} MS5837_State_t;

typedef struct
{
    int32_t temperature_01c;     // 0.01 °C，例如 1981 = 19.81 °C
    int32_t pressure_01mbar;     // 0.1 mbar，例如 10132 = 1013.2 mbar
    float depth_m;               // 深度，单位 m
} MS5837_Data_t;

/* 初始化：复位、读取 PROM、启动第一次 D2 转换 */
HAL_StatusTypeDef MS5837_Init(void);

/* 每 25 ms 调用一次；在 TIM12 回调中调用 */
void MS5837_TimerTick(void);

/* 在 HAL I2C 回调中转交调用 */
void MS5837_I2C_TxCpltCallback(I2C_HandleTypeDef *hi2c);
void MS5837_I2C_RxCpltCallback(I2C_HandleTypeDef *hi2c);
void MS5837_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c);

/* 主循环调用：有新压力数据时解算一次 */
void MS5837_Process(void);

/* 水面/零深度时设置。单位：0.1 mbar */
void MS5837_SetSurfacePressure(int32_t pressure_01mbar);

/* 读取最新已解算结果 */
const MS5837_Data_t *MS5837_GetData(void);

#endif
