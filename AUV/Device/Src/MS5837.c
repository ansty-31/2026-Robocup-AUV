#include "MS5837.h"
#include "i2c.h"

static uint16_t s_C[7];         // 使用 C[1] ~ C[6]
static uint8_t s_tx_byte;
static uint8_t s_adc_rx[3];

static volatile uint32_t s_d1_raw;
static volatile uint32_t s_d2_raw;
static volatile bool s_pressure_ready = false;

static volatile uint8_t s_d1_count = 0;
static volatile MS5837_ConvType_t s_conv_type;
static volatile MS5837_State_t s_state = MS5837_STATE_IDLE;

static int32_t s_surface_pressure_01mbar = 9306.5;
static MS5837_Data_t s_data;


/* 读取 PROM：0xA0 到 0xAC */
static HAL_StatusTypeDef MS5837_ReadPROM(void)
{
    uint8_t command;
    uint8_t rx[2];

    for (uint8_t i = 0; i < 7; i++)
    {
        command = (uint8_t)(0xA0U + i * 2U);

        if (HAL_I2C_Master_Transmit(&hi2c1, MS5837_ADDR,
                                    &command, 1, 20) != HAL_OK)
        {
            return HAL_ERROR;
        }

        if (HAL_I2C_Master_Receive(&hi2c1, MS5837_ADDR,
                                   rx, 2, 20) != HAL_OK)
        {
            return HAL_ERROR;
        }

        s_C[i] = ((uint16_t)rx[0] << 8) | rx[1];
    }

    return HAL_OK;
}


/* 非 DMA 发送下一次转换命令；转换完成后由 Tx 回调进入 CONVERTING */
static void MS5837_StartConversion(MS5837_ConvType_t type)
{
    s_conv_type = type;

    if (type == MS5837_CONV_D1)
    {
        s_tx_byte = MS5837_CMD_D1_8192;
    }
    else
    {
        s_tx_byte = MS5837_CMD_D2_8192;
    }

    s_state = MS5837_STATE_CONV_TX;

    if (HAL_I2C_Master_Transmit_IT(&hi2c1, MS5837_ADDR,
                                   &s_tx_byte, 1) != HAL_OK)
    {
        s_state = MS5837_STATE_IDLE;
    }
}


HAL_StatusTypeDef MS5837_Init(void)
{
    s_tx_byte = MS5837_CMD_RESET;

    if (HAL_I2C_Master_Transmit(&hi2c1, MS5837_ADDR,
                                &s_tx_byte, 1, 20) != HAL_OK)
    {
        return HAL_ERROR;
    }

    HAL_Delay(10);

    if (MS5837_ReadPROM() != HAL_OK)
    {
        return HAL_ERROR;
    }

    s_d1_count = 0;
    s_pressure_ready = false;

    /* 首次先转换温度，供后续 9 个 D1 使用 */
    MS5837_StartConversion(MS5837_CONV_D2);

    return HAL_OK;
}


/*
 * 在 HAL_TIM_PeriodElapsedCallback 中每 25 ms 调用。
 * 此时上一笔 D1/D2 转换已完成，启动 ADC 读取过程。
 */
void MS5837_TimerTick(void)
{
    if (s_state != MS5837_STATE_CONVERTING)
    {
        return;
    }

    s_tx_byte = MS5837_CMD_ADC_READ;
    s_state = MS5837_STATE_ADC_CMD_TX;

    if (HAL_I2C_Master_Transmit_IT(&hi2c1, MS5837_ADDR,
                                   &s_tx_byte, 1) != HAL_OK)
    {
        s_state = MS5837_STATE_IDLE;
    }
}


/* 从 HAL_I2C_MasterTxCpltCallback 转调 */
void MS5837_I2C_TxCpltCallback(I2C_HandleTypeDef *hi2c)
{
    if (hi2c != &hi2c1)
    {
        return;
    }

    if (s_state == MS5837_STATE_CONV_TX)
    {
        /* D1/D2 命令已经真正送出，开始等待转换完成 */
        s_state = MS5837_STATE_CONVERTING;
    }
    else if (s_state == MS5837_STATE_ADC_CMD_TX)
    {
        s_state = MS5837_STATE_ADC_DMA;

        if (HAL_I2C_Master_Receive_DMA(&hi2c1, MS5837_ADDR,
                                       s_adc_rx, 3) != HAL_OK)
        {
            s_state = MS5837_STATE_IDLE;
        }
    }
}


/* 从 HAL_I2C_MasterRxCpltCallback 转调 */
void MS5837_I2C_RxCpltCallback(I2C_HandleTypeDef *hi2c)
{
    uint32_t raw;

    if ((hi2c != &hi2c1) || (s_state != MS5837_STATE_ADC_DMA))
    {
        return;
    }

    raw = ((uint32_t)s_adc_rx[0] << 16) |
          ((uint32_t)s_adc_rx[1] << 8)  |
          ((uint32_t)s_adc_rx[2]);

    if (s_conv_type == MS5837_CONV_D2)
    {
        s_d2_raw = raw;
        s_d1_count = 0;

        /* 刚得到温度，随后连续测 9 次压力 */
        MS5837_StartConversion(MS5837_CONV_D1);
    }
    else
    {
        s_d1_raw = raw;
        s_pressure_ready = true;
        s_d1_count++;

        if (s_d1_count >= MS5837_D1_PER_D2)
        {
            MS5837_StartConversion(MS5837_CONV_D2);
        }
        else
        {
            MS5837_StartConversion(MS5837_CONV_D1);
        }
    }
}


/* 从 HAL_I2C_ErrorCallback 转调 */
void MS5837_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c)
{
    if (hi2c == &hi2c1)
    {
        s_state = MS5837_STATE_IDLE;
    }
}


/* 一阶和二阶温度补偿，并换算深度 */
static void MS5837_Calculate(uint32_t D1, uint32_t D2)
{
    int32_t dT;
    int32_t temp;
    int64_t off;
    int64_t sens;
    int64_t ti;
    int64_t offi;
    int64_t sensi;
    int32_t pressure;

    dT = (int32_t)D2 - ((int32_t)s_C[5] << 8);

    temp = 2000 +
           (int32_t)(((int64_t)dT * s_C[6]) >> 23);

    off = ((int64_t)s_C[2] << 16) +
          (((int64_t)s_C[4] * dT) >> 7);

    sens = ((int64_t)s_C[1] << 15) +
           (((int64_t)s_C[3] * dT) >> 8);

    /* 二阶温度补偿 */
    if (temp < 2000)
    {
        ti = (3LL * dT * dT) >> 33;

        offi = (3LL * (int64_t)(temp - 2000) *
                (temp - 2000)) >> 1;

        sensi = (5LL * (int64_t)(temp - 2000) *
                 (temp - 2000)) >> 3;

        if (temp < -1500)
        {
            offi += 7LL * (int64_t)(temp + 1500) *
                    (temp + 1500);

            sensi += 4LL * (int64_t)(temp + 1500) *
                     (temp + 1500);
        }
    }
    else
    {
        ti = (2LL * dT * dT) >> 37;

        offi = ((int64_t)(temp - 2000) *
                (temp - 2000)) >> 4;

        sensi = 0;
    }

    temp -= (int32_t)ti;
    off  -= offi;
    sens -= sensi;

    pressure = (int32_t)(((((int64_t)D1 * sens) >> 21) - off) >> 13);

    s_data.temperature_01c = temp;
    s_data.pressure_01mbar = pressure;

    /*
     * 淡水：约 98.0665 mbar/m。
     * pressure 的单位为 0.1 mbar，因此除数是 980.665。
     */
    s_data.depth_m =
        ((float)(pressure - s_surface_pressure_01mbar)) / 980.665f;
}


/* 放在 while(1) 中调用 */
void MS5837_Process(void)
{
    uint32_t d1_local;
    uint32_t d2_local;

    if (!s_pressure_ready)
    {
        return;
    }

    __disable_irq();

    s_pressure_ready = false;
    d1_local = s_d1_raw;
    d2_local = s_d2_raw;

    __enable_irq();

    MS5837_Calculate(d1_local, d2_local);
}


void MS5837_SetSurfacePressure(int32_t pressure_01mbar)
{
    s_surface_pressure_01mbar = pressure_01mbar;
}


const MS5837_Data_t *MS5837_GetData(void)
{
    return &s_data;
}
