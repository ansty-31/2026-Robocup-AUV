#include "Imu.h"
#include "usart.h"
#include "Angle_resolve.h"
#include <stdint.h>
#include "RC.h"

#define H30_MAX_FRAME_LEN 270U

IMU_Data imu_data = {0};
FLOAT_Angle imu_offset = {0.0f, 0.0f, 0.0f};
uint8_t imu_offset_calibrated = 0U;
uint8_t imu_data_ready = 0U;

typedef enum { H30_STATE_IDLE = 0, H30_STATE_HEADER1,
               H30_STATE_HEADER2, H30_STATE_RECEIVING } H30_State;


static H30_State h30_state = H30_STATE_IDLE;
static uint8_t h30_rx_buffer[H30_MAX_FRAME_LEN];
static uint16_t h30_rx_index = 0U, h30_expected_len = 0U;


typedef struct { uint8_t ck1; uint8_t ck2; } H30_Checksum;

static H30_Checksum h30_calculate_checksum(const uint8_t *p, uint16_t n)
{
    H30_Checksum s = {0, 0};
    for (uint16_t i = 0U; i < n; i++) {
        s.ck1 = (uint8_t)(s.ck1 + p[i]);
        s.ck2 = (uint8_t)(s.ck2 + s.ck1);
    }
    return s;
}

static int32_t h30_read_i32_le(const uint8_t *p)
{
    return (int32_t)(((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static float h30_wrap_deg(float a)
{
    while (a > 180.0f) a -= 360.0f;
    while (a <= -180.0f) a += 360.0f;
    return a;
}

void h30_configure(void)
{
    static const uint8_t output_cmd[] = {0x59,0x53,0x04,0x12,0x00,0xD0,0x00,0xE6,0xFC};
    static const uint8_t frequency_cmd[] = {0x59,0x53,0x03,0x0A,0x00,0x08,0x15,0x32};

    HAL_UART_Transmit(&huart1, (uint8_t *)output_cmd, sizeof(output_cmd), 100);
    HAL_Delay(50);
    HAL_UART_Transmit(&huart1, (uint8_t *)frequency_cmd, sizeof(frequency_cmd), 100);
    HAL_Delay(50);
}

void h30_parse_data(uint8_t *data, uint16_t len)
{
    if ((data == NULL) || (len < 7U) || data[0] != 0x59U || data[1] != 0x53U) return; // 防呆

    uint8_t plen = data[4]; // 数据区长度（应为42）
    uint16_t total = (uint16_t)plen + 7; // 总长度（帧头2+长度3+数据区42+校验和2）

    if ((len != total) || (total > H30_MAX_FRAME_LEN)) return;
    H30_Checksum sum = h30_calculate_checksum(&data[2], (uint16_t)plen + 3);
    if (sum.ck1 != data[total-2] || sum.ck2 != data[total-1]) return; // 校验


    FLOAT_Accel acc = {0.0f,0.0f,0.0f};
    FLOAT_Gyro gyro = {0.0f,0.0f,0.0f};
    FLOAT_Angle ang = {0.0f,0.0f,0.0f};


    uint8_t ga=0U, gg=0U, ge=0U;
    uint8_t *payload = &data[5];
    uint16_t pos = 0U;
    while (pos + 2U <= plen) {
        uint8_t id = payload[pos], dl = payload[pos+1U];
        if (pos + 2U + dl > plen) return;
        uint8_t *v = &payload[pos+2U];
        if (id == 0x10U && dl == 12U) {
            acc.ax=h30_read_i32_le(&v[0])*0.000001f;
            acc.ay=h30_read_i32_le(&v[4])*0.000001f;
            acc.az=h30_read_i32_le(&v[8])*0.000001f; ga=1U;
        } else if (id == 0x20U && dl == 12U) {
            gyro.wx=h30_read_i32_le(&v[0])*0.000001f;
            gyro.wy=h30_read_i32_le(&v[4])*0.000001f;
            gyro.wz=h30_read_i32_le(&v[8])*0.000001f; gg=1U;
        } else if (id == 0x40U && dl == 12U) {
            ang.rol=h30_read_i32_le(&v[0])*0.000001f;
            ang.pit=h30_read_i32_le(&v[4])*0.000001f;
            ang.yaw=h30_read_i32_le(&v[8])*0.000001f; ge=1U;
        }
        pos = (uint16_t)(pos + 2U + dl);
    }

    //数据存入imu_data,初始数据存入imu_offset,并设置imu_offset_calibrated标志
    if (ga && gg && ge) 
    {
        if (!imu_offset_calibrated) 
        { 
            imu_offset=ang; 
            imu_offset_calibrated=1U; 
            Angle.Pitch_ori = imu_offset.pit;
            Angle.Roll_ori  = h30_wrap_deg(imu_offset.rol);
            Angle.Yaw_ori   = h30_wrap_deg(imu_offset.yaw);
        }

        imu_data.angle.pit=ang.pit;
        imu_data.angle.rol=h30_wrap_deg(ang.rol);
        imu_data.angle.yaw=h30_wrap_deg(ang.yaw);
        imu_data.gyro=gyro; 
        imu_data.accel=acc;

        /*
         * 同步写入 Angle_resolve.c 中定义的全局 Angle。
         * Angle 中的角度采用相对首次有效姿态的值，角速度和加速度
         * 采用当前 H30 帧解析值，单位分别为 deg/s、m/s^2。
         */
        if(MyRCKey[2] - MyRCKey[1] != 0) {
            Angle.Yaw_ori = h30_wrap_deg(imu_data.angle.yaw);
        }


        Angle.Pitch = imu_data.angle.pit;
        Angle.Roll  = imu_data.angle.rol;
        Angle.Yaw   = imu_data.angle.yaw;

        Angle.Pitch_rate = imu_data.gyro.wx;
        Angle.Roll_rate  = imu_data.gyro.wy;
        Angle.Yaw_rate   = imu_data.gyro.wz;

        Angle.Accel_x = imu_data.accel.ax;
        Angle.Accel_y = imu_data.accel.ay;
        Angle.Accel_z = imu_data.accel.az;

        /* 所有字段写完后再通知主循环。 */
        imu_data_ready = 1U;
    }
}

uint8_t h30_data_callback(uint8_t byte)
{
    switch (h30_state) {
    case H30_STATE_IDLE:
        if (byte==0x59U) { h30_rx_buffer[0]=byte; h30_rx_index=1U; h30_state=H30_STATE_HEADER1; } 
        break;
    case H30_STATE_HEADER1:
        if (byte==0x53U) { h30_rx_buffer[h30_rx_index++]=byte;
                           h30_state=H30_STATE_HEADER2; }
        else if (byte==0x59U) { h30_rx_buffer[0]=byte; h30_rx_index=1U; }
        else { h30_state=H30_STATE_IDLE; h30_rx_index=0U; } 
        break;
    case H30_STATE_HEADER2:
        if (h30_rx_index >= H30_MAX_FRAME_LEN) { h30_state=H30_STATE_IDLE; h30_rx_index=0U; break; }
        h30_rx_buffer[h30_rx_index++]=byte;
        if (h30_rx_index==5U) {
            h30_expected_len=(uint16_t)h30_rx_buffer[4]+7U;
            if (h30_expected_len>=7U && h30_expected_len<=H30_MAX_FRAME_LEN)
                h30_state=H30_STATE_RECEIVING;
            else { h30_state=H30_STATE_IDLE; h30_rx_index=0U; }
        } break;
    case H30_STATE_RECEIVING:
        if (h30_rx_index >= H30_MAX_FRAME_LEN) { h30_state=H30_STATE_IDLE; h30_rx_index=0U; break; }
        h30_rx_buffer[h30_rx_index++]=byte;
        if (h30_rx_index>=h30_expected_len) {
            h30_parse_data(h30_rx_buffer,h30_rx_index);
            h30_state=H30_STATE_IDLE; h30_rx_index=0U; h30_expected_len=0U;
        } break;
    default: h30_state=H30_STATE_IDLE; h30_rx_index=0U; h30_expected_len=0U; break;
    }
    return imu_data_ready;
}

void imu_reset_offset(void)
{
    imu_offset.pit=imu_offset.rol=imu_offset.yaw=0.0f;
    imu_offset_calibrated=0U; imu_data_ready=0U;
}
