#include "../sim2real_RL/deploy/rmuc_low_level.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static RmucHardwareCalibration synthetic_calibration(void) {
    RmucHardwareCalibration calibration = {0};
    calibration.counts_per_output_nm[0] = 2000.0f;
    calibration.counts_per_output_nm[1] = 2000.0f;
    calibration.joint_to_motor_sign[0] = 1;
    calibration.joint_to_motor_sign[1] = -1;
    calibration.current_limit_counts[0] = 12000;
    calibration.current_limit_counts[1] = 12000;
    calibration.current_mapping_verified = true;
    calibration.wheel_sign_verified = true;
    calibration.direct_current_mode_verified = true;
    calibration.leg_joint_torque_interface_verified = true;
    return calibration;
}

static RmucJointSample standing_sample(void) {
    RmucJointSample sample = {0};
    sample.leg_scale_rad = 0.22f;
    for (int i = 0; i < RMUC_ACTION_DIM; ++i) {
        sample.joint_lower_rad[i] = -3.0f;
        sample.joint_upper_rad[i] = 3.0f;
    }
    return sample;
}

static void test_unverified_and_insufficient_hardware(void) {
    RmucLowLevelController controller;
    RmucHardwareCalibration calibration = synthetic_calibration();
    calibration.wheel_sign_verified = false;
    assert(rmuc_low_level_init(&controller, &calibration) == RMUC_UNVERIFIED_HARDWARE);
    const float zeros[RMUC_ACTION_DIM] = {0};
    assert(rmuc_low_level_set_action(&controller, zeros) == RMUC_UNVERIFIED_HARDWARE);

    calibration = synthetic_calibration();
    calibration.counts_per_output_nm[0] = 19.0f / (0.00036f * (268.0f / 17.0f));
    calibration.current_limit_counts[0] = 8192;
    assert(rmuc_low_level_init(&controller, &calibration) == RMUC_INSUFFICIENT_WHEEL_PEAK);
}

static void test_pd_hold_wheel_peak_and_policy_deadline(void) {
    RmucLowLevelController controller;
    const RmucHardwareCalibration calibration = synthetic_calibration();
    assert(rmuc_low_level_init(&controller, &calibration) == RMUC_OK);
    const float action[RMUC_ACTION_DIM] = {1.0f, 1.0f, -2.0f, 1.0f, 1.0f, 2.0f};
    RmucJointSample sample = standing_sample();
    RmucActuatorOutput output;
    assert(rmuc_low_level_set_action(&controller, action) == RMUC_OK);
    assert(rmuc_low_level_set_action(&controller, action) == RMUC_POLICY_DEADLINE);
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_OK);
    assert(output.leg_pd_updated && output.tick_1ms == 0);
    for (int i = 0; i < RMUC_LEG_COUNT; ++i) {
        assert(fabsf(output.leg_target_rad[i] - 0.22f) < 1e-6f);
        assert(fabsf(output.leg_joint_torque_nm[i] - 13.2f) < 1e-5f);
    }
    assert(output.wheel_joint_torque_nm[0] == 5.5f);
    assert(output.wheel_joint_torque_nm[1] == -5.5f);
    assert(output.wheel_current_counts[0] == 11000);
    assert(output.wheel_current_counts[1] == 11000);
    assert(output.wheel_above_continuous_reference[0]);
    assert(output.wheel_above_continuous_reference[1]);
    assert(rmuc_low_level_set_action(&controller, action) == RMUC_POLICY_DEADLINE);

    sample.joint_velocity_rad_s[0] = 10.0f;
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_OK);
    assert(!output.leg_pd_updated);
    assert(fabsf(output.leg_joint_torque_nm[0] - 13.2f) < 1e-5f);
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_OK);
    assert(output.leg_pd_updated);
    assert(fabsf(output.leg_joint_torque_nm[0] + 6.8f) < 1e-5f);
    for (int i = 3; i < RMUC_POLICY_TICKS; ++i) {
        assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_OK);
    }
    assert(controller.tick_1ms == RMUC_POLICY_TICKS);
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_POLICY_DEADLINE);
    assert(output.wheel_current_counts[0] == 0);
    assert(rmuc_low_level_set_action(&controller, action) == RMUC_OK);
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_OK);
}

static void test_invalid_sensor_fails_zero_output(void) {
    RmucLowLevelController controller;
    const RmucHardwareCalibration calibration = synthetic_calibration();
    const float action[RMUC_ACTION_DIM] = {0};
    assert(rmuc_low_level_init(&controller, &calibration) == RMUC_OK);
    assert(rmuc_low_level_set_action(&controller, action) == RMUC_OK);
    RmucJointSample sample = standing_sample();
    sample.joint_position_rad[0] = NAN;
    RmucActuatorOutput output;
    memset(&output, 0x7f, sizeof(output));
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_BAD_INPUT);
    assert(output.wheel_current_counts[0] == 0 && output.leg_joint_torque_nm[0] == 0.0f);
}

static void test_active_feedback_damping_uses_the_motor_peak_clip(void) {
    RmucLowLevelController controller;
    RmucHardwareCalibration calibration = synthetic_calibration();
    calibration.wheel_feedback_damping_nm_s_rad[0] = RMUC_STAGE_D_COMP_NM_S_RAD;
    calibration.wheel_feedback_damping_nm_s_rad[1] = RMUC_STAGE_D_COMP_NM_S_RAD;
    const float action[RMUC_ACTION_DIM] = {0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f};
    RmucJointSample sample = standing_sample();
    sample.joint_velocity_rad_s[2] = 20.0f;
    sample.joint_velocity_rad_s[5] = -300.0f;
    RmucActuatorOutput output;
    assert(rmuc_low_level_init(&controller, &calibration) == RMUC_OK);
    assert(rmuc_low_level_set_action(&controller, action) == RMUC_OK);
    assert(rmuc_low_level_step(&controller, &sample, &output) == RMUC_OK);
    assert(fabsf(output.wheel_policy_torque_nm[0] - 5.5f) < 1e-6f);
    assert(fabsf(output.wheel_feedback_torque_nm[0] + 1.04f) < 1e-6f);
    assert(fabsf(output.wheel_joint_torque_nm[0] - 4.46f) < 1e-6f);
    assert(fabsf(output.wheel_policy_torque_nm[1] + 5.5f) < 1e-6f);
    assert(fabsf(output.wheel_feedback_torque_nm[1] - 15.6f) < 1e-6f);
    assert(fabsf(output.wheel_joint_torque_nm[1] - 5.5f) < 1e-6f);
}

/* Inputs here are normalized JOINT-direction actions. The existing policy
 * action-to-joint sign is -1; installation polarity is applied afterward. */
static float probe_motor(float joint_action, float omega) {
    RmucHardwareCalibration h = synthetic_calibration();
    h.wheel_feedback_damping_nm_s_rad[0] = RMUC_STAGE_D_COMP_NM_S_RAD;
    h.wheel_feedback_damping_nm_s_rad[1] = RMUC_STAGE_D_COMP_NM_S_RAD;
    RmucLowLevelController c;
    RmucJointSample s = standing_sample();
    RmucActuatorOutput out;
    float action[RMUC_ACTION_DIM] = {0};
    action[2] = action[5] = -joint_action;
    s.joint_velocity_rad_s[2] = s.joint_velocity_rad_s[5] = omega;
    assert(rmuc_low_level_init(&c, &h) == RMUC_OK);
    assert(rmuc_low_level_set_action(&c, action) == RMUC_OK);
    assert(rmuc_low_level_step(&c, &s, &out) == RMUC_OK);
    assert(out.wheel_joint_torque_nm[0] == out.wheel_joint_torque_nm[1]);
    assert(out.wheel_current_counts[0] == -out.wheel_current_counts[1]);
    return out.wheel_joint_torque_nm[0];
}

static void test_full_d_comp_boundaries(void) {
    assert(fabsf(probe_motor(0.0f, 30.0f) + 1.56f) < 1e-6f);
    assert(fabsf(probe_motor(0.0f, -30.0f) - 1.56f) < 1e-6f);
    assert(probe_motor(0.0f, 0.0f) == 0.0f);
    assert(fabsf(probe_motor(1.0f, 30.0f) - 3.94f) < 1e-6f);
    assert(probe_motor(1.0f, -30.0f) == 5.5f);
    assert(probe_motor(1.7f, 0.0f) == 5.5f);
    for (int a = -2; a <= 2; ++a) {
        float previous = probe_motor((float)a, -60.0f);
        for (int step = 1; step <= 1200; ++step) {
            float current = probe_motor((float)a, -60.0f + 0.1f * step);
            assert(fabsf(current) <= 5.5f);
            assert(current <= previous + 1e-6f);
            assert(fabsf(current - previous) <= 0.00521f);
            previous = current;
        }
    }
}

int main(void) {
    test_full_d_comp_boundaries();
    test_unverified_and_insufficient_hardware();
    test_pd_hold_wheel_peak_and_policy_deadline();
    test_invalid_sensor_fails_zero_output();
    test_active_feedback_damping_uses_the_motor_peak_clip();
    return 0;
}
