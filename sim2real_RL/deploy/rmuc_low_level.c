#include "rmuc_low_level.h"

#include <math.h>
#include <string.h>

static const int k_leg_indices[RMUC_LEG_COUNT] = {0, 1, 3, 4};
static const int k_wheel_indices[RMUC_WHEEL_COUNT] = {2, 5};
static const float k_wheel_action_sign[RMUC_WHEEL_COUNT] = {-1.0f, -1.0f};

static float clampf(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

RmucControlStatus rmuc_low_level_init(RmucLowLevelController *controller,
                                      const RmucHardwareCalibration *hardware) {
    if (controller == NULL || hardware == NULL) return RMUC_BAD_INPUT;
    memset(controller, 0, sizeof(*controller));
    if (!hardware->current_mapping_verified || !hardware->wheel_sign_verified ||
        !hardware->direct_current_mode_verified ||
        !hardware->leg_joint_torque_interface_verified) {
        return RMUC_UNVERIFIED_HARDWARE;
    }
    for (int i = 0; i < RMUC_WHEEL_COUNT; ++i) {
        const float gain = hardware->counts_per_output_nm[i];
        if (!isfinite(gain) || gain <= 0.0f ||
            !isfinite(hardware->wheel_feedback_damping_nm_s_rad[i]) ||
            hardware->wheel_feedback_damping_nm_s_rad[i] < 0.0f ||
            (hardware->joint_to_motor_sign[i] != -1 && hardware->joint_to_motor_sign[i] != 1) ||
            hardware->current_limit_counts[i] <= 0) {
            return RMUC_BAD_INPUT;
        }
        if (ceilf(5.5f * gain) > (float)hardware->current_limit_counts[i]) {
            return RMUC_INSUFFICIENT_WHEEL_PEAK;
        }
    }
    controller->hardware = *hardware;
    controller->hardware_ready = true;
    return RMUC_OK;
}

RmucControlStatus rmuc_low_level_set_action(RmucLowLevelController *controller,
                                            const float action[RMUC_ACTION_DIM]) {
    if (controller == NULL || action == NULL) return RMUC_BAD_INPUT;
    if (!controller->hardware_ready) return RMUC_UNVERIFIED_HARDWARE;
    if (controller->tick_1ms % RMUC_POLICY_TICKS != 0 ||
        (controller->has_action && controller->last_policy_tick == controller->tick_1ms)) {
        return RMUC_POLICY_DEADLINE;
    }
    for (int i = 0; i < RMUC_ACTION_DIM; ++i) {
        if (!isfinite(action[i])) return RMUC_BAD_INPUT;
    }
    memcpy(controller->action, action, sizeof(controller->action));
    controller->last_policy_tick = controller->tick_1ms;
    controller->has_action = true;
    return RMUC_OK;
}

RmucControlStatus rmuc_low_level_step(RmucLowLevelController *controller,
                                      const RmucJointSample *sample,
                                      RmucActuatorOutput *output) {
    if (output == NULL) return RMUC_BAD_INPUT;
    memset(output, 0, sizeof(*output));
    if (controller == NULL || sample == NULL) return RMUC_BAD_INPUT;
    if (!controller->hardware_ready) return RMUC_UNVERIFIED_HARDWARE;
    if (!controller->has_action ||
        controller->tick_1ms - controller->last_policy_tick >= RMUC_POLICY_TICKS) {
        return RMUC_POLICY_DEADLINE;
    }
    if (!isfinite(sample->leg_scale_rad) || sample->leg_scale_rad < 0.0f ||
        sample->leg_scale_rad > 1.0f) return RMUC_BAD_INPUT;
    for (int i = 0; i < RMUC_ACTION_DIM; ++i) {
        if (!isfinite(sample->joint_position_rad[i]) || !isfinite(sample->joint_velocity_rad_s[i]) ||
            !isfinite(sample->default_joint_position_rad[i]) ||
            !isfinite(sample->joint_lower_rad[i]) || !isfinite(sample->joint_upper_rad[i])) {
            return RMUC_BAD_INPUT;
        }
    }
    const bool update_pd = controller->tick_1ms % RMUC_LEG_PD_TICKS == 0;
    if (update_pd) {
        for (int i = 0; i < RMUC_LEG_COUNT; ++i) {
            const int joint = k_leg_indices[i];
            if (sample->joint_lower_rad[joint] + 0.02f >=
                sample->joint_upper_rad[joint] - 0.02f) return RMUC_BAD_INPUT;
        }
        for (int i = 0; i < RMUC_LEG_COUNT; ++i) {
            const int joint = k_leg_indices[i];
            const float low = sample->joint_lower_rad[joint] + 0.02f;
            const float high = sample->joint_upper_rad[joint] - 0.02f;
            const float target = clampf(sample->default_joint_position_rad[joint] +
                                        sample->leg_scale_rad * controller->action[joint], low, high);
            controller->held_leg_target_rad[i] = target;
            controller->held_leg_torque_nm[i] = clampf(
                60.0f * (target - sample->joint_position_rad[joint]) -
                2.0f * sample->joint_velocity_rad_s[joint], -40.0f, 40.0f);
        }
    }
    for (int i = 0; i < RMUC_LEG_COUNT; ++i) {
        output->leg_target_rad[i] = controller->held_leg_target_rad[i];
        output->leg_joint_torque_nm[i] = controller->held_leg_torque_nm[i];
    }
    for (int i = 0; i < RMUC_WHEEL_COUNT; ++i) {
        const int joint = k_wheel_indices[i];
        const float policy_torque = 5.5f * k_wheel_action_sign[i] *
                                    clampf(controller->action[joint], -1.0f, 1.0f);
        const float feedback_torque = -controller->hardware.wheel_feedback_damping_nm_s_rad[i] *
                                      sample->joint_velocity_rad_s[joint];
        /* This is the sole clip: C620 electromagnetic output at the wheel. */
        const float torque = clampf(policy_torque + feedback_torque, -5.5f, 5.5f);
        const float counts = torque * controller->hardware.counts_per_output_nm[i] *
                             (float)controller->hardware.joint_to_motor_sign[i];
        output->wheel_policy_torque_nm[i] = policy_torque;
        output->wheel_feedback_torque_nm[i] = feedback_torque;
        output->wheel_joint_torque_nm[i] = torque;
        output->wheel_current_counts[i] = (int32_t)lroundf(counts);
        output->wheel_above_continuous_reference[i] = fabsf(torque) > 3.0f;
    }
    output->leg_pd_updated = update_pd;
    output->tick_1ms = controller->tick_1ms;
    ++controller->tick_1ms;
    return RMUC_OK;
}
