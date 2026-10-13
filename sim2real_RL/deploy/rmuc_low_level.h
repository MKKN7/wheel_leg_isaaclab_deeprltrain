#ifndef RMUC_LOW_LEVEL_H
#define RMUC_LOW_LEVEL_H

#include <stdbool.h>
#include <stdint.h>

enum {
    RMUC_ACTION_DIM = 6,
    RMUC_LEG_COUNT = 4,
    RMUC_WHEEL_COUNT = 2,
    RMUC_POLICY_TICKS = 8,
    RMUC_LEG_PD_TICKS = 2
};

/* D=.05 active plus b=.002 passive in the current training stage, assuming
 * negligible real passive b. Explicit calibration still supplies each wheel. */
#define RMUC_STAGE_D_COMP_NM_S_RAD 0.052f

typedef enum {
    RMUC_OK = 0,
    RMUC_BAD_INPUT = -1,
    RMUC_UNVERIFIED_HARDWARE = -2,
    RMUC_INSUFFICIENT_WHEEL_PEAK = -3,
    RMUC_POLICY_DEADLINE = -4
} RmucControlStatus;

typedef struct {
    float counts_per_output_nm[RMUC_WHEEL_COUNT];
    /* Active low-level velocity feedback, not mechanical b or Coulomb loss. */
    float wheel_feedback_damping_nm_s_rad[RMUC_WHEEL_COUNT];
    int8_t joint_to_motor_sign[RMUC_WHEEL_COUNT];
    int32_t current_limit_counts[RMUC_WHEEL_COUNT];
    bool current_mapping_verified;
    bool wheel_sign_verified;
    bool direct_current_mode_verified;
    bool leg_joint_torque_interface_verified;
} RmucHardwareCalibration;

typedef struct {
    float joint_position_rad[RMUC_ACTION_DIM];
    float joint_velocity_rad_s[RMUC_ACTION_DIM];
    float default_joint_position_rad[RMUC_ACTION_DIM];
    float joint_lower_rad[RMUC_ACTION_DIM];
    float joint_upper_rad[RMUC_ACTION_DIM];
    float leg_scale_rad;
} RmucJointSample;

typedef struct {
    float leg_target_rad[RMUC_LEG_COUNT];
    float leg_joint_torque_nm[RMUC_LEG_COUNT];
    float wheel_joint_torque_nm[RMUC_WHEEL_COUNT];
    float wheel_policy_torque_nm[RMUC_WHEEL_COUNT];
    float wheel_feedback_torque_nm[RMUC_WHEEL_COUNT];
    int32_t wheel_current_counts[RMUC_WHEEL_COUNT];
    bool wheel_above_continuous_reference[RMUC_WHEEL_COUNT];
    bool leg_pd_updated;
    uint64_t tick_1ms;
} RmucActuatorOutput;

typedef struct {
    RmucHardwareCalibration hardware;
    float action[RMUC_ACTION_DIM];
    float held_leg_target_rad[RMUC_LEG_COUNT];
    float held_leg_torque_nm[RMUC_LEG_COUNT];
    uint64_t tick_1ms;
    uint64_t last_policy_tick;
    bool has_action;
    bool hardware_ready;
} RmucLowLevelController;

/* The action is the executed 6D policy action after consume_action(). */
RmucControlStatus rmuc_low_level_init(RmucLowLevelController *controller,
                                      const RmucHardwareCalibration *hardware);
RmucControlStatus rmuc_low_level_set_action(RmucLowLevelController *controller,
                                            const float action[RMUC_ACTION_DIM]);
RmucControlStatus rmuc_low_level_step(RmucLowLevelController *controller,
                                      const RmucJointSample *sample,
                                      RmucActuatorOutput *output);

#endif
