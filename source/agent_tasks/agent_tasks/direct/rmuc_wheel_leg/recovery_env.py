"""Independent recovery prototype. Does not alter the locomotion environment."""
import copy
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from .env import RmucWheelLegEnv
from .env_cfg import RmucWheelLegCourseEnvCfg
from .recovery_rewards import DEFAULT_WEIGHTS, compute_recovery_rewards, standing_recovery_terms, refall_event
from .recovery_rewards import (recovery_phase_progress, upright_without_side_income,
    supported_rise_shaping, stable_pose_shaping, joint_limit_barrier,
    contact_impact_penalty, linear_slew_schedule, supported_braking_cost,
    double_contact_acquired, advance_recovery_stage, supported_wheel_gap_progress, debounced_contact_event, recovery_cohort_layout)

# Geometric candidates audited in sim2sim/recovery_curriculum_candidates.json.
# Keep the historical 90-degree values exact; smaller tilts require native checks.
RECOVERY_START_HEIGHTS = {
    0: (0.4029930000000001, 0.4029930000000001, 0.4029930000000001, 0.4029930000000001),
    5: (0.4016965221602816, 0.401701925816332, 0.42206309792585656, 0.42212777561817677),
    15: (0.3913999960671011, 0.3914160428478974, 0.45063998238758834, 0.45070652973363684),

    30: (.3574270751088828,.3574580751088828,.4678440397355106,.46790959933594906),
    45: (.30338943554574105,.30343327616617466,.45336974396655466,.4534298480429555),
    60: (.23296965321248275,.2330233467875174,.40820349441747034,.4082540469763536),
    75: (.2145191860836161,.2144894355681664,.33542329635427104,.33546085232229084),
    90: (.1850853,.1850545,.239989,.240011),
}


@configclass
class RmucRecoveryEnvCfg(RmucWheelLegCourseEnvCfg):
    curriculum_stage_name = 'recovery_prototype'
    policy_outer_feedback_enabled = False
    position_authority_enabled = False
    domain_randomization_strength_override = 0.0
    recovery_communication_delay_max_steps = -1
    recovery_communication_delay_probability_override = -1.0
    wheel_torque_scale_nm = 5.5
    wheel_authored_effort_limit_nm = 5.5
    wheel_damping_anneal_target = -1.0
    wheel_damping_compensation_ratio = 0.0
    gas_spring_radial_force_n = 0.0
    gas_spring_model = 'equivalent_assist_joint_torque'
    gas_spring_assist_torque_nm = 10.775
    motor_response_enabled = False
    episode_length_s = 8.0
    recovery_hold_seconds = .5
    recovery_upright_degrees = 12.0
    recovery_min_height = .28
    # Fixed physical slope per run, independent of dynamics DR. Zero preserves
    # the existing flat collider and all flat-coordinate calculations.
    recovery_slope_degrees = 0.0
    recovery_slope_azimuth_degrees = 0.0
    # Independent process target; curriculum may use 0.25, 0.28 or 0.36 m.
    # recovery_min_height remains the strict stable gate for that run.
    recovery_rise_shaping_height = .36
    recovery_joint_limit_penalty = 1.0
    recovery_impact_penalty = .002
    recovery_braking_penalty = 0.0
    recovery_wheel_gap_progress_weight = 0.0
    recovery_contact_soft_limit_n = 300.
    # Opt-in coverage matching termination and the full-body contact audit.
    recovery_full_safety_scope = False
    recovery_max_planar_speed = .15
    recovery_max_angular_speed = .3
    recovery_joint_limit_tolerance_rad = 0.0
    # Optional output-side target slew limit for recovery experiments. The
    # default is disabled to preserve the historical action contract.
    recovery_leg_target_slew_rate_rad_s = float('inf')
    recovery_slew_start_rad_s = float('inf')
    recovery_slew_end_rad_s = float('inf')
    recovery_slew_anneal_steps = 0
    # In-training per-environment reset curriculum. Disabled by default so
    # historical checkpoint evaluation keeps its recorded fixed tilt.
    recovery_curriculum_enabled = False
    recovery_curriculum_tilt_stages = (0,5,15,30,45,60,75,90)
    recovery_curriculum_start_stage = 0
    # Raise only after independent four-direction evaluation and retention.
    recovery_curriculum_max_stage = 0
    recovery_curriculum_success_episodes = 8
    recovery_reset_cohorts = ()
    # Explicit profile prevents old checkpoint evaluation from silently using
    # the next experiment's reward. New trials select recovery_v1 via config.
    recovery_reward_profile = 'legacy_v0'
    recovery_progress_observation = False
    recovery_initial_tilt_degrees = 90
    recovery_reward_weights = dict(DEFAULT_WEIGHTS)
    recovery_standing_weights = {'supported_height':2.,'height_reached':2.,'stable_hold':3.,'stable_pose':4.,'double_contact':0.}
    # Sparse per-case overrides inherit the live global weights. Explicit
    # historical banks remain authoritative when restored from params.
    recovery_case_standing_weights = {}
    recovery_contact_event_mode = 'debounced_once'
    recovery_contact_hold_seconds = .04
    recovery_refall_penalty = 0.0
    recovery_refall_arm_seconds = .1
    recovery_refall_dwell_seconds = .08
    recovery_refall_height_m = .18
    recovery_refall_tilt_degrees = 60.

    def __post_init__(self):
        if self.recovery_initial_tilt_degrees not in RECOVERY_START_HEIGHTS:
            raise ValueError('Recovery tilt must be one of the audited 0/5/15/30/45/60/75/90-degree stages')
        if not math.isfinite(self.recovery_refall_penalty) or self.recovery_refall_penalty<0:
            raise ValueError('Refall penalty must be finite and nonnegative')
        if any(not math.isfinite(float(v)) or float(v)<0 for v in (
                self.recovery_joint_limit_penalty,self.recovery_impact_penalty,
                self.recovery_braking_penalty,self.recovery_wheel_gap_progress_weight)):
            raise ValueError('Recovery safety penalties must be finite and nonnegative')
        if not math.isfinite(self.recovery_contact_soft_limit_n) or self.recovery_contact_soft_limit_n<=0:
            raise ValueError('Recovery contact soft limit must be positive')
        if not all(math.isfinite(v) and v>0 for v in (self.recovery_refall_arm_seconds,self.recovery_refall_dwell_seconds)):
            raise ValueError('Refall timing must be finite and positive')
        if not 0<self.recovery_refall_height_m<self.recovery_min_height or not self.recovery_upright_degrees<self.recovery_refall_tilt_degrees<180:
            raise ValueError('Refall thresholds must provide hysteresis from standing')
        if (not math.isfinite(self.recovery_rise_shaping_height)
                or self.recovery_rise_shaping_height < .25):
            raise ValueError('Rise shaping height must be finite and at least 0.25 m')
        if not math.isfinite(self.recovery_joint_limit_tolerance_rad) or not 0. <= self.recovery_joint_limit_tolerance_rad <= 1.e-5:
            raise ValueError('Recovery numerical joint tolerance must be between 0 and 1e-5 rad')
        if not (math.isfinite(self.recovery_leg_target_slew_rate_rad_s)
                and self.recovery_leg_target_slew_rate_rad_s > 0.) and self.recovery_leg_target_slew_rate_rad_s != float('inf'):
            raise ValueError('Recovery target slew rate must be positive or infinity')
        if self.recovery_slew_anneal_steps < 0 or not isinstance(self.recovery_slew_anneal_steps,int):
            raise ValueError('Recovery slew anneal steps must be a nonnegative integer')
        for value in (self.recovery_slew_start_rad_s,self.recovery_slew_end_rad_s):
            if value != float('inf') and (not math.isfinite(value) or value<=0):
                raise ValueError('Recovery slew anneal rates must be positive or infinity')
        if self.recovery_slew_anneal_steps>0 and (
                not math.isfinite(self.recovery_slew_start_rad_s)
                or not math.isfinite(self.recovery_slew_end_rad_s)):
            raise ValueError('Enabled slew anneal requires finite start and end rates')
        stages=tuple(int(v) for v in self.recovery_curriculum_tilt_stages)
        if not stages or tuple(sorted(set(stages)))!=stages or any(a not in RECOVERY_START_HEIGHTS for a in stages):
            raise ValueError('Recovery curriculum must be an increasing sequence of audited angles')
        if not 0 <= self.recovery_curriculum_start_stage <= self.recovery_curriculum_max_stage < len(stages):
            raise ValueError('Recovery curriculum start stage is out of range')
        if not isinstance(self.recovery_curriculum_success_episodes,int) or self.recovery_curriculum_success_episodes<1:
            raise ValueError('Recovery curriculum success episodes must be a positive integer')
        if self.recovery_reward_profile not in ('legacy_v0','recovery_v1','recovery_v2','recovery_v3'):
            raise ValueError('Unknown recovery reward profile')
        if set(self.recovery_standing_weights)!={'supported_height','height_reached','stable_hold','stable_pose','double_contact'} or not all(
                math.isfinite(float(v)) and float(v)>=0 for v in self.recovery_standing_weights.values()):
            raise ValueError('Invalid recovery standing weights')
        names=('front_down','back_down','left_wheel_side_down','right_wheel_side_down')
        if not set(self.recovery_case_standing_weights)<=set(names):
            raise ValueError('Recovery case reward banks must cover all four starts')
        for bank in self.recovery_case_standing_weights.values():
            if not set(bank)<=set(self.recovery_standing_weights) or not all(math.isfinite(float(v)) and float(v)>=0 for v in bank.values()):
                raise ValueError('Invalid recovery case reward bank')
        if self.recovery_contact_event_mode not in ('legacy_edge','debounced_once'):
            raise ValueError('Unknown contact event mode')
        if not math.isfinite(self.recovery_contact_hold_seconds) or self.recovery_contact_hold_seconds<=0:
            raise ValueError('Contact hold must be finite and positive')
        if set(self.recovery_reward_weights)!=set(DEFAULT_WEIGHTS) or not all(
                math.isfinite(float(v)) for v in self.recovery_reward_weights.values()):
            raise ValueError('Invalid recovery reward weights')
        if self.sim.physics is None:
            from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
            self.sim.physics=PhysxCfg()
        self.sim.physics.enable_external_forces_every_iteration=True
        self.robot_cfg=copy.deepcopy(self.robot_cfg)
        self.robot_cfg.actuators['wheel_motors'].effort_limit=5.5
        self.robot_cfg.actuators['wheel_motors'].damping=.0225


class RmucRecoveryEnv(RmucWheelLegEnv):
    def _setup_scene(self):
        from .recovery_slope import ground_frame, plane_elevation
        slope=float(self.cfg.recovery_slope_degrees)
        azimuth=float(self.cfg.recovery_slope_azimuth_degrees)
        if not math.isfinite(slope) or abs(slope)>10 or not math.isfinite(azimuth):
            raise ValueError('Recovery slope must be finite within +/-10 degrees')
        self._recovery_ground_normal=None
        self._recovery_ground_quaternion=None
        if slope:
            from .flat_terrain import RmucSlopeTerrainImporter
            self.cfg.terrain=copy.deepcopy(self.cfg.terrain)
            self.cfg.terrain.class_type=RmucSlopeTerrainImporter
            self.cfg.terrain.recovery_slope_degrees=slope
            self.cfg.terrain.recovery_slope_azimuth_degrees=azimuth
        super()._setup_scene()
        if slope:
            normal,rotation=ground_frame(torch.tensor([slope],device=self.device),
                                        torch.tensor([azimuth],device=self.device))
            self._recovery_ground_normal=normal
            # Isaac Lab 3 math/root state uses xyzw; geometry helper is wxyz.
            self._recovery_ground_quaternion=rotation[:,[1,2,3,0]]
            origins=self.scene.env_origins
            origins[:,2]=plane_elevation(origins[:,:2],torch.zeros_like(origins),normal)

    def _recovery_ground_height(self,points):
        if self._recovery_ground_normal is None:
            return points[...,2]-self.scene.env_origins[:,2]
        from .recovery_slope import ground_clearance
        return ground_clearance(points,self.scene.env_origins,self._recovery_ground_normal)

    def _progressive_domain_randomization_strength(self):
        """Recovery uses an explicit experiment strength, not locomotion levels."""
        strength=float(self.cfg.domain_randomization_strength_override)
        if not math.isfinite(strength) or (strength != -1. and not 0.<=strength<=1.):
            raise ValueError('Recovery DR strength must be -1 (off) or within [0,1]')
        if (not self.cfg.domain_randomization_enabled
                or (getattr(self.cfg,'play',False) and not self.cfg.domain_randomization_apply_in_play)):
            return 0.
        return max(0.,strength)

    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self._recovery_hold=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self._recovery_case=torch.arange(self.num_envs,device=self.device)%4
        self._recovery_cohorts=None
        if cfg.recovery_reset_cohorts:
            if cfg.recovery_curriculum_enabled:
                raise ValueError('Explicit rehearsal cohorts cannot auto-promote')
            self._recovery_cohorts=recovery_cohort_layout(self.num_envs,cfg.recovery_reset_cohorts,device=self.device)
            self._recovery_case=self._recovery_cohorts[:,1].clone()
        self._recovery_previous_up=torch.zeros(self.num_envs,device=self.device)
        self._recovery_previous_valid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._recovery_previous_leg_target=None
        self._recovery_applied_leg_target=None
        self._recovery_refall_armed=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._recovery_refall_bad_steps=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self._recovery_previous_both=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._recovery_contact_dwell=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self._recovery_contact_paid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._recovery_resolved_banks={name:{**cfg.recovery_standing_weights,**cfg.recovery_case_standing_weights.get(name,{})}
            for name in ('front_down','back_down','left_wheel_side_down','right_wheel_side_down')}
        self._recovery_previous_wheel_gap=torch.zeros(self.num_envs,device=self.device)
        self._recovery_previous_supported=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._recovery_tilt_stage=torch.full((self.num_envs,),int(self.cfg.recovery_curriculum_start_stage),
            device=self.device,dtype=torch.long)
        self._recovery_success_episodes=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self._recovery_physical_steps=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)

    def _reset_idx(self,env_ids):
        if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
        if hasattr(self,'_recovery_tilt_stage') and self.cfg.recovery_curriculum_enabled:
            # Only completed timeouts with terminal continuous hold count.
            # Manual resets, transient success followed by refall, and unsafe
            # episodes never authorize promotion.
            complete=((self.episode_length_buf[env_ids]>=self.max_episode_length-1)
                      & (self._recovery_physical_steps[env_ids]>=self.max_episode_length-1))
            ready=self.recovery_metrics()[2][env_ids]
            success=(complete & ~self._estop[env_ids] & ready
                     & (self._recovery_hold[env_ids]*self.step_dt>=self.cfg.recovery_hold_seconds))
            stage,count=advance_recovery_stage(self._recovery_tilt_stage[env_ids],
                self._recovery_success_episodes[env_ids],success,
                self.cfg.recovery_curriculum_success_episodes,self.cfg.recovery_curriculum_max_stage)
            self._recovery_tilt_stage[env_ids]=stage
            self._recovery_success_episodes[env_ids]=count
        super()._reset_idx(env_ids)
        from .recovery_delay import cap_communication_delay
        if not hasattr(self,'_recovery_sampled_communication_delay_steps'):
            self._recovery_sampled_communication_delay_steps=self._communication_delay_steps.clone()
        self._recovery_sampled_communication_delay_steps[env_ids]=self._communication_delay_steps[env_ids]
        self._communication_delay_steps[env_ids]=cap_communication_delay(
            self._communication_delay_steps[env_ids],self.cfg.recovery_communication_delay_max_steps)
        if not hasattr(self,'_recovery_hold'):return
        case=self._recovery_case[env_ids]
        # Poses audited against this unchanged USD/MJCF topology. The offset
        # gives 3mm initial floor clearance; settling remains part of rollout.
        if self._recovery_cohorts is not None:
            tilt_values=self._recovery_cohorts[env_ids,0].float()
            height_table=torch.tensor([RECOVERY_START_HEIGHTS[a] for a in sorted(RECOVERY_START_HEIGHTS)],device=self.device)
            angle_table=torch.tensor(sorted(RECOVERY_START_HEIGHTS),device=self.device)
            stage=torch.searchsorted(angle_table,tilt_values.long())
            heights=height_table[stage,case]
        elif self.cfg.recovery_curriculum_enabled:
            stage=self._recovery_tilt_stage[env_ids]
            stage_angles=tuple(int(v) for v in self.cfg.recovery_curriculum_tilt_stages)
            height_table=torch.tensor([RECOVERY_START_HEIGHTS[a] for a in stage_angles],device=self.device)
            heights=height_table[stage,case]
            tilt_values=torch.tensor(stage_angles,device=self.device,dtype=torch.float32)[stage]
        else:
            heights=torch.tensor(RECOVERY_START_HEIGHTS[self.cfg.recovery_initial_tilt_degrees],device=self.device)[case]
            tilt_values=torch.full((len(env_ids),),float(self.cfg.recovery_initial_tilt_degrees),device=self.device)
        # Per-environment curriculum angles must remain aligned with the
        # four direction cases; construct signed roll/pitch after indexing.
        roll=torch.where((case==0)|(case==1),
                         torch.where(case==0,-torch.deg2rad(tilt_values),torch.deg2rad(tilt_values)),
                         torch.zeros_like(tilt_values))
        pitch=torch.where(case==2,torch.deg2rad(tilt_values),
                          torch.where(case==3,-torch.deg2rad(tilt_values),torch.zeros_like(tilt_values)))
        delta=quat_from_euler_xyz(roll,pitch,torch.zeros_like(roll))
        pose=self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:,:3]=self.scene.env_origins[env_ids]
        pose[:,2]+=heights
        pose[:,3:7]=quat_mul(delta,self._asset_frame_quat[env_ids])
        if self._recovery_ground_normal is not None:
            pose[:,:3]=self.scene.env_origins[env_ids]+heights[:,None]*self._recovery_ground_normal
            pose[:,3:7]=quat_mul(self._recovery_ground_quaternion.expand(len(env_ids),-1),pose[:,3:7])
        self.robot.write_root_pose_to_sim(pose,env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids),6,device=self.device),env_ids=env_ids)
        self._recovery_hold[env_ids]=0
        self._recovery_physical_steps[env_ids]=0
        self._recovery_refall_armed[env_ids]=False
        self._recovery_refall_bad_steps[env_ids]=0
        self._recovery_previous_both[env_ids]=False
        self._recovery_contact_dwell[env_ids]=0
        self._recovery_contact_paid[env_ids]=False
        self._recovery_previous_supported[env_ids]=False
        self._recovery_previous_wheel_gap[env_ids]=0.
        self._recovery_previous_valid[env_ids]=False
        if getattr(self,'_recovery_previous_leg_target',None) is not None:
            self._recovery_previous_leg_target[env_ids]=self.robot.data.joint_pos.torch[env_ids][:,self._leg_control_ids]
        if getattr(self,'_recovery_applied_leg_target',None) is not None:
            self._recovery_applied_leg_target[env_ids]=self.robot.data.joint_pos.torch[env_ids][:,self._leg_control_ids]

    def _apply_action(self):
        # Preserve spring and wheel torque execution. Replace only the staged
        # leg position target, before robot.write_data_to_sim executes IdealPD.
        super()._apply_action()
        limits=self.robot.data.joint_pos_limits.torch[:,self._leg_control_ids,:]
        low=limits[...,0]+self.cfg.leg_joint_limit_margin_rad
        high=limits[...,1]-self.cfg.leg_joint_limit_margin_rad
        action=self._policy_actions[:,[0,1,3,4]].clamp(-1.,1.)
        target=(low+high)/2.+(high-low)/2.*action
        rate=self.cfg.recovery_leg_target_slew_rate_rad_s
        if self.cfg.recovery_slew_anneal_steps>0:
            start=self.cfg.recovery_slew_start_rad_s
            end=self.cfg.recovery_slew_end_rad_s
            if math.isfinite(float(start)) and math.isfinite(float(end)):
                rate=linear_slew_schedule(start,end,self._global_step_counter+
                    getattr(self,'_recovery_slew_sample_offset',0),
                    self.cfg.recovery_slew_anneal_steps)
        if math.isfinite(float(rate)):
            if self._recovery_previous_leg_target is None:
                self._recovery_previous_leg_target=self.robot.data.joint_pos.torch[:,self._leg_control_ids].clone()
            max_delta=float(rate)*self.cfg.sim.dt
            target=self._recovery_previous_leg_target+(target-self._recovery_previous_leg_target).clamp(-max_delta,max_delta)
            self._recovery_previous_leg_target.copy_(target.detach())
        if self._recovery_applied_leg_target is None:
            self._recovery_applied_leg_target=target.detach().clone()
        self._recovery_applied_leg_target.copy_(target.detach())
        self.robot.set_joint_position_target(target,joint_ids=self._leg_control_ids)

    def recovery_metrics(self):
        gravity,linear,angular=self._state()
        up=(-gravity[:,2]).clamp(-1.,1.)
        height=self._recovery_ground_height(self.robot.data.root_pos_w.torch)
        forces=self.contact_sensor.data.net_forces_w[:,self._wheel_contact_sensor_ids]
        both=(torch.linalg.vector_norm(forces,dim=-1)>2.).all(dim=-1)
        ready=((up>math.cos(math.radians(self.cfg.recovery_upright_degrees)))
               &(height>self.cfg.recovery_min_height)
               &(torch.linalg.vector_norm(linear[:,:2],dim=-1)<self.cfg.recovery_max_planar_speed)
               &(torch.linalg.vector_norm(angular,dim=-1)<self.cfg.recovery_max_angular_speed)&both)
        return up,height,ready

    def _get_rewards(self):
        self._recovery_physical_steps+=1
        up,height,ready=self.recovery_metrics()
        _,linear,angular=self._state()
        planar_speed=torch.linalg.vector_norm(linear[:,:2],dim=-1)
        angular_speed=torch.linalg.vector_norm(angular,dim=-1)
        self._recovery_hold=torch.where(ready,self._recovery_hold+1,torch.zeros_like(self._recovery_hold))
        established=self._recovery_hold*self.step_dt>=self.cfg.recovery_hold_seconds
        effort=self.robot.data.applied_torque.torch[:,self._leg_control_ids]/40.
        safety_ids=self._dof_ids if self.cfg.recovery_full_safety_scope else self._leg_control_ids
        q=self.robot.data.joint_pos.torch[:,safety_ids]
        qlimits=self.robot.data.joint_pos_limits.torch[:,safety_ids,:]
        limit_cost=joint_limit_barrier(q,qlimits)
        wheel_forces=self.contact_sensor.data.net_forces_w[:,self._wheel_contact_sensor_ids]
        contact_forces=wheel_forces
        if self.cfg.recovery_full_safety_scope:
            contact_forces=self.contact_sensor.data.net_forces_w
        impact_cost=contact_impact_penalty(contact_forces,self.cfg.recovery_contact_soft_limit_n)
        # Full safety scope expands impact penalties, never wheel support.
        wheel_contact=(torch.linalg.vector_norm(wheel_forces,dim=-1)>2.).any(-1)
        both_contact=(torch.linalg.vector_norm(wheel_forces,dim=-1)>2.).all(-1)
        if self.cfg.recovery_contact_event_mode=='legacy_edge':
            double_contact=double_contact_acquired(self._recovery_previous_both,both_contact)
        else:
            self._recovery_contact_dwell,self._recovery_contact_paid,double_contact=debounced_contact_event(
                self._recovery_contact_dwell,self._recovery_contact_paid,both_contact & (up>0.),
                math.ceil(self.cfg.recovery_contact_hold_seconds/self.step_dt))
        self._recovery_previous_both.copy_(both_contact)
        braking_cost=supported_braking_cost(up,wheel_contact,planar_speed)
        wheel_height=self.robot.data.body_pos_w.torch[:,self._wheel_body_ids,2]
        if self._recovery_ground_normal is not None:
            from .recovery_slope import ground_clearance
            wheel_height=ground_clearance(self.robot.data.body_pos_w.torch[:,self._wheel_body_ids],
                self.scene.env_origins[:,None,:],self._recovery_ground_normal[:,None,:])
        wheel_gap=(wheel_height[:,0]-wheel_height[:,1]).abs()
        gap_progress=supported_wheel_gap_progress(self._recovery_previous_wheel_gap,wheel_gap,
            self._recovery_previous_valid,self._recovery_previous_supported,wheel_contact,
            self._recovery_previous_up,up,self._recovery_case>=2)
        gap_reward=self.cfg.recovery_wheel_gap_progress_weight*gap_progress
        self._recovery_previous_wheel_gap.copy_(wheel_gap)
        self._recovery_previous_supported.copy_(wheel_contact)
        rate=(self._policy_actions-self._previous_actions).square().mean(-1)
        if self.cfg.recovery_reward_profile=='legacy_v0':
            reward=3.*(up+1.)/2.+2.*(height/.4).clamp(0.,1.)+5.*established.float()-.1*rate-.05*effort.square().mean(-1)
        else:
            reward,terms=compute_recovery_rewards(up,height,established,
                self._recovery_previous_up,self._recovery_previous_valid,
                self._policy_actions,self._previous_actions,effort,self.cfg.recovery_reward_weights)
            if self.cfg.recovery_reward_profile=='recovery_v3':
                old=terms['upright']
                terms['upright']=float(self.cfg.recovery_reward_weights['upright'])*upright_without_side_income(up)
                reward=reward-old+terms['upright']
            if self.cfg.recovery_reward_profile in ('recovery_v2','recovery_v3'):
                forces=self.contact_sensor.data.net_forces_w[:,self._wheel_contact_sensor_ids]
                supported=(torch.linalg.vector_norm(forces,dim=-1)>2.).all(-1)
                extra=standing_recovery_terms(up,height,supported,ready,
                    self._recovery_hold*self.step_dt,self.cfg.recovery_min_height,
                    math.cos(math.radians(self.cfg.recovery_upright_degrees)),self.cfg.recovery_hold_seconds)
                if self.cfg.recovery_reward_profile=='recovery_v3':
                    extra['supported_height']=supported_rise_shaping(
                        up,height,supported,self.cfg.recovery_min_height,
                        self.cfg.recovery_rise_shaping_height)
                    extra['stable_pose']=stable_pose_shaping(
                        up,height,supported,planar_speed,angular_speed,
                        self.cfg.recovery_min_height,self.cfg.recovery_rise_shaping_height,
                        self.cfg.recovery_max_planar_speed,self.cfg.recovery_max_angular_speed)
                # Replace the old ungated height term; preserve other v1 terms.
                reward=reward-terms['upright_height']
                terms['upright_height']=torch.zeros_like(reward)
                for name,value in extra.items():
                    bank_names=('front_down','back_down','left_wheel_side_down','right_wheel_side_down')
                    bank=self._recovery_resolved_banks[bank_names[int(self._recovery_case[0])]] if self.num_envs==1 else None
                    if bank is None:
                        weights=torch.stack([torch.full_like(value,float(self._recovery_resolved_banks[n][name])) for n in bank_names])
                        env_index=torch.arange(self.num_envs,device=value.device)
                        weight=weights[self._recovery_case,env_index]
                    else:
                        weight=float(bank[name])
                    terms[name]=weight*value
                    reward=reward+terms[name]
                terms['double_contact']=double_contact*torch.ones_like(reward)
                bank_names=('front_down','back_down','left_wheel_side_down','right_wheel_side_down')
                bank_weight=torch.stack([torch.full_like(reward,float(self._recovery_resolved_banks[n]['double_contact'])) for n in bank_names])
                reward=reward+bank_weight[self._recovery_case,torch.arange(self.num_envs,device=reward.device)]*terms['double_contact']
            self.extras['recovery_reward']={name:value.mean() for name,value in terms.items()}
        self._recovery_previous_up.copy_(up)
        self._recovery_previous_valid.fill_(True)
        self._previous_actions.copy_(self._policy_actions)
        fallen=(height<self.cfg.recovery_refall_height_m)|(up<math.cos(math.radians(self.cfg.recovery_refall_tilt_degrees)))
        self._recovery_refall_armed,self._recovery_refall_bad_steps,refell=refall_event(
            self._recovery_refall_armed,self._recovery_refall_bad_steps,self._recovery_hold,fallen,
            math.ceil(self.cfg.recovery_refall_arm_seconds/self.step_dt),
            math.ceil(self.cfg.recovery_refall_dwell_seconds/self.step_dt))
        refall_cost=-self.cfg.recovery_refall_penalty*refell.float()
        reward=reward+refall_cost
        safety_cost=(-self.cfg.recovery_joint_limit_penalty*limit_cost
                     -self.cfg.recovery_impact_penalty*impact_cost
                     -self.cfg.recovery_braking_penalty*braking_cost)
        reward=reward+safety_cost+gap_reward
        self._estop|=~torch.isfinite(reward)
        self.extras['recovery']={'ready_fraction':ready.float().mean(),
            'established_fraction':established.float().mean()}
        # RSL-RL consumes episode/log only. Allocate a fresh dictionary each
        # step because its logger retains references until the PPO update.
        # These are rollout step means, not episode recovery success rates.
        self.extras['episode']={
            'Recovery/ready_step_fraction':ready.float().mean(),
            'Recovery/established_step_fraction':established.float().mean(),
            'Recovery/up_mean':up.mean(),
            'Recovery/root_height_mean_m':height.mean(),
            'Recovery/refall_event_fraction':refell.float().mean(),
            'RecoveryReward/refall':refall_cost.mean(),
            'RecoveryReward/joint_limit':(-self.cfg.recovery_joint_limit_penalty*limit_cost).mean(),
            'RecoveryReward/contact_impact':(-self.cfg.recovery_impact_penalty*impact_cost).mean(),
            'RecoveryReward/braking':(-self.cfg.recovery_braking_penalty*braking_cost).mean(),
            'Recovery/double_contact_event_fraction':double_contact.mean(),
            'RecoveryReward/wheel_gap_progress':gap_reward.mean(),
        }
        self.extras['episode'].update({
            f'Recovery/communication_delay_{n}_fraction':(self._communication_delay_steps==n).float().mean()
            for n in range(4)})
        if self.cfg.recovery_curriculum_enabled:
            stages=torch.tensor(self.cfg.recovery_curriculum_tilt_stages,device=self.device,dtype=torch.float32)
            self.extras['episode'].update({
                'Recovery/curriculum_tilt_mean':stages[self._recovery_tilt_stage].mean(),
                'Recovery/curriculum_stage_mean':self._recovery_tilt_stage.float().mean(),
                'Recovery/curriculum_promotable_fraction':(
                    (self._recovery_success_episodes>=self.cfg.recovery_curriculum_success_episodes-1).float().mean()),
            })
        if self.cfg.recovery_reward_profile in ('recovery_v1','recovery_v2','recovery_v3'):
            self.extras['episode'].update({
                f'RecoveryReward/{name}':value.mean() for name,value in terms.items()})
        return torch.nan_to_num(reward,nan=-10.,posinf=-10.,neginf=-10.)

    def _get_observations(self):
        observations=super()._get_observations()
        if self.cfg.recovery_progress_observation:
            up,height,ready=self.recovery_metrics()
            hold=getattr(self,'_recovery_hold',torch.zeros_like(up))
            progress=recovery_phase_progress(up,height,ready,hold*self.step_dt,
                self.cfg.recovery_min_height,math.cos(math.radians(self.cfg.recovery_upright_degrees)),
                self.cfg.recovery_hold_seconds)
            observations['policy']=observations['policy'].clone()
            # Recovery-only schema: replace zero target-vx slot. Other 24
            # elements and locomotion semantics remain unchanged.
            observations['policy'][:,7]=progress
        return observations

    def _get_dones(self):
        # A fallen attitude is the task's initial state, not a termination.
        # Keep joint/nonfinite/escape guards independent of locomotion tilt.
        _,linear,angular=self._state()
        root=self.robot.data.root_pos_w.torch
        q=self.robot.data.joint_pos.torch[:,self._dof_ids]
        reasons=dict(nonfinite=(~torch.isfinite(root).all(-1)|~torch.isfinite(q).all(-1)
                    |~torch.isfinite(linear).all(-1)|~torch.isfinite(angular).all(-1)),
            joint_limit=((q<self._joint_pos_limits[...,0]-self.cfg.recovery_joint_limit_tolerance_rad)
                         |(q>self._joint_pos_limits[...,1]+self.cfg.recovery_joint_limit_tolerance_rad)).any(-1),
            tilt=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device),
            root_low=self._recovery_ground_height(root)<.02,
            root_high=self._recovery_ground_height(root)>self.cfg.max_height_m,
            xy_distance=torch.linalg.vector_norm(root[:,:2]-self.scene.env_origins[:,:2],dim=-1)>self.cfg.max_xy_distance_m,
            linear_speed=torch.linalg.vector_norm(linear,dim=-1)>self.cfg.max_velocity_mps)
        self._last_termination_reasons=reasons
        self._estop|=torch.stack(list(reasons.values())).any(0)
        return self._estop,self.episode_length_buf>=self.max_episode_length-1
