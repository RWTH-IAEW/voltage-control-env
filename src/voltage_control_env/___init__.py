from gymnasium.envs.registration import register

register(
    id='VoltageControlEnv-v0',
    entry_point='voltage_control_env.voltage_control_env:VoltageControlEnv')

register(
    id ='DeltaStepVoltageControlEnv-v0',
    entry_point='voltage_control_env.voltage_control_env:DeltaStepVoltageControlEnv'
)