from gymnasium.envs.registration import register

print('Registering the environments...')

register(
    id='VoltageControlEnv-v0',
    entry_point='voltage_control_env.env:VoltageControlEnv')

register(
    id ='DeltaStepVoltageControlEnv-v0',
    entry_point='voltage_control_env.env:DeltaStepVoltageControlEnv'
)