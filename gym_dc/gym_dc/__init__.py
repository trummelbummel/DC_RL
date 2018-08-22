
from gym.envs.registration import register

register(
   	id='DCEnv-v0',
   	entry_point='gym_dc.envs.distributioncenter_env:DCEnv',
)
