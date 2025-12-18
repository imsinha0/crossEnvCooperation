import jax
from jaxmarl.environments.overcooked import make_cramped_room_9x9
from jaxmarl import make

rng = jax.random.PRNGKey(42)
environments = []

for i in range(3):
    rng, subkey = jax.random.split(rng)
    layout_dict = make_cramped_room_9x9(subkey)
    env = make('overcooked', layout=layout_dict)
    environments.append((f"cramped_room_var_{i}", env))

print(environments)
