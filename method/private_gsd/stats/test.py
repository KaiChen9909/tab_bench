import jax
import jax.numpy as jnp



key = jax.random.PRNGKey(0)

A = jax.random.uniform(key, shape=(10, ))


J = jnp.array([
    0, 10,
    10, 100,
    100, 150,
    150, 200,
    200, 300,
    300, 350,
    350, 400,
    400, 800,
    800, 1000
])



print(jnp.split(A,jnp.array( [2, 7])))
# print(A[J[:,0]:J[:,1]])