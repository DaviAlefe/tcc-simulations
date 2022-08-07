from sw_simulation import SWSimulation

import asyncio
import jax.numpy as jnp

async def run_simulation(w0_mult):
    sim = SWSimulation(w_0_mult=w0_mult)
    await sim.run()

if __name__ == '__main__':
    # asynchronously run the simulation for a range of w_0_mult values from 0.1 to 1.0
    w0_mult_range = jnp.linspace(0.1, 1.0, num=10)
    tasks = asyncio.gather(*[run_simulation(w0_mult) for w0_mult in w0_mult_range])
