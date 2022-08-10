from sw_simulation import SWSimulation

import asyncio
import jax.numpy as jnp

def run_simulation(w0_mult):
    print(f'{w0_mult} started')
    sim = SWSimulation(w_0_mult=w0_mult)
    sim.run()
    print(f'{w0_mult} finished')

async def main():
    # asynchronously run the simulation for a range of w_0_mult values from 0.1 to 1.0
    w0_mult_range = jnp.linspace(0.1, 1.0, num=10)
    tasks = await asyncio.gather(*[run_simulation(w0_mult) for w0_mult in w0_mult_range])
    print(tasks)

if __name__ == '__main__':
    asyncio.run(main())