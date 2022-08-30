from sw_simulation import SWSimulation

import asyncio
import jax.numpy as jnp

def run_simulation(w0_mult, s_id):
    print(f'{w0_mult} started')
    sim = SWSimulation(w_0_mult=w0_mult, simulation_id=s_id)
    sim.run()
    print(f'{w0_mult} finished')

async def main():
    # asynchronously run the simulation for a range of w_0_mult values from 0.1 to 1.0
    w0_mult_range = [0.1, 0.3, 0.5,0.7,0.]#[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]#jnp.linspace(0.0, 1.0, num=20)
    tasks = await asyncio.gather(*[run_simulation(w0_mult) for w0_mult in w0_mult_range])
    print(tasks)

if __name__ == '__main__':
    # asyncio.run(main())
    run_simulation(0.3, s_id='test')