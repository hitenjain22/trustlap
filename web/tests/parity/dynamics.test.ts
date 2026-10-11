import { expect, it } from 'vitest'
import { controlStep } from '../../src/sim/dynamics.ts'
import { carFrom, condFrom, loadConfig, loadGolden, Tolerance } from './helpers.ts'

it('dynamics parity: 2,000 random control steps (|a − b| ≤ 1e-4 + 1e-4·|a|)', () => {
  const cfg = loadConfig()
  const g = loadGolden('dynamics')
  const tol = new Tolerance()
  for (let i = 0; i < g.n; i++) {
    const car = carFrom(g.state, i)
    const cond = condFrom(g.conditions, i)
    const [aSteer, aLong] = g.action[i]
    const [nx, ny] = g.gust_noise[i]
    controlStep(car, cond, aSteer, aLong, nx, ny, cfg)
    const e = carFrom(g.next_state, i)
    tol.check(`${i}.x`, e.xM, car.xM)
    tol.check(`${i}.y`, e.yM, car.yM)
    tol.check(`${i}.yaw`, e.yawRad, car.yawRad, true)
    tol.check(`${i}.vx`, e.vxMps, car.vxMps)
    tol.check(`${i}.vy`, e.vyMps, car.vyMps)
    tol.check(`${i}.r`, e.yawRateRadps, car.yawRateRadps)
    tol.check(`${i}.steer`, e.steerRad, car.steerRad)
    tol.check(`${i}.odometer`, e.odometerM, car.odometerM)
    e.actionBuffer.forEach((v, k) => tol.check(`${i}.buffer[${k}]`, v, car.actionBuffer[k]!))
    e.gustMps.forEach((v, k) => tol.check(`${i}.gust[${k}]`, v, car.gustMps[k]!))
  }
  expect(tol.failures).toEqual([])
})
