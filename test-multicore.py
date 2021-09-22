import numpy as np
import paraPropPython as ppp
from paraPropPython import receiver as rx
import util
import time
import multiprocessing as mpl


def southpole(z):
    A = 1.78
    B = -0.43
    C = -0.0132
    return A + B * np.exp(C * z)

#if __name__ == "__main__":
def make_simul(sourceDepth, sim0, rxList0, dt0, freq0, sig0):
    sim0.set_dipole_source_profile(freq0, sourceDepth)
    sim0.set_td_source_signal(sig0, dt0)
    tstart = time.time()
    sim0.do_solver(rxList0)
    tend = time.time()
    print(rxList0[0].get_signal(), 'solver time:', tend-tstart)
    #return sim.get_field(receiverRange, sourceDepth)

def main():
    # Example from https://tutorialedge.net/python/python-multiprocessing-tutorial/
    iceDepth = 200.  # m
    iceLength = 100.  # m
    dx = 1  # m
    dz = 0.05  # m

    freq = 0.2  # GHz
    receiverRange = 50.
    sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=50.)
    sim.set_n('func', nFunc=southpole)

    dt = 1
    impulse = np.zeros(2 ** 8, dtype='complex')
    impulse[10] = 1 + 0j
    sig = util.normToMax(util.butterBandpassFilter(impulse, 0.09, 0.25, 1 / dt, 4))

    nCpus = mpl.cpu_count()
    max_depth = iceDepth

    nSources = nCpus
    source_depth_list = max_depth*np.random.random_sample(nSources)
    print(source_depth_list, len(source_depth_list))
    """
    with mpl.Pool(nCpus) as p:
        print(p.map(make_simul, source_depth_list))
    """
    print('Series')
    tstart = time.time()
    for i in range(nSources):
        rxList = [rx(receiverRange, source_depth_list[i])]
        make_simul(source_depth_list[i], sim, rxList, dt, freq, sig)
    tend = time.time()
    time_series = tend - tstart
    print('script time', time_series,'\n')

    """
    print('Pool')
    tstart = time.time()
    with mpl.Pool(nSources) as p:
        p.map(make_simul, source_depth_list)
    tend = time.time()
    """


    print('Process')
    tstart = time.time()

    processes = []
    for i in range(nSources):
        sourceDepth_i = source_depth_list[i]
        rxList = [rx(receiverRange, source_depth_list[i])]
        p = mpl.Process(target=make_simul, args=(sourceDepth_i, sim, rxList, dt, freq, sig))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    tend = time.time()
    time_parallel = tend - tstart
    print('script time', time_parallel, '\n')
    print('Speed ratio', time_series / time_parallel)

if __name__ == "__main__":
    main()
