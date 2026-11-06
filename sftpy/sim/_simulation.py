import multiprocessing as mp

from ._actor import Actor
from ._data import Data
from ._event import Event
from ._timeline import Timeline
from .constants import NUM_THREADS



class Simulation:

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 sequence: list[Actor],
                 nthreads: int = NUM_THREADS):
        self._time = time
        self._data = data
        self._sequence = sequence
        self._nthreads = nthreads
        if nthreads > 1:
            self._pool = mp.Pool(nthreads)

    def run(self):
        for _ in range(self._time.nstep):
            for a in self._sequence:
                a.simulate()

    def register_event(self, event: Event):
        ct = self._sequence.count(event)
        if ct == 0:
            print("Event not found in Actor sequence.")
            return

        i_evt = 0
        for _ in range(ct):

            i_evt = self._sequence.index(event, i_evt)

            def simulate(self_actor):
                if event.trigger():
                    event.simulate()
                if event.terminate():
                    self._sequence.remove(self_actor)

            # create anonymous Actor wrapper class for each Event
            event_actor = type("", (Actor,), {"simulate": simulate})()

            # replace Event with wrapper EventActor
            self._sequence[i_evt] = event_actor
