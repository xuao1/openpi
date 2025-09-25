import logging
import threading
import time
import queue
import tree
import numpy as np

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0

        self.action_queue = queue.Queue()
        self.action_queue_lock = threading.Lock()
        self._action_thread = None
        self._stop_action_thread = threading.Event()

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False
        self._stop_action_thread.set()

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        self._stop_action_thread.clear()
        self._action_thread = threading.Thread(target=self._action_worker)
        self._action_thread.start()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
        
        if self._action_thread is not None:
            self._action_thread.join()
            self._action_thread = None

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _action_worker(self):
        while not self._stop_action_thread.is_set():
            try:
                observation = self._environment.get_observation()
                action_block = self._agent.get_action(observation)
                
                action_list = self._split_action_block(action_block)

                with self.action_queue_lock:
                    while not self.action_queue.empty():
                        try:
                            self.action_queue.get_nowait()
                        except queue.Empty:
                            break

                    for action in action_list:
                        self.action_queue.put((observation, action))
            except Exception as e:
                logging.error(f"Error in action worker: {e}")
                self._stop_action_thread.set()

    def _split_action_block(self, action_block: dict) -> list:
        if not action_block:
            return []
    
        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[_cur_step, ...]
            else:
                return x
                
        _cur_step = 0
        action_list = []
        while _cur_step < 10:   # assuming action_horizon is 10
            action = tree.map_structure(slicer, action_block)
            action_list.append(action)
            _cur_step += 1

        return action_list

    def _step(self) -> None:
        """A single step of the runtime loop."""
        action = None
        while action is None:
            try:
                with self.action_queue_lock:
                    observation, action = self.action_queue.get_nowait()
            except queue.Empty:
                if self._stop_action_thread.is_set():
                    logging.warning("Action thread has been stopped, exiting step.")
                    self.mark_episode_complete()
                    return
                time.sleep(0.01)  # wait for action to be available
        
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
