import traceback
import multiprocessing as mp


def _worker_loop(
    repair_module_name,
    env_builder_factory_module_name,
    env_builder_factory_func_name,
    env_builder_cfg,
    task_queue,
    result_queue,
):
    """Worker process: build envs from the config and run ortools_repair_fn on queued tasks."""
    try:
        import importlib

        repair_module = importlib.import_module(repair_module_name)
        ortools_repair_fn = getattr(repair_module, "ortools_repair_fn")
        factory_module = importlib.import_module(env_builder_factory_module_name)
        build_env_builder = getattr(factory_module, env_builder_factory_func_name)
        env_builder = build_env_builder(env_builder_cfg)
    except Exception:
        result_queue.put(("error", "worker init failed", traceback.format_exc()))
        return

    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            graph_df, init_actions, repair_kwargs = task
            env_for_repair = env_builder(graph_df)
            repaired_actions, info = ortools_repair_fn(
                env=env_for_repair,
                init_actions=init_actions,
                **repair_kwargs,
            )
            result_queue.put(("ok", repaired_actions, info))
        except Exception as e:
            result_queue.put(("error", str(e), traceback.format_exc()))


class _WorkerHandle:
    """One worker process and its task/result queues."""

    def __init__(self, ctx, spawn_args):
        self._ctx = ctx
        self._spawn_args = spawn_args
        self.process = None
        self.task_queue = None
        self.result_queue = None
        self.tasks_done = 0
        self.cold = True


    def _ensure_queues(self):
        if self.task_queue is None:
            self.task_queue = self._ctx.Queue()
        if self.result_queue is None:
            self.result_queue = self._ctx.Queue()


    def _discard_queues(self):
        for name in ("task_queue", "result_queue"):
            q = getattr(self, name)
            if q is not None:
                for op in (q.cancel_join_thread, q.close, q.join_thread):
                    try:
                        op()
                    except Exception:
                        pass
            setattr(self, name, None)


    def is_alive(self):
        return self.process is not None and self.process.is_alive()


    def spawn(self):
        self._ensure_queues()
        self.process = self._ctx.Process(
            target=_worker_loop,
            args=(*self._spawn_args, self.task_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()
        self.tasks_done = 0
        self.cold = True


    def shutdown(self, graceful=True):
        if self.process is None:
            if not graceful:
                self._discard_queues()
            return

        had_to_kill = False
        try:
            if graceful and self.process.is_alive():
                try:
                    self.task_queue.put_nowait(None)
                except Exception:
                    pass
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                had_to_kill = True
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
                had_to_kill = True
        finally:
            self.process = None
            if had_to_kill or not graceful:
                self._discard_queues()


class IsolatedOrtoolsRepair:
    """Pool of repair workers used by the HG trainers."""

    def __init__(
        self,
        repair_module_name: str,
        env_builder_factory_module_name: str,
        env_builder_factory_func_name: str,
        env_builder_cfg: dict,
        timeout_buffer_s: float = 30.0,
        worker_recycle_every: int = 100,
        num_workers: int = 1,
    ):
        self.timeout_buffer_s = float(timeout_buffer_s)
        self.worker_recycle_every = int(worker_recycle_every)
        self.num_workers = max(1, int(num_workers))

        ctx = mp.get_context("spawn")
        spawn_args = (
            repair_module_name,
            env_builder_factory_module_name,
            env_builder_factory_func_name,
            env_builder_cfg,
        )
        self._workers = [_WorkerHandle(ctx, spawn_args) for _ in range(self.num_workers)]


    def _ensure_worker(self, w: _WorkerHandle):
        if not w.is_alive():
            w.shutdown(graceful=False)
            w.spawn()
        # Recycle the worker to release the memory OR-Tools accumulates
        elif w.tasks_done >= self.worker_recycle_every:
            w.shutdown(graceful=True)
            w.spawn()


    def close(self):
        for w in self._workers:
            w.shutdown(graceful=True)
            w._discard_queues()


    def __del__(self):
        try:
            for w in self._workers:
                w.shutdown(graceful=False)
        except Exception:
            pass


    def repair_batch(self, items, time_limit_s, **repair_kwargs):
        n = len(items)
        if n == 0:
            return []

        results = [None] * n
        assignments = [[] for _ in range(self.num_workers)]

        used_workers = min(self.num_workers, n)
        for wi in range(used_workers):
            self._ensure_worker(self._workers[wi])

        full_kwargs = dict(repair_kwargs)
        full_kwargs["time_limit_s"] = time_limit_s
        for k, (graph_df, init_actions) in enumerate(items):
            wi = k % used_workers
            w = self._workers[wi]
            w.task_queue.put((graph_df, list(init_actions), full_kwargs))
            assignments[wi].append((k, list(init_actions)))

        for wi in range(used_workers):
            w = self._workers[wi]
            dead = False
            for j, (orig_idx, init_actions) in enumerate(assignments[wi]):
                if dead:
                    results[orig_idx] = (
                        list(init_actions),
                        {"status": "subprocess_timeout_or_crash",
                         "used_time_limit_s": time_limit_s},
                    )
                    continue

                timeout = time_limit_s + self.timeout_buffer_s
                if w.cold and j == 0:
                    timeout += 30.0

                try:
                    status, a, b = w.result_queue.get(timeout=timeout)
                except Exception:
                    w.shutdown(graceful=False)
                    dead = True
                    results[orig_idx] = (
                        list(init_actions),
                        {"status": "subprocess_timeout_or_crash",
                         "used_time_limit_s": time_limit_s},
                    )
                    continue

                w.cold = False
                w.tasks_done += 1

                if status == "error":
                    if a == "worker init failed":
                        w.shutdown(graceful=False)
                        dead = True
                    results[orig_idx] = (
                        list(init_actions),
                        {"status": "subprocess_error", "error": a, "traceback": b,
                         "used_time_limit_s": time_limit_s},
                    )
                else:
                    results[orig_idx] = (a, b)

        return results
