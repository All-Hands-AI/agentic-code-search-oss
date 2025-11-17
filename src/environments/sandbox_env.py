import atexit
import signal
import time
from typing import Any

import verifiers as vf

import asyncio
import os
import subprocess
import uuid


class _LocalDockerClient:
    """Minimal async wrapper around docker CLI for container lifecycle and exec."""

    def __init__(self, executable: str = os.getenv("DOCKER_EXECUTABLE", "docker")):
        self.executable = executable

    async def create(
        self,
        *,
        name: str,
        image: str,
        start_command: str,
        workdir: str = "/",
        environment_vars: dict[str, str] | None = None,
    ):
        container_name = name or f"sandbox-{uuid.uuid4().hex[:8]}"
        cmd: list[str] = [
            self.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "--entrypoint",
            "",
            "-w",
            workdir,
        ]
        for key, value in (environment_vars or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([image, start_command])

        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        container_id = result.stdout

        return container_id

    async def execute_command(self, container_id: str, command: str) -> tuple[str, str]:
        exec_cmd = [
            self.executable,
            "exec",
            container_id,
            "sh",
            "-c",
            command,
        ]
        proc = await asyncio.to_thread(
            subprocess.run,
            exec_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout or "", proc.stderr or "")

    async def delete(self, container_id: str) -> None:
        # Force remove the container; ignore errors.
        await asyncio.to_thread(
            subprocess.run,
            [self.executable, "rm", "-f", container_id],
            capture_output=True,
            text=True,
            check=False,
        )


class SandboxEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        sandbox_name: str = "sandbox-env",
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        environment_vars: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._docker = _LocalDockerClient()
        self._sandbox_name = sandbox_name
        self._docker_image = docker_image
        self._start_command = start_command
        self._environment_vars = environment_vars or {}
        self._shared_sandbox_id: str | None = None
        self._ensure_lock = asyncio.Lock()

        # Install handlers for regular exception, sigint (Ctrl-C) and sigterm (standard termination signal)
        atexit.register(self.cleanup_sandboxes)
        signal.signal(
            signal.SIGINT,
            lambda sig, frame: (
                self.cleanup_sandboxes(),
                signal.default_int_handler(sig, frame),
            ),
        )
        signal.signal(
            signal.SIGTERM, lambda _, __: (self.cleanup_sandboxes(), exit(143))
        )

        self.add_tool(self.bash, args_to_skip=["sandbox_id"])

    async def bash(self, command: str, sandbox_id: str) -> str:
        """Execute `command` inside persistent sandbox container."""
        # sandbox_id is passed via update_tool_args, not seen by model
        s = time.time()
        self.logger.debug(f"Executing command {command} in sandbox {sandbox_id}")
        stdout, stderr = await self._docker.execute_command(sandbox_id, command)
        e = time.time()
        stdout = stdout.strip()
        stderr = stderr.strip()
        combined = stdout
        if stderr:
            if combined:
                combined = f"{combined}\nstderr:\n{stderr}"
            else:
                combined = f"stderr:\n{stderr}"
        output = combined or "(no output)"
        self.logger.debug(f"Executed command in {e - s:.1f}s. Got output: {output}")
        return output

    async def post_rollout(self, messages: vf.Messages, state: vf.State, **kwargs):
        """
        Override for custom post-rollout logic. If sandbox state is needed for reward functions,
        run computation here and cache the result in state. The shared sandbox persists across rollouts.
        """
        pass

    async def _ensure_shared_container(self) -> str:
        async with self._ensure_lock:
            if self._shared_sandbox_id is not None:
                return self._shared_sandbox_id
            name = f"{self._sandbox_name}-{uuid.uuid4().hex[:8]}"
            sandbox_id = await self._docker.create(
                name=name,
                image=self._docker_image,
                start_command=self._start_command,
                workdir="/",
                environment_vars=self._environment_vars,
            )
            self._shared_sandbox_id = sandbox_id
            self.logger.debug(f"Created shared sandbox {sandbox_id}")
            return sandbox_id

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Attach shared sandbox to per-rollout state (no per-rollout container)."""
        shared_id = await self._ensure_shared_container()
        state["sandbox_id"] = shared_id
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        if tool_name == "bash":
            updated_args = dict(tool_args)
            updated_args["sandbox_id"] = state["sandbox_id"]
            return updated_args
        else:
            return tool_args

    async def is_completed(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> bool:
        """
        When overriding, if sandbox state is needed for reward functions,
        run computation here and cache the result in state.
        """
        completed = await super().is_completed(messages, state, **kwargs)
        if completed:
            await self.post_rollout(messages, state, **kwargs)
        return completed

    def cleanup_sandboxes(self):
        """Delete the shared sandbox container"""
        if self._shared_sandbox_id is None:
            return
        try:
            self.logger.info(f"Cleaning up shared sandbox {self._shared_sandbox_id}")
            subprocess.run(
                [os.getenv("DOCKER_EXECUTABLE", "docker"), "rm", "-f", self._shared_sandbox_id],
                capture_output=True,
                text=True,
                check=False,
            )
            self.logger.debug(f"Successfully deleted sandbox {self._shared_sandbox_id}")
            self._shared_sandbox_id = None
        except Exception as e:
            self.logger.error(f"Failed to delete sandbox {self._shared_sandbox_id}: {e}")
