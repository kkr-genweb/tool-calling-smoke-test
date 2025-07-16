"""
Filesystem Domain (Gorilla FS)
==============================

Mock filesystem API for testing agent interactions with file systems.
"""

from typing import Dict, List, Optional

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class GorillaFSClient(BaseDomainClient):
    """Mock filesystem API client."""
    
    def __init__(self, logger: CallLogger, *, files: Optional[Dict[str, str]] = None) -> None:
        super().__init__(logger)
        self.cwd = "/"
        self.files = files or {
            "/readme.txt": "Hello world", 
            "/data/trips.csv": "trip_id,cost\n1,100\n2,200\n3,300\n", 
            "/etc/passwd": "root:x:0:0:root:/root:/bin/bash"
        }

    @property
    def domain_name(self) -> str:
        return "gfs"

    async def ls(self, path: str | None = None) -> List[str]:
        """List files in directory."""
        p = path or self.cwd
        listed = sorted([k for k in self.files if k.startswith(p)])
        return self.logger.log("gfs", "ls", {"path": p}, listed)

    async def cd(self, path: str) -> str:
        """Change directory."""
        self.cwd = path
        return self.logger.log("gfs", "cd", {"path": path}, path)

    async def cat(self, path: str) -> str:
        """Read file contents."""
        data = self.files.get(path, "")
        return self.logger.log("gfs", "cat", {"path": path}, data)


def register_filesystem_tools(agent, deps_type):
    """Register filesystem tools with the agent."""
    
    @agent.tool
    async def gfs_ls(ctx: RunContext[deps_type], path: str | None = None) -> List[str]:
        """List files in directory."""
        return await ctx.deps.gfs.ls(path)

    @agent.tool
    async def gfs_cd(ctx: RunContext[deps_type], path: str) -> str:
        """Change directory."""
        return await ctx.deps.gfs.cd(path)

    @agent.tool
    async def gfs_cat(ctx: RunContext[deps_type], path: str) -> str:
        """Read file contents."""
        return await ctx.deps.gfs.cat(path)