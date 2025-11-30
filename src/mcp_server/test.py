#!/usr/bin/env python3
"""Check if MCP server is healthy."""

import subprocess
import sys
import time

def check_mcp_server(workspace_path: str):
    """Test if MCP server can start and respond."""
    
    # Start server
    proc = subprocess.Popen(
        ["bash", "run_mcp_server_training.sh"],
        env={"WORKSPACE_PATH": workspace_path},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait a bit
    time.sleep(5)
    
    # Check if still running
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        print("MCP server died!")
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return False
    
    # Kill it
    proc.terminate()
    proc.wait()
    
    return True

if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test"
    if check_mcp_server(workspace):
        print("✓ MCP server healthy")
    else:
        print("✗ MCP server unhealthy")
        sys.exit(1)