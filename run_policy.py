#!/usr/bin/env python3
"""
Generic policy runner script for DroneVLA.

This script can run any policy node from the policy_nodes/ directory
and forwards all arguments to the selected policy.

Usage:
    # List available policies
    python run_policy.py --list
    
    # Run a specific policy with its arguments
    python run_policy.py queued --task gate_nav --horizon 10
    python run_policy.py replay --dataset_path /path/to/data --episode_id 5
    python run_policy.py human --rate_hz 15
    
    # Get help for a specific policy
    python run_policy.py queued --help
"""

import sys
import argparse
from pathlib import Path
import importlib.util


# Map of policy aliases to their module names
POLICY_MAP = {
    "gello": "gello_node",
    "replay": "policy_node_replay",
    "rtc": "policy_node_rtc2",
    "rtc2": "policy_node_rtc2",
    "rtc_viz": "policy_node_rtc_viz",
    "base": "policy_node3",
    "base3": "policy_node3",
    "act": "policy_node_act",
    "openloop": "policy_node_openloop",
}


def list_policies():
    """List all available policy nodes."""
    print("=" * 80)
    print("Available Policy Nodes")
    print("=" * 80)
    
    policy_dir = Path(__file__).parent / "policy_nodes"
    
    for alias, module_name in sorted(POLICY_MAP.items()):
        module_path = policy_dir / f"{module_name}.py"
        exists = "✓" if module_path.exists() else "✗"
        print(f"  {exists} {alias:<12} → {module_name}.py")
    
    print("\n" + "=" * 80)
    print("Usage Examples")
    print("=" * 80)
    print("  # Run queued policy with task")
    print("  python run_policy.py queued --task gate_nav --horizon 10")
    print()
    print("  # Run replay policy")
    print("  python run_policy.py replay --dataset_path /path/to/data --episode_id 5")
    print()
    print("  # Get help for a specific policy")
    print("  python run_policy.py queued --help")
    print()


def load_policy_module(policy_name: str):
    """
    Dynamically load a policy module.
    
    Args:
        policy_name: Name or alias of the policy to load
        
    Returns:
        Loaded module object
    """
    # Resolve alias to module name
    if policy_name in POLICY_MAP:
        module_name = POLICY_MAP[policy_name]
    elif policy_name.startswith("policy_node_"):
        module_name = policy_name
    else:
        module_name = f"policy_node_{policy_name}"
    
    # Construct path to module
    policy_dir = Path(__file__).parent / "policy_nodes"
    module_path = policy_dir / f"{module_name}.py"
    
    if not module_path.exists():
        # Try without policy_node_ prefix
        module_path = policy_dir / f"{policy_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(
                f"Policy module not found: {policy_name}\n"
                f"Tried: {module_name}.py and {policy_name}.py\n"
                f"Use --list to see available policies."
            )
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for: {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module, module_path


def main():
    # Create parser for top-level arguments
    parser = argparse.ArgumentParser(
        description="Run DroneVLA policy nodes with argument forwarding",
        add_help=False  # We'll handle help ourselves
    )
    parser.add_argument(
        "policy",
        nargs="?",
        help="Policy to run (queued, replay, human, debug, wait, time_sync, base)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available policy nodes"
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show help message"
    )
    
    # Parse only known args to get the policy name
    args, remaining_args = parser.parse_known_args()
    
    # Handle --list
    if args.list:
        list_policies()
        sys.exit(0)
    
    # Handle --help without policy specified
    if args.help and not args.policy:
        parser.print_help()
        print()
        list_policies()
        sys.exit(0)
    
    # Check if policy is specified
    if not args.policy:
        print("Error: No policy specified\n")
        parser.print_help()
        print()
        list_policies()
        sys.exit(1)
    
    # Load the policy module
    try:
        policy_module, module_path = load_policy_module(args.policy)
        print(f"✓ Loaded: {module_path.name}")
    except (FileNotFoundError, ImportError) as e:
        print(f"❌ Error: {e}")
        print()
        list_policies()
        sys.exit(1)
    
    # Check if module has a main function
    if not hasattr(policy_module, "main"):
        print(f"❌ Error: {module_path.name} does not have a main() function")
        sys.exit(1)
    
    # Prepare sys.argv for the policy module
    # Format: [script_name, *remaining_args]
    original_argv = sys.argv.copy()
    sys.argv = [f"policy_nodes/{module_path.name}"] + remaining_args
    
    # If --help is in remaining args, show help for the specific policy
    if "--help" in remaining_args or "-h" in remaining_args:
        print(f"\n{'=' * 80}")
        print(f"Help for: {args.policy}")
        print(f"{'=' * 80}\n")
    
    try:
        # Run the policy's main function
        print(f"🚀 Running policy: {args.policy}")
        print(f"   Arguments: {' '.join(remaining_args)}\n")
        print("=" * 80)
        policy_module.main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Policy interrupted by user")
        sys.exit(0)  # Exit cleanly on Ctrl+C
    except Exception as e:
        # Check if it's just a shutdown error that we can ignore
        error_msg = str(e)
        if "rcl_shutdown already called" in error_msg or "RCLError" in str(type(e)):
            # This is just a harmless double-shutdown, exit cleanly
            sys.exit(0)
        else:
            print(f"\n❌ Error running policy: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()

