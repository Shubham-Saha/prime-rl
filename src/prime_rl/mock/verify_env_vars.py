"""
Verify that the verifiers library respects environment variables.

Run this BEFORE the full simulation to confirm traffic redirection works.
"""
import os
import sys

# Set env vars FIRST
os.environ["PRIME_SANDBOX_API_URL"] = "http://localhost:8000"
os.environ["SANDBOX_API_URL"] = "http://localhost:8000"
os.environ["SANDBOX_BASE_URL"] = "http://localhost:8000"
os.environ["PRIME_API_KEY"] = "test-key"

# Now import and inspect
try:
    import verifiers
    
    # Try to find where the sandbox URL is configured
    import inspect
    
    # Look for RemoteSandbox or Sandbox class
    for name, obj in inspect.getmembers(verifiers):
        if 'sandbox' in name.lower():
            print(f"Found: {name} = {obj}")
            if hasattr(obj, '__init__'):
                sig = inspect.signature(obj.__init__)
                print(f"  Signature: {sig}")
    
    # Check if there's a config module
    if hasattr(verifiers, 'config'):
        print(f"\nConfig module: {verifiers.config}")
        for attr in dir(verifiers.config):
            if not attr.startswith('_'):
                print(f"  {attr} = {getattr(verifiers.config, attr, 'N/A')}")
    
    print("\nVerifiers imported successfully")
    print("   Now check if any config points to localhost:8000")

except ImportError as e:
    print(f"Cannot import verifiers: {e}")
    sys.exit(1)