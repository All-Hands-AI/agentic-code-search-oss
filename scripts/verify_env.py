#!/usr/bin/env python3
"""
Verify that the swe-grep-oss-env can be loaded by verifiers.
This should be run before attempting RL training.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Verifying SWE-Grep Environment Setup")
    logger.info("=" * 60)
    
    # Test 1: Import verifiers
    logger.info("\n[1/5] Testing verifiers import...")
    try:
        import verifiers as vf
        logger.info("✓ verifiers imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import verifiers: {e}")
        return False
    
    # Test 2: Import swe_grep_oss_env module
    logger.info("\n[2/5] Testing swe_grep_oss_env module import...")
    try:
        import swe_grep_oss_env
        logger.info("✓ swe_grep_oss_env module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import swe_grep_oss_env: {e}")
        return False
    
    # Test 3: Check for load_environment function
    logger.info("\n[3/5] Checking for load_environment function...")
    if hasattr(swe_grep_oss_env, 'load_environment'):
        logger.info("✓ load_environment function found")
    else:
        logger.error("✗ load_environment function not found in module")
        return False
    
    # Test 4: Load environment using verifiers
    logger.info("\n[4/5] Loading environment via verifiers.load_environment()...")
    try:
        env = vf.load_environment("swe-grep-oss-env", max_tokens=2048, max_tool_calls=20)
        logger.info(f"✓ Environment loaded successfully")
        logger.info(f"  - Type: {type(env).__name__}")
        logger.info(f"  - Environment ID: {env.env_id}")
        logger.info(f"  - Max turns: {env.max_turns}")
    except Exception as e:
        logger.error(f"✗ Failed to load environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Check dataset
    logger.info("\n[5/5] Checking dataset...")
    try:
        dataset = env.dataset
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  - Dataset size: {len(dataset)}")
        logger.info(f"  - Dataset columns: {dataset.column_names}")
        
        # Check first example
        if len(dataset) > 0:
            first_example = dataset[0]
            logger.info(f"  - First example keys: {list(first_example.keys())}")
            logger.info(f"  - Has 'prompt': {'prompt' in first_example}")
            logger.info(f"  - Has 'info': {'info' in first_example}")
    except Exception as e:
        logger.error(f"✗ Failed to access dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ All checks passed! Environment is ready for RL training.")
    logger.info("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

