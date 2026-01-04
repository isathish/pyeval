#!/usr/bin/env python3
"""
Run All PyEval Tests
====================

Execute this script to run all unit tests.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Run all test modules."""
    print("\n" + "="*60)
    print("  PYEVAL TEST SUITE")
    print("="*60)
    
    from tests import test_ml, test_nlp, test_recommender, test_speech
    
    results = []
    
    print("\n--- ML Metrics Tests ---")
    results.append(("ML", test_ml.run_tests()))
    
    print("\n--- NLP Metrics Tests ---")
    results.append(("NLP", test_nlp.run_tests()))
    
    print("\n--- Recommender Metrics Tests ---")
    results.append(("Recommender", test_recommender.run_tests()))
    
    print("\n--- Speech Metrics Tests ---")
    results.append(("Speech", test_speech.run_tests()))
    
    # Summary
    print("\n" + "="*60)
    print("  OVERALL RESULTS")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All test suites passed!\n")
    else:
        print("\n❌ Some tests failed!\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
