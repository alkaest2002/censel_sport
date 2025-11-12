

import sys

from lib_tests.cutoffs_test import get_cutoffs_test


def main() -> int:
    results = get_cutoffs_test()
    print(results)
    return 0

if __name__ == "__main__":
    sys.exit(main())
