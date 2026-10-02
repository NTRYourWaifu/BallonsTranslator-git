"""PR1.5 壓力測試 baseline：跑單 worker、無擾動，產生 baseline.json 供 stress_test_lifecycle.py 比對。

這是 stress_test_lifecycle.py 的 thin wrapper（傳 --baseline_only），方便獨立呼叫。

用法：
    python -m scripts.stress_test_baseline --proj <dir> --max_pages 20
"""
import sys
from scripts.stress_test_lifecycle import main as _main

if __name__ == '__main__':
    # 注入 --baseline_only 旗
    if '--baseline_only' not in sys.argv:
        sys.argv.append('--baseline_only')
    _main()
