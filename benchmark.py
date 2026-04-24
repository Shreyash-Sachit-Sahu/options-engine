"""Quick benchmark of the C++ pricer."""
import sys, time
sys.path.insert(0, '.')
import pricer

n = 1_000_000
start = time.perf_counter()
for _ in range(n):
    pricer.bs_call(100, 100, 1, 0.05, 0.2)
elapsed = time.perf_counter() - start

print(f"{n:,} BSM calls: {elapsed*1000:.1f}ms ({elapsed/n*1e6:.3f}us/call)")
target_met = elapsed < 0.4
print(f"Target: < 400ms -> {'PASS' if target_met else 'FAIL'}")

# MC benchmark
start2 = time.perf_counter()
mc = pricer.mc_price(100, 100, 1, 0.05, 0.2, n_paths=50000)
mc_elapsed = time.perf_counter() - start2
print(f"\nMC 50k paths: {mc_elapsed*1000:.1f}ms (price={mc.price:.4f}, SE={mc.std_error:.6f})")

# IV benchmark
start3 = time.perf_counter()
for _ in range(10000):
    pricer.implied_vol(10.45, 100, 100, 1, 0.05)
iv_elapsed = time.perf_counter() - start3
print(f"10k IV solves: {iv_elapsed*1000:.1f}ms ({iv_elapsed/10000*1e6:.1f}us/solve)")
