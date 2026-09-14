import random
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        adj = [[] for _ in range(n)]

        for _ in range(m):
            u, v = map(int, f.readline().split())
            adj[u].append(v)
            adj[v].append(u)

    return n, adj

def greedy_coloring(n, adj, order=None):
    if order is None:
        order = sorted(range(n), key=lambda v: len(adj[v]) + random.uniform(0, 5), reverse=True)

    color = [-1] * n

    for v in order:
        used = [False] * n

        for to in adj[v]:
            if color[to] != -1:
                used[color[to]] = True

        c = 0
        while used[c]:
            c += 1

        color[v] = c

    return max(color) + 1, color

def grasp_coloring(n, adj, time_limit=300.0, position=0, test_name=""):
    best_k = n + 1
    best_color = []
    best_at_sec = 0.0
    
    start_time = time.time()
    
    pbar = tqdm(
        total=int(time_limit),
        desc=f"{test_name:<12}",
        position=position,
        leave=True,
        unit="s",
        dynamic_ncols=True
    )
    
    last_sec = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        current_sec = int(elapsed)
        if current_sec > last_sec:
            pbar.update(current_sec - last_sec)
            last_sec = current_sec

        k, color = greedy_coloring(n, adj)

        for _ in range(20):
            classes = [[] for _ in range(k)]
            for v in range(n):
                classes[color[v]].append(v)
            
            if random.random() < 0.6:
                classes.sort(key=len)
            else:
                random.shuffle(classes)

            new_order = [v for cls in classes for v in cls]

            k_new, color_new = greedy_coloring(n, adj, order=new_order)
            color, k = color_new, k_new

        if k < best_k:
            best_k, best_color = k, color[:]
            best_at_sec = time.time() - start_time
            pbar.set_postfix({"Best": best_k, "FoundAt": f"{best_at_sec:.1f}s"})

    pbar.update(int(time_limit) - last_sec)
    pbar.close()

    return best_k, best_color, best_at_sec

def run_single_test(args):
    idx, test = args
    n, adj = read_instance(test)
    test_short_name = test.split("/")[-1]
    
    value, _, best_at_sec = grasp_coloring(
        n, adj, 
        time_limit=300, 
        position=idx, 
        test_name=test_short_name
    )
    return test, value, best_at_sec

tests = [
    "data/gc_50_3",
    "data/gc_70_7",
    "data/gc_100_5",
    "data/gc_250_9",
    "data/gc_500_1",
    "data/gc_1000_5",
]

if __name__ == "__main__":
    indexed_tests = list(enumerate(tests))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(run_single_test, indexed_tests))

    print("\n" + "="*45)
    print(f"{'Тест':<20} | {'Цветов':<8} | {'Найдено на':<10}")
    print("="*45)
    for test, value, best_at in results:
        print(f"{test:<20} | {value:<8} | {best_at:.1f}s")