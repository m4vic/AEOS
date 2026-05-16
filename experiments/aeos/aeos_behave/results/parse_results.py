import json
import glob
import os

results = []
for f in glob.glob('*.json'):
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            results.append(data)
    except Exception as e:
        pass

results.sort(key=lambda x: x.get('best_accuracy') or 0, reverse=True)

print(f"{'Model/Combo':<45} | {'Best Acc':<10} | {'Iters':<6} | {'Time(s)':<8} | {'Stop Reason'}")
print('-'*110)

for r in results:
    model_name = r.get('model', r.get('reviewer_model', '') + ' + ' + r.get('coder_model', ''))
    acc = r.get('best_accuracy') or 0.0
    iters = r.get('total_iterations') or 0
    time_sec = r.get('total_time_seconds') or 0.0
    stop_reason = r.get('stop_reason', '')[:50]
    
    print(f"{model_name:<45} | {acc:<10.4f} | {iters:<6} | {time_sec:<8.1f} | {stop_reason}")
