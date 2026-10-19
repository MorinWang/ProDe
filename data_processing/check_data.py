import json

with open('diagnosis/model_responses_converted.jsonl') as f:
    data = [json.loads(line) for line in f][:5]

for i, sample in enumerate(data):
    pass
