import json


def load_data(subset):
    with open('train.json', 'r') as f:
        all_questions = json.load(f)
    new_item = []
    if subset == 'unsafes':
        all_questions = [q for q in all_questions if q['safe'] == False]

        for question in all_questions:
            item = {}
            item['id'] = question['id']
            item['question'] = question['instr-resp'][0]['instruction']
            item['response'] = question['instr-resp'][0]['response']
            item['image'] = question['image']
            new_item.append(item)
            # question['question'] = question['instr-resp'][0]['instruction']
    else:
        all_questions = [q for q in all_questions if q['safe'] == True]
        key = 'unsafe_instruction' if subset == 'safe_unsafes' else 'safe_instruction'
        for question in all_questions:
            item = {}
            question['question'] = next(
                (q[key] for q in question['instr-resp'] if key in q), None
            )
            question['response'] = next(
                (q['response'] for q in question['instr-resp'] if key in q), None
            )
            item['id'] = question['id']
            item['question'] = question['question']
            item['response'] = question['response']
            item['image'] = question['image']
            new_item.append(item)
    return new_item
need_subsets = ['safe_safes','safe_unsafes','unsafes']
for subset in need_subsets:
    all_questions = load_data(subset)
    print(all_questions[0])
    print(len(all_questions))
    with open('train_'+subset+'.json', 'w') as f:
        json.dump(all_questions, f, indent=4)