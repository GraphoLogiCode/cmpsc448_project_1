"""Read-only checks supporting the report's implementation descriptions."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import build_dataset
import data
import evaluate
import rq4
from models import TextCNN, LSTMClassifier

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

torch.set_num_threads(4)
prompts = build_dataset.read_jsonl(ROOT / 'data/prompts/prompts.jsonl')
assert len({p['id'] for p in prompts}) == len(prompts)
assert len({' '.join(p['text'].lower().split()) for p in prompts}) == len(prompts)
assignment = build_dataset.assign_splits(prompts, .15, .15, 448)
split_rows = {s: data.read_split(s) for s in ('train', 'val', 'test')}
ids = {s: {r['prompt_id'] for r in rows} for s, rows in split_rows.items()}
assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
for split, rows in split_rows.items():
    groups = defaultdict(list)
    for row in rows:
        assert assignment[row['prompt_id']] == split
        groups[row['prompt_id']].append(row)
    assert all(len(rs) == 3 and len({r['llm_name'] for r in rs}) == 3
               and len({r['llm_input'] for r in rs}) == 1 for rs in groups.values())
print('Splits: pre-cleaning assignment reproduced; retained groups complete and disjoint.', flush=True)

raw_files = [p for p in (ROOT / 'data/responses').glob('*.jsonl') if not p.stem.endswith('_trial')]
raw = [r for p in raw_files for r in build_dataset.read_jsonl(p)]
retained = {}
for r in raw:
    r = {**r, 'llm_output': r['llm_output'].strip()}
    if r['llm_output'] and not build_dataset.names_itself(r):
        retained.setdefault(r['prompt_id'], {})[r['llm_name']] = r
expected = {(r['prompt_id'], r['llm_name']): {c:r[c] for c in build_dataset.COLUMNS}
            for rs in retained.values() if len(rs) == 3 for r in rs.values()}
actual = {(r['prompt_id'], r['llm_name']):r for rows in split_rows.values() for r in rows}
assert actual == expected
print('Cleaning: all 5,994 saved rows reproduced from raw responses.', flush=True)

example = {'llm_input':' '.join(['prompt']*250), 'llm_output':' '.join(['response']*600)}
tokens = data.row_tokens(example, 'both', 512)
assert len(tokens)==512 and tokens[:170]==['prompt']*170 and tokens[170]=='<sep>' and tokens[171:]==['response']*341
assert data.row_tokens({'llm_input':'', 'llm_output':''}, 'answer', 512)==['<unk>']
assert data.tokenize(' **Example**\n') == data.tokenize('**Example**')
print('Representations: 170 prompt + separator + 341 response tokens; whitespace discarded.', flush=True)

for cls in (TextCNN, LSTMClassifier):
    torch.manual_seed(448)
    model = cls(30,3).eval()
    for values in ([3], [3,4,5], [3,4,5,6,7,8,9]):
        batch = [(torch.tensor(values),0), (torch.tensor(list(range(3,25))),1)]
        x,lengths,_=data.collate(batch)
        with torch.no_grad():
            single = model(torch.tensor([values]),torch.tensor([len(values)]))
            together = model(x,lengths)[:1]
        torch.testing.assert_close(single,together,rtol=1e-5,atol=1e-6)
        assert single.shape==(1,3) and torch.isfinite(single).all()
print('Forward passes: CNN and LSTM give finite 3-logit outputs, including short sequences; batch padding checks passed.', flush=True)

style = rq4.style_by_model(split_rows['train'], sorted({r['llm_name'] for r in split_rows['train']}))
assert style == read('results/rq4.json')['style_by_model']
eligible = Counter(r['llm_name'] for r in split_rows['train'] if len(data.tokenize(r['llm_output']))>=100)
truncated = sum(len(data.tokenize(r['llm_output']))>512 for r in split_rows['train'])
print('RQ4 descriptive statistics reproduced exactly. First-100-token eligible counts:', dict(eligible), flush=True)
print('Classifier truncation:',truncated,'of',len(split_rows['train']),f'({truncated/len(split_rows["train"]):.1%}).',flush=True)

device='cuda' if torch.cuda.is_available() else 'cpu'
classes={'cnn':TextCNN,'lstm':LSTMClassifier}
matched=0
for file in sorted((ROOT/'results/runs').glob('*/config.json')):
    cfg=json.loads(file.read_text())
    if cfg['model'] not in classes:
        continue
    history=json.loads(file.with_name('history.json').read_text())
    best=max(history,key=lambda r:(r['val_acc'],-r['val_loss']))
    assert best['epoch']==cfg['best_epoch']
    tasks=cfg['held_out']
    loaders=data.make_loaders(cfg['mode'],cfg['max_len'],cfg['batch_size'],
        train_tasks=[t for t in ('coding','factual','math','writing') if t!=tasks] if tasks else None,
        test_tasks=[tasks] if tasks else None,seed=cfg['seed'])
    assert len(loaders['vocab'])==cfg['vocabulary']
    model=classes[cfg['model']](len(loaders['vocab']),len(loaders['names']),emb_dim=cfg['emb_dim'],dropout=cfg['dropout']).to(device)
    assert sum(p.numel() for p in model.parameters())==cfg['parameters']
    state=torch.load(ROOT/f'models/{cfg["run"]}.pt',map_location=device,weights_only=True)
    model.load_state_dict(state)
    indices=evaluate.predict(model,loaders['test'],device)
    predictions=[loaders['names'][i] for i in indices]
    saved=build_dataset.read_jsonl(file.with_name('predictions_test.jsonl'))
    difference=sum(a!=b['predicted'] for a,b in zip(predictions,saved))
    assert len(predictions)==len(saved)
    scores=evaluate.score([r['llm_name'] for r in saved],predictions,[r['task'] for r in saved],loaders['names'])
    print(cfg['run'],'vocabulary and parameters verified; prediction differences:',difference,'/',len(saved),'accuracy:',scores['accuracy'],flush=True)
    assert difference==0,(cfg['run'],difference)
    assert scores==read(f'results/runs/{cfg["run"]}/metrics.json')['test']
    matched+=1
print(f'All {matched} saved neural checkpoints reproduce test predictions and every stored test metric.',flush=True)

