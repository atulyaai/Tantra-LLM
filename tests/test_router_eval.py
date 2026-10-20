from Tantra.eval.router_eval import NONE, evaluate, labelled
from Tantra.model.ngram_router import train_ngram_router


def rows(expert, texts, family):
    return [{'expert': expert, 'family_id': f'{expert}:{family}:{i // 2}',
             'messages': [{'role': 'user', 'content': t}, {'role': 'assistant', 'content': 'ok'}]} for i, t in enumerate(texts)]


IDENTITY = ['who made you', 'who created you', 'who built you', 'who is your creator', 'who designed you', 'who coded you']
WEATHER = ['it is raining today', 'so hot outside', 'it is very cold', 'the weather is nice', 'heavy rain here', 'foggy morning']


def test_none_class_and_metrics():
    names = ['identity_self', NONE]
    data = labelled(rows('identity_self', IDENTITY, 'f') + rows('smalltalk', WEATHER, 'g'), names, none=['smalltalk'])
    assert {names[y] for _, y, _ in data} == {'identity_self', NONE}
    router, _ = train_ngram_router([t for t, _, _ in data], [y for _, y, _ in data], names, epochs=30, holdout=0)
    report = evaluate(router, data, names)
    assert report['rows'] == 12 and report['top1_accuracy'] >= 0.9 and report['top2_recall'] == 1.0
    assert 0.3 <= report['routed_to_base_only'] <= 0.7


def test_grouped_holdout_never_splits_a_family():
    texts = IDENTITY + WEATHER
    groups = [f'g{i // 2}' for i in range(len(texts))]
    labels = [0] * 6 + [1] * 6
    import Tantra.model.ngram_router as module
    seen = {}
    original = module.NgramRouter.forward
    def spy(self, batch):
        seen.setdefault('sizes', []).append(len(batch))
        return original(self, batch)
    module.NgramRouter.forward = spy
    try:
        _, acc = train_ngram_router(texts, labels, ['a', 'b'], epochs=1, holdout=0.34, groups=groups, batch_size=100)
    finally:
        module.NgramRouter.forward = original
    # 6 families, 2 held out -> 4 held-out rows and 8 training rows, never 1 row of a pair on each side
    assert seen['sizes'][0] == 8 and seen['sizes'][-1] == 4 and 0.0 <= acc <= 1.0
