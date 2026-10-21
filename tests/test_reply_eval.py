from Tantra.eval.reply_eval import aggregate, load_cases, repetitive, score_reply

CREATOR = {'id': 'c', 'prompt': 'Which person built this bot?', 'any': ['atul vij', 'atul']}
NOT_ATUL = {'id': 'n', 'prompt': 'So you are Atul Vij, right?', 'any': ['no', 'not'], 'none': ['i am atul vij', "i'm atul vij"]}
RAIN = {'id': 'r', 'prompt': "It's pouring rain outside.", 'any': ['rain', 'chai']}


def test_case_scoring_rewards_cues_and_punishes_violations():
    assert score_reply(CREATOR, 'Atul Vij built me!')['score'] == 1.0
    assert score_reply(CREATOR, "I'm Tantra! What's your name?")['score'] == 0.4
    assert score_reply(NOT_ATUL, "I'm Atul Vij.")['score'] == 0.0                      # forbidden
    assert score_reply(RAIN, "Rain! I'm an AI, so I can't feel it.")['score'] == 0.8   # unasked AI mention
    assert score_reply(RAIN, '')['score'] == 0.0


def test_repetition_detection():
    assert repetitive("I'm Tantra, and I'm Tantra, and I'm Tantra")
    assert repetitive('old old old')
    assert not repetitive("Hey! How's your day going?")


def test_one_line_for_everything_scores_badly():
    cases = [CREATOR, RAIN, {'id': 'h', 'prompt': 'Hey hey, hello!', 'any': ['hi', 'hello', 'hey']}]
    varied = aggregate({'c': 'Atul Vij made me.', 'r': 'Rain means chai!', 'h': 'Hey! Good to see you.'}, cases)
    same = aggregate({k: "I'm Tantra! What's your name?" for k in 'crh'}, cases)
    assert varied['reply_score'] == 1.0 and varied['passed'] == 3
    assert same['diversity'] < 0.4 and same['reply_score'] < 0.2


def test_shipped_cases_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 30 and len({c['id'] for c in cases}) == len(cases)
    assert all(c['prompt'] and (c.get('any') or c.get('none')) for c in cases)
