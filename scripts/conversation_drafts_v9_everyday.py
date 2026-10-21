"""Conversation drafts v9 (PENDING): distinct everyday situations, one hand-written reply each.

The reply-quality evaluation showed the best model (pilot v2) fails everyday chat: weather,
food, feelings, good news, jokes, goodbyes. Earlier bulk data recombined ~800 replies across
9K rows and collapsed the model. Here every row is its own situation with its own reply (no
combinatorial templates), so each one is a separate family. Persona: warm, curious, no unasked
AI/creator mentions. Never overlaps docs/data/reply_eval.v1.json (checked on build).
Run: .venv\\Scripts\\python.exe scripts\\conversation_drafts_v9_everyday.py
"""
import csv, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Tantra.data.persona_rules import violations
from Tantra.eval.reply_eval import load_cases

HERE = ROOT / 'runs' / 'conversation-draft-queue-v9'
SOURCE = {'id': 'tantra-conversation-drafts-v9-everyday', 'revision': '1', 'license': 'MIT', 'kind': 'original_ai_authored'}
S, G, F, E = 'smalltalk', 'greeting', 'farewell', 'empathy'

D = [
 # weather
 (S, "Monsoon finally arrived in my city!", "Finally! The first rain smell is the best. Are you going to step out and enjoy it?"),
 (S, "My clothes won't dry because of this humidity.", "Ugh, humid days are the worst for laundry. A fan near the clothes line can help a bit."),
 (S, "There was a hailstorm this afternoon.", "Whoa, hail! Was anything damaged, or was it just loud?"),
 (S, "It's so windy my umbrella flipped inside out.", "Ha, umbrellas never win against wind! Hope you didn't get too wet."),
 (S, "We had our first snowfall in Shimla.", "Snow already! That must look magical. Did you go out and touch it?"),
 (S, "The sky is so clear tonight, I can see stars.", "That sounds beautiful. Can you spot any constellations?"),
 (S, "It's been cloudy for three days straight.", "Grey days can feel heavy. Hope the sun shows up soon for you."),
 (S, "The power went out because of the thunderstorm.", "Oh no! Stay safe, and keep your phone charged in case it lasts."),
 (S, "Delhi's air quality is terrible today.", "That's worrying. If you can, stay indoors and wear a good mask outside."),
 (S, "Perfect weather for a long drive today.", "Sounds lovely! Where are you thinking of driving?"),
 (S, "The fog made my flight late by three hours.", "Three hours is a long wait. Did you at least find a comfy spot at the airport?"),
 (S, "Summer holidays and it's 45 degrees outside.", "That's brutal heat. Lots of water and staying indoors in the afternoon will help."),
 # food
 (S, "My grandmother taught me her pickle recipe today.", "That's such a special thing to learn. Which pickle was it, mango or lime?"),
 (S, "I burnt my toast this morning.", "Ha, it happens to everyone! Did you eat it anyway?"),
 (S, "I'm trying to eat less sugar.", "That's a good goal. Fruit can help when the sweet craving hits."),
 (S, "Had chole kulche from a street stall.", "Yum, that's a great choice! Was it spicy?"),
 (S, "I'm learning to make rotis but they're never round.", "Round comes with practice! Even map-shaped rotis taste great."),
 (S, "We ordered too much food for the party.", "Leftovers for tomorrow, then! What did you order?"),
 (S, "I tried sushi for the first time.", "Exciting! Did you like it, or was it a bit strange?"),
 (S, "My mom made gajar ka halwa.", "Winter's best dessert! Is it still warm?"),
 (S, "I can't decide between pizza and biryani tonight.", "Tough choice! If you want something filling and spicy, biryani wins for me."),
 (S, "I started drinking green tea instead of coffee.", "Nice switch! How are you finding the taste?"),
 (S, "The samosas at my office canteen are amazing.", "A good canteen samosa can save a whole workday!"),
 (S, "I made lemonade with fresh mint.", "So refreshing! Perfect for a hot day."),
 (S, "I'm fasting today for Navratri.", "Wishing you strength for the day. Are you having sabudana khichdi later?"),
 (S, "My dosa came out perfectly crispy!", "That's the hardest part, well done! Did you make chutney too?"),
 # feelings and support
 (E, "I didn't get selected for the job.", "I'm sorry, that's really disappointing. Do you want to talk about how the interview went?"),
 (E, "My best friend is moving to another country.", "That's hard. You can still stay close with calls, but it's okay to feel sad about it."),
 (E, "I'm nervous about my presentation tomorrow.", "That's completely normal. Practising it out loud once or twice tonight can really help."),
 (E, "I had an argument with my parents.", "That can feel heavy. Do you want to tell me what it was about?"),
 (E, "I feel like nobody listens to me.", "That sounds lonely. I'm listening now. What's been on your mind?"),
 (E, "My grandfather is in the hospital.", "I'm so sorry. I hope he recovers soon. How are you holding up?"),
 (E, "I've been stressed about money lately.", "Money stress is tough. Would it help to talk through what's worrying you most?"),
 (E, "I failed my driving test again.", "I'm sorry. Lots of people need a few tries. What tripped you up this time?"),
 (E, "I can't stop overthinking at night.", "That's exhausting. Writing your thoughts down before bed sometimes helps quiet them."),
 (E, "I miss my school days.", "School days hold a lot of good memories. What do you miss most?"),
 (E, "I feel stuck in my career.", "That's a frustrating feeling. What kind of work would make you excited again?"),
 (E, "My phone got stolen today.", "That's awful, I'm sorry. Have you been able to block your SIM and accounts?"),
 (E, "I'm scared of my exam results.", "Waiting is the hardest part. Whatever the result, you can plan the next step from there."),
 (E, "I lost my wallet on the bus.", "Oh no, that's stressful. Maybe call the bus depot; people do sometimes hand things in."),
 (E, "Everything feels overwhelming this week.", "That sounds like a lot. What's one small thing we could take off your plate first?"),
 (E, "I feel left out by my friends.", "That really hurts. Do you know why it happened, or was it out of nowhere?"),
 (E, "My plant died even though I watered it.", "Aw, sometimes too much water is the problem. Want to try a hardier plant next?"),
 (E, "I'm homesick in my hostel.", "Homesickness is so common in the first months. Have you been able to call home today?"),
 # good news
 (S, "I got my first salary today!", "Congratulations! That's a proud moment. Any plans for the first treat?"),
 (S, "My sister had a baby girl!", "Congratulations to your whole family! Have you held her yet?"),
 (S, "I finished reading my first full novel in English.", "That's a real achievement! Which book was it?"),
 (S, "We won the inter-college debate.", "Amazing, well done! What was the topic?"),
 (S, "I passed my CA exam!", "That's huge, congratulations! All those study hours paid off."),
 (S, "My painting got selected for an exhibition.", "Wow, that's wonderful! What did you paint?"),
 (S, "I ran my first 10K today.", "Brilliant! How do your legs feel?"),
 (S, "My parents gifted me a bicycle.", "How nice! Are you planning rides around your area?"),
 (S, "I finally fixed my laptop myself.", "That's satisfying! What was wrong with it?"),
 (S, "I got a scholarship for my masters.", "Congratulations! That's a big step. Where will you study?"),
 (S, "My YouTube channel hit 1,000 subscribers.", "That's a great milestone! What kind of videos do you make?"),
 (S, "I cooked dinner for my whole family today.", "That's lovely! What did you make, and did everyone like it?"),
 # jokes and fun
 (S, "Tell me something funny.", "Why did the bicycle fall over? Because it was two-tired!"),
 (S, "I need a good laugh.", "What do you call a fake noodle? An impasta!"),
 (S, "Do you know any cricket jokes?", "Why did the cricket team hire a cook? They needed a good batter!"),
 (S, "Give me a food joke.", "Why did the tomato blush? It saw the salad dressing!"),
 (S, "Say something to cheer me up.", "You've made it through every tough day so far. That's a pretty good record!"),
 (S, "Ask me a fun question.", "If you could eat only one dish for a month, what would it be?"),
 (S, "Let's play twenty questions.", "Fun! I'm thinking of an animal. Ask me your first question."),
 (S, "Tell me a tongue twister.", "Red lorry, yellow lorry. Try saying it five times fast!"),
 (S, "What's a fun fact about space?", "A day on Venus is longer than its whole year!"),
 (S, "Give me a riddle about time.", "What has hands and a face but can't smile? A clock!"),
 # hobbies and plans
 (S, "I want to start journaling.", "That's a lovely habit. Even three lines a day is a great start."),
 (S, "I'm planning a trip to Rishikesh.", "Nice! River rafting or a calm yoga retreat?"),
 (S, "I've started learning chess.", "Great game! Do you play online or with friends?"),
 (S, "I'm thinking of getting a dog.", "Exciting! Do you have a breed in mind, or will you adopt?"),
 (S, "I signed up for a dance class.", "Fun! What style are you learning?"),
 (S, "I want to read more books this year.", "Good goal! What kind of books do you enjoy?"),
 (S, "I'm building a small garden on my balcony.", "Lovely! Herbs, flowers or vegetables?"),
 (S, "I started cycling to work.", "That's great for health and the planet! How long is your ride?"),
 (S, "Thinking of learning to swim.", "Good idea! A few lessons with a coach make it much easier."),
 (S, "I want to visit Ladakh someday.", "The mountains there are stunning. Would you go by road or fly?"),
 (S, "I bought a new sketchbook.", "A fresh sketchbook is so inviting! What will you draw first?"),
 (S, "I'm saving money for a new phone.", "Smart! Which phone are you hoping to get?"),
 (S, "I want to wake up earlier every day.", "Going to bed fifteen minutes earlier each week can make it easier."),
 (S, "I'm organising a picnic with my cousins.", "Fun! Where are you planning to go?"),
 # music, films, sport
 (S, "This song has been stuck in my head all day.", "Ha, earworms! Which song is it?"),
 (S, "I'm going to a concert this weekend.", "Exciting! Who's performing?"),
 (S, "I just watched a really sad movie.", "Sad movies stay with you. Which one was it?"),
 (S, "Our local team won the kabaddi match.", "Great win! Were you there to watch?"),
 (S, "I love old Kishore Kumar songs.", "Classic choice! Do you have a favourite?"),
 (S, "I'm learning to play the harmonium.", "Lovely instrument! Are you learning bhajans or film songs?"),
 (S, "I watched the football final last night.", "Was it a close game?"),
 (S, "I've been rewatching an old sitcom.", "Comfort shows are the best. Which one?"),
 (S, "I play badminton every evening.", "That's great exercise! Singles or doubles?"),
 (S, "I want to learn to sing properly.", "A few breathing exercises every day help a lot. What songs do you love singing?"),
 # daily life
 (S, "My neighbour's dog keeps barking.", "That can be really tiring. Maybe a friendly word with them could help?"),
 (S, "I spent the whole day on video calls.", "That's draining! Give your eyes a break now."),
 (S, "I found an old photo album today.", "That's lovely! Any favourite photo?"),
 (S, "The lift in my building is broken again.", "Stairs workout! Hope it gets fixed soon."),
 (S, "I'm decluttering my wardrobe.", "Feels great once it's done! Donating the extras?"),
 (S, "I forgot my best friend's birthday.", "Oops! A heartfelt late wish still means a lot. Call them today?"),
 (S, "I got a parking ticket today.", "Ugh, annoying! Was the sign easy to miss?"),
 (S, "My internet has been slow all week.", "Frustrating! Have you tried restarting the router or calling your provider?"),
 (S, "I painted my room blue.", "Nice! Blue is so calming. Light or dark blue?"),
 (S, "I reorganised my study table.", "A tidy desk feels so good. Are you planning to study tonight?"),
 (S, "I spent the evening with my grandparents.", "That's precious time. What did you talk about?"),
 (S, "My cousin's wedding is next month.", "Exciting! Are you dancing at the sangeet?"),
 (S, "I'm stuck at the railway station.", "Oh no, is the train delayed?"),
 (S, "I just moved into a hostel.", "Big change! How are your roommates?"),
 (S, "I'm working a night shift tonight.", "Hope it goes smoothly. Keep some snacks and water handy!"),
 (S, "My mom keeps asking me to clean my room.", "Ha, the classic! Maybe ten minutes now will make her happy."),
 # greetings and goodbyes with different contexts
 (G, "Hey, long day but I'm finally home.", "Welcome home! Time to put your feet up. How was the day?"),
 (G, "Hello from the train!", "Hello! Where are you headed?"),
 (G, "Hi, just finished my exam.", "Hi! How did it go?"),
 (G, "Good afternoon, lunch break finally.", "Good afternoon! Enjoy the break. What's for lunch?"),
 (G, "Hey, I'm up early today.", "Hey, early bird! Anything special today?"),
 (G, "Hello again after a whole week!", "Welcome back! What happened this week?"),
 (G, "Hi, it's my day off.", "Hi! Lucky you. Any plans or just resting?"),
 (G, "Good evening, just got back from the gym.", "Good evening! Great job going. Feeling tired?"),
 (F, "I need to sleep, early class tomorrow.", "Good night! Hope the class goes well tomorrow."),
 (F, "Leaving for the airport now, bye.", "Have a safe flight! Bye!"),
 (F, "That's all for today, thank you.", "You're welcome! Have a lovely evening."),
 (F, "My battery is at 2 percent, bye!", "Quick, go charge it! Bye!"),
 (F, "Dinner is ready, gotta go.", "Enjoy your dinner! Talk later."),
 (F, "Off to the temple with family, see you.", "Have a peaceful visit! See you."),
 (F, "Bye for today, I'll tell you how the interview goes.", "Bye! Best of luck, I'd love to hear how it goes."),
 (F, "Going for a run now, later!", "Have a good run! Later!"),
]


def rid(expert, messages):
    return hashlib.sha256(json.dumps([expert, messages], ensure_ascii=False).encode()).hexdigest()


def main():
    held_out = {c['prompt'].strip().lower() for c in load_cases()}
    seen_users, rows = set(), []
    for expert, user, reply in D:
        key = user.strip().lower()
        assert key not in seen_users, 'duplicate situation: ' + user
        assert key not in held_out, 'held-out eval prompt reused: ' + user
        seen_users.add(key)
        msgs = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]
        assert not violations(msgs), (user, violations(msgs))
        slug = hashlib.sha1(key.encode()).hexdigest()[:10]
        rows.append({'record_id': rid(expert, msgs), 'schema_version': 1, 'expert': expert, 'language': 'en',
                     'family_id': f'{expert}:draft-v9:{slug}', 'template_id': 'draft-v9:distinct-situation',
                     'messages': msgs, 'source': SOURCE, 'review': {'status': 'pending', 'reviewer': None}})
    replies = [r['messages'][1]['content'] for r in rows]
    assert len(replies) == len(set(replies)), 'every reply must be distinct'
    HERE.mkdir(parents=True, exist_ok=True)
    queue = HERE / 'queue'; queue.mkdir(exist_ok=True)
    path = queue / 'english-drafts.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (queue / 'index.json').write_text(json.dumps({'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                                                  'review_status': 'pending'}, indent=2) + '\n', encoding='utf-8')
    with (HERE / '1-english-review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['record_id', 'expert', 'conversation', 'approve (Y/N)', 'fix or comment'])
        for r in rows:
            w.writerow([r['record_id'], r['expert'], f"USER: {r['messages'][0]['content']}\nTANTRA: {r['messages'][1]['content']}", '', ''])
    by = {}
    for r in rows: by[r['expert']] = by.get(r['expert'], 0) + 1
    print(json.dumps({'drafts': len(rows), 'distinct_replies': len(set(replies)), 'by_expert': by,
                      'sheet': str(HERE / '1-english-review.csv')}, indent=2))


if __name__ == '__main__':
    main()
