import argparse
import json
import random
import statistics
import time
import urllib.request

"""
Mixed-domain decode benchmark against an OpenAI-compatible endpoint (TabbyAPI), complementing
eval/pentest_bench.py. That bench recycles the same 14 prompts, so after one round the expert cache
has already seen their routing. Here every request is generated fresh from a seed (topics, numbers,
names, lengths are drawn per request), across domains far from security analysis: code generation,
SQL, Polish and English prose, math word problems, structured extraction, translation, long-document
summarization and multi-turn chat. No prompt repeats within or across rounds, so the measured decode
speed reflects the cache on routing it has not been fed before.

Stdlib only. The API key is read from the TabbyAPI token file and never printed. Prompts and outputs
are not written anywhere; only per-request timings are printed.

    python eval/mixed_domain_bench.py --url http://127.0.0.1:5000 \
        --key_file /dev/shm/tabbyAPI/api_tokens.yml --model RED-SNOW-EXL3-4.91bpw --n 30 --seed 1

Use a different --seed for every run you compare (or interleave two variants with the same seed
list) so neither variant gets prompts the cache already saw.
"""

LANGS = ["Python", "Rust", "Go", "TypeScript", "C++", "Kotlin", "Haskell", "Bash"]
TASKS = ["an LRU cache with O(1) get/put", "a token-bucket rate limiter", "a trie with prefix search",
         "a CSV parser that handles quoted fields", "Dijkstra's shortest path on an adjacency list",
         "a thread-safe bounded queue", "an interval-merging function", "a simple regex matcher for . and *",
         "a JSON pretty-printer without libraries", "a union-find with path compression",
         "a min-heap priority queue", "a run-length encoder and decoder"]
TABLES = ["orders", "invoices", "shipments", "patients", "enrollments", "rentals", "tickets", "payments"]
PL_TOPICS = ["historia Gdańska", "uprawa pomidorów na balkonie", "zasady gry w brydża", "jak działa fotowoltaika",
             "kuchnia kaszubska", "budowa mostów wiszących", "migracje ptaków nad Polską", "pieczenie chleba na zakwasie",
             "życie Marii Skłodowskiej-Curie", "ochrona Puszczy Białowieskiej", "tramwaje w Krakowie", "Tatry zimą"]
EN_TOPICS = ["the history of the printing press", "how coral reefs form", "the economics of container shipping",
             "why the sky is blue", "the life cycle of stars", "the invention of refrigeration",
             "how vaccines train the immune system", "the Roman road network", "beekeeping for beginners",
             "the design of Japanese joinery", "how tides work", "the origins of chess"]
STYLES = ["a short essay", "a blog post", "an explanation for a 12-year-old", "a structured summary with headings",
          "a persuasive op-ed", "a FAQ with five questions"]
NAMES = ["Anna", "Piotr", "Kasia", "Marek", "Olga", "Tomasz", "Ewa", "Jan", "Zofia", "Adam", "Lena", "Igor"]
CITIES = ["Wrocław", "Poznań", "Lublin", "Toruń", "Gdynia", "Opole", "Rzeszów", "Olsztyn", "Kielce", "Łódź"]
TRANSLATE_TO = ["German", "Spanish", "French", "Italian", "Czech", "Ukrainian"]


def p_code(r):
    return (f"Write {r.choice(TASKS)} in {r.choice(LANGS)}. Include brief comments, a small usage example "
            f"and state the time complexity of each operation.")


def p_sql(r):
    t = r.choice(TABLES)
    n = r.randint(3, 12)
    return (f"Given a table {t}(id, customer_id, amount, status, created_at) and customers(id, name, country), "
            f"write a SQL query returning the top {n} customers by total {t} amount in the last "
            f"{r.randint(2, 18)} months, excluding status '{r.choice(['cancelled', 'void', 'refunded'])}', "
            f"with their share of the overall total. Then explain which indexes would help and why.")


def p_pl(r):
    return f"Napisz po polsku {r.choice(['krótki esej', 'artykuł na bloga', 'opowiadanie', 'poradnik krok po kroku'])} na temat: {r.choice(PL_TOPICS)}. Około {r.randint(250, 450)} słów."


def p_en(r):
    return f"Write {r.choice(STYLES)} about {r.choice(EN_TOPICS)}, about {r.randint(250, 450)} words."


def p_math(r):
    a, b, c = r.randint(3, 40), r.randint(2, 15), r.randint(5, 90)
    return (f"{r.choice(NAMES)} buys {a} notebooks at {b}.{r.randint(10, 99)} zł each and {c} pens. The total "
            f"bill after a {r.randint(5, 25)}% discount on pens only is {r.randint(200, 900)} zł. What did one pen "
            f"cost before the discount? Solve step by step, then verify the answer by substitution.")


def p_extract(r):
    people = []
    for _ in range(r.randint(5, 9)):
        people.append(f"{r.choice(NAMES)} from {r.choice(CITIES)}, aged {r.randint(19, 77)}, ordered "
                      f"{r.randint(1, 9)} items worth {r.randint(20, 2000)} zł on 2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}")
    return ("Extract the records below into a JSON array with fields name, city, age, items, value_pln, date. "
            "Then give the total value per city.\n\n" + ".\n".join(people) + ".")


def p_translate(r):
    src = (f"Dear {r.choice(NAMES)}, thank you for your order number {r.randint(10000, 99999)}. Unfortunately the "
           f"delivery to {r.choice(CITIES)} will be delayed by {r.randint(2, 9)} days because of a supplier issue. "
           f"We have applied a discount of {r.randint(5, 30)} percent to your next purchase. If you prefer to "
           f"cancel, reply to this message and we will refund the full amount within {r.randint(3, 14)} days.")
    return f"Translate the following message into {r.choice(TRANSLATE_TO)} and into Polish, keeping a polite tone:\n\n{src}"


def p_longdoc(r):
    paras = []
    for i in range(r.randint(25, 45)):
        paras.append(f"Section {i + 1}. In {2000 + r.randint(0, 25)}, the {r.choice(['northern', 'southern', 'eastern', 'western'])} "
                     f"division of a {r.choice(['logistics', 'retail', 'energy', 'farming', 'publishing'])} company in "
                     f"{r.choice(CITIES)} reported revenue of {r.randint(1, 900)} million zł, {r.choice(['up', 'down'])} "
                     f"{r.randint(1, 40)}% year on year, mainly due to {r.choice(['new contracts', 'staff shortages', 'a price war', 'weather', 'a merger', 'regulation'])}. "
                     f"Manager {r.choice(NAMES)} proposed {r.choice(['hiring', 'closing a site', 'new software', 'a pricing change', 'an export push'])}.")
    return ("Summarize the report below in 8 bullet points, then list the three biggest risks and which section "
            "supports each.\n\n" + "\n".join(paras))


def p_chat(r):
    topic = r.choice(EN_TOPICS + PL_TOPICS)
    return [
        {"role": "user", "content": f"I want to learn about {topic}. Where should I start?"},
        {"role": "assistant", "content": "Start with the basic concepts and a short overview, then go deeper into the parts that interest you most."},
        {"role": "user", "content": f"Give me a {r.randint(3, 8)}-step plan for the next {r.randint(2, 6)} weeks, with one concrete exercise per step."},
    ]


GENS = [("code", p_code), ("sql", p_sql), ("prose-pl", p_pl), ("prose-en", p_en), ("math", p_math),
        ("extract", p_extract), ("translate", p_translate), ("longdoc", p_longdoc), ("chat", p_chat)]


def read_key(path):
    with open(path) as f:
        for line in f:
            if line.strip().startswith("api_key:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    raise SystemExit("api_key not found in key file")


def one(url, headers, model, msgs, max_tokens):
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens,
                       "temperature": 0.7, "top_p": 0.95, "stream": False}).encode()
    req = urllib.request.Request(url, data = body, headers = headers, method = "POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout = 900) as r:
        resp = json.load(r)
    dt = time.monotonic() - t0
    u = resp.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default = "http://127.0.0.1:5000")
    ap.add_argument("--key_file", default = "/dev/shm/tabbyAPI/api_tokens.yml")
    ap.add_argument("--model", required = True)
    ap.add_argument("--n", type = int, default = 27, help = "requests (cycled over the domains)")
    ap.add_argument("--seed", type = int, default = 1)
    ap.add_argument("--max_tokens", type = int, default = 400)
    ap.add_argument("--label", default = "")
    args = ap.parse_args()

    url = args.url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + read_key(args.key_file)}
    r = random.Random(args.seed)
    order = [GENS[i % len(GENS)] for i in range(args.n)]
    r.shuffle(order)

    rates, by_dom = [], {}
    for i, (dom, gen) in enumerate(order):
        p = gen(r)
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        try:
            pt, ct, dt = one(url, headers, args.model, msgs, args.max_tokens)
        except Exception as e:
            print(f"{i:3d} {dom:10s} error: {type(e).__name__}", flush = True)
            continue
        tps = ct / dt if dt > 0 else 0.0
        rates.append(tps)
        by_dom.setdefault(dom, []).append(tps)
        print(f"{i:3d} {dom:10s} prompt {pt:5d}  gen {ct:4d}  {dt:6.2f} s  {tps:6.2f} tok/s", flush = True)

    print(f"\n--- {args.label or 'run'} seed {args.seed}: median client-clock tok/s by domain ---")
    for dom in sorted(by_dom):
        v = by_dom[dom]
        print(f"  {dom:10s} n={len(v):2d}  median {statistics.median(v):6.2f}")
    if rates:
        print(f"overall n={len(rates)}  median {statistics.median(rates):.2f}  min {min(rates):.2f}  max {max(rates):.2f}")


if __name__ == "__main__":
    main()
