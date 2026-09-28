"""New stories for CLadder's questions: the same structural models, numbers and
questions, told in new words (prereg/NEW_STORIES.md).

Why. Real names beat letters on CLadder, but CLadder's text has been public
since 2023, so part of that gain could be recall of the benchmark's own
wording. Each of the 37 stories of Study 1 is moved to a new domain whose
variables play the same roles in the same graph (a confounder stays a
confounder, a mediator a mediator), and every story-specific word of the
question is replaced, not only the names in CLadder's variable mapping: the
unit nouns ("patients", "farms"), the grammatical variants ("receiving",
"blown out") and the hand-written phrases of the counterfactual questions
("Would the alarm rings the next morning"). The graph, the probabilities, the
query and the gold label do not change.

CLadder's anticommonsense versions replace the treatment or the outcome with an
unrelated phrase ("having a sister", "freckles"). Their new versions do the same
with other unrelated phrases (ANTI), so a new question is plausible or
implausible exactly when its CLadder original is: the label comes from the
construction, as CLadder's own does.

The swap is one pass over the prompt with every phrase of the story's table and
of ANTI, longest first, whole words only (a letter, an apostrophe or a hyphen
next to a match blocks it, so "male" leaves "non-male" alone). rewrite() then
checks that no story-specific word of the original survives: every word of the
original prompt that is not common to CLadder's templates (see template_vocab)
must be gone, unless it is in KEEP_WORDS, a list of generic words.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Unrelated phrases of CLadder's anticommonsense versions -> new unrelated phrases.
ANTI = [
    ("sister", "garden"), ("brother", "bicycle"),
    ("coffee", "lemonade"), ("spicy food", "folk music"),
    ("english", "Italian"), ("England", "Norway"),
    ("swim", "whistle"), ("jazz", "podcasts"),
    ("freckles", "dimples"),
    ("black hair", "red hair"), ("blond hair", "grey hair"),
    ("brown eyes", "hazel eyes"), ("blue eyes", "green eyes"),
    ("lip thickness", "eyelash length"), ("thick lips", "long eyelashes"),
    ("thin lips", "short eyelashes"),
    ("foot size", "finger length"), ("large feet", "long fingers"),
    ("small feet", "short fingers"),
    ("lactose intolerance", "ticklishness"), ("lactose intolerant", "ticklish"),
    ("peanut allergy", "fear of heights"), ("allergic to peanuts", "afraid of heights"),
    ("full moon", "leap year"), ("rainfall", "sunspot activity"),
    ("solar eclipse", "meteor shower"),
    ("curly hair", "long hair"), ("straight hair", "short hair"),
]

STORIES: dict[str, list[tuple[str, str]]] = {
    # mediation: husband -> wife -> alarm, husband -> alarm
    "alarm": [
        ("alarm set by husband", "porch light switched on by father"),
        ("alarm not set by husband", "porch light not switched on by father"),
        ("alarm set by wife", "porch light switched on by mother"),
        ("alarm not set by wife", "porch light not switched on by mother"),
        ("ringing alarm", "glowing porch light"), ("silent alarm", "dark porch light"),
        ("alarm clock", "porch light"),
        ("don't set the alarm", "don't switch on the porch light"),
        ("set the alarm", "switch on the porch light"),
        ("had not set the alarm", "had not switched on the porch light"),
        ("the alarm rings the next morning", "the porch light glows at night"),
        ("husbands", "fathers"), ("husband", "father"), ("wives", "mothers"), ("wife", "mother"),
    ],
    # mediation: medication -> blood pressure -> heart, medication -> heart
    "blood_pressure": [
        ("medication", "sleeping medicine"),
        ("high blood pressure", "deep sleep"), ("low blood pressure", "light sleep"),
        ("blood pressure", "sleep depth"),
        ("heart condition", "daytime alertness"), ("healthy heart", "alert day"),
        ("heart problems", "drowsy day"),
    ],
    # fork: man blows out candle -> dark room <- candle wax
    "candle": [
        ("the man in the room", "the woman in the cabin"),
        ("not blowing out the candle", "not opening the window"),
        ("blowing out the candle", "opening the window"),
        ("not blowing out candles", "not opening windows"),
        ("blow out candles", "open windows"),
        ("blown out the candle", "opened the window"),
        ("candle with wax", "heater with fuel"), ("candle out of wax", "heater out of fuel"),
        ("the candle", "the heater"), ("candles", "heaters"), ("candle", "heater"),
        ("dark room", "cold cabin"), ("bright room", "warm cabin"), ("room", "cabin"),
    ],
    # collision: appearance -> fame <- talent
    "celebrity": [
        ("attractive appearance", "fast running"), ("unattractive appearance", "slow running"),
        ("appearance", "speed"),
        ("considered attractive", "considered fast"), ("considered unattractive", "considered slow"),
        ("lack of talent", "lack of stamina"), ("talent", "stamina"),
        ("lack of fame", "lack of team selection"), ("fame", "team selection"),
        ("famous", "selected"),
    ],
    # IV: assignment -> drug taken -> cholesterol, unobserved confounders
    "cholesterol": [
        ("treatment assignment", "voucher allocation"),
        ("absence of assignment of drug treatment", "absence of allocation of a gym voucher"),
        ("assignment of drug treatment", "allocation of a gym voucher"),
        ("assigned the drug treatment", "allocated the gym voucher"),
        ("assigned the treatment", "allocated the voucher"),
        ("drug taken", "gym attendance"),
        ("not taking of any assigned drugs", "attendance at no voucher sessions"),
        ("taking of all assigned drugs", "attendance at all voucher sessions"),
        ("cholesterol level", "fitness level"),
        ("low cholesterol", "high fitness"), ("high cholesterol", "low fitness"),
        ("patients", "members"), ("patient", "member"),
    ],
    # chain: education -> skill -> salary
    "college_salary": [
        ("education level", "licence status"),
        ("college degree or higher", "full driving licence"),
        ("high school degree or lower", "no driving licence"),
        ("college degree", "driving licence"),
        ("high skill level", "high mobility level"), ("low skill level", "low mobility level"),
        ("skill", "mobility"),
        ("salary", "income"), ("employee", "worker"),
    ],
    # IV: proximity to a college -> education -> salary
    "college_wage": [
        ("proximity to a college", "proximity to a training centre"),
        ("close to a college", "close to a training centre"),
        ("far from a college", "far from a training centre"),
        ("education level", "training level"),
        ("college degree or higher", "vocational certificate or higher"),
        ("high school degree or lower", "no vocational certificate"),
        ("salary", "wage"), ("employee", "tradesperson"),
    ],
    # collision: talent -> elite admission <- effort
    "elite_students": [
        ("elite institution admission status", "top orchestra audition result"),
        ("elite institution acceptance", "top orchestra acceptance"),
        ("elite institution rejection", "top orchestra rejection"),
        ("accepted to elite institutions", "accepted to top orchestras"),
        ("rejected from elite institutions", "rejected from top orchestras"),
        ("lack of talent", "lack of technique"), ("talented", "skilled"), ("talent", "technique"),
        ("being hard-working", "being disciplined"), ("being lazy", "being careless"),
        ("effort", "discipline"),
        ("students", "musicians"),
    ],
    # mediation: encouragement -> studying -> exam score
    "encouagement_program": [
        ("encouragement level", "coaching level"),
        ("discouragement", "absence of coaching"), ("encouragement", "coaching"),
        ("not encouraged", "not coached"), ("encouraged", "coached"),
        ("studying habit", "training habit"), ("studying hard", "training daily"),
        ("lack of studying", "lack of training"),
        ("do not study hard", "do not train daily"), ("study hard", "train daily"),
        ("exam score", "match score"),
        ("students", "players"), ("student", "player"),
    ],
    # diamondcut: CEO -> director, CEO -> manager; manager, director -> employee
    "firing_employee": [
        ("CEO's decision to fire the employee", "principal's decision to expel the student"),
        ("CEO's decision to retain the employee", "principal's decision to keep the student"),
        ("decides to fire", "decides to expel"), ("decides to retain", "decides to keep"),
        ("CEOs", "principals"), ("CEO", "principal"),
        ("directors", "vice principals"), ("director", "vice principal"),
        ("managers", "class teachers"), ("manager", "class teacher"),
        ("termination letters", "expulsion forms"), ("termination letter", "expulsion form"),
        ("employees", "students"), ("employee", "student"),
        ("fired", "expelled"), ("fire", "expel"),
    ],
    # diamond: captain -> corporal, private -> prisoner
    "firing_squad": [
        ("the captain's order to execute the prisoner", "the foreman's order to demolish the shed"),
        ("the captain's order to release the prisoner", "the foreman's order to spare the shed"),
        ("ordered the release of the", "ordered the sparing of the"),
        ("the captain", "the foreman"), ("captains", "foremen"), ("captain", "foreman"),
        ("execute", "demolish"), ("release", "spare"),
        ("the private shooting", "the labourer hammering"),
        ("the private not shooting", "the labourer not hammering"),
        ("the private", "the labourer"),
        ("the corporal shooting", "the crane operator swinging"),
        ("the corporal not shooting", "the crane operator not swinging"),
        ("the corporal", "the crane operator"),
        ("the prisoner's death", "the shed's collapse"),
        ("the prisoner being alive", "the shed standing"),
        ("prisoners", "sheds"), ("prisoner", "shed"),
        ("is alive", "is standing"), ("is dead", "is collapsed"),
    ],
    # diamond: season -> rain, sprinkler -> wet ground
    "floor_wet": [
        ("rainy season", "festival week"), ("dry season", "ordinary week"), ("season", "week type"),
        ("had been dry", "had been ordinary"),
        ("sprinkler on", "street market open"), ("sprinkler off", "street market closed"),
        ("sprinkler", "street market"),
        ("weather", "crowding"), ("no rain", "no crowds"), ("rain", "crowds"),
        ("wet ground", "noisy street"), ("dry ground", "quiet street"),
        ("is dry", "is quiet"), ("is wet", "is noisy"), ("ground", "street"),
    ],
    # fork: camper -> forest fire <- smoker
    "forest_fire": [
        ("the chain-smoker lighting a match", "the plumber opening a valve"),
        ("the chain-smoker not lighting a match", "the plumber not opening a valve"),
        ("the chain-smoker", "the plumber"), ("the smoker", "the plumber"),
        ("the camper lighting a match", "the gardener opening a valve"),
        ("the camper not lighting a match", "the gardener not opening a valve"),
        ("the camper", "the gardener"), ("campers", "gardeners"), ("camper", "gardener"),
        ("light a match", "open a valve"), ("lit a match", "opened a valve"),
        ("the forest on fire", "the cellar flooded"),
        ("the forest not on fire", "the cellar not flooded"),
        ("the forest", "the cellar"),
        ("burns down", "floods"), ("burn down", "flood"),
    ],
    # mediation: gender -> department -> admission, gender -> admission
    "gender_admission": [
        ("non-male gender", "local nationality"), ("male gender", "foreign nationality"),
        ("non-male", "local"), ("gender", "nationality"), ("male", "foreign"),
        ("department competitiveness", "course popularity"),
        ("non-competitive department", "less popular course"),
        ("competitive department", "popular course"),
        ("admission status", "scholarship status"),
        ("admission acceptance", "scholarship award"),
        ("admission rejection", "scholarship refusal"),
        ("applicants", "candidates"), ("applicant", "candidate"),
    ],
    # arrowhead: gender -> department -> admission; residency unobserved
    "gender_admission_state": [
        ("non-male gender", "young age"), ("male gender", "mature age"),
        ("non-male", "young"), ("gender", "age group"), ("male", "mature"),
        ("residency status", "commuting status"),
        ("in-state residency", "local commuting"), ("out-of-state residency", "distant commuting"),
        ("in-state residents", "local commuters"), ("out-of-state residents", "distant commuters"),
        ("department competitiveness", "programme demand"),
        ("non-competitive department", "low-demand programme"),
        ("competitive department", "high-demand programme"),
        ("admission status", "enrolment status"),
        ("admission acceptance", "enrolment offer"),
        ("admission rejection", "enrolment refusal"),
        ("applicants", "candidates"), ("applicant", "candidate"),
    ],
    # arrowhead: gender -> occupation -> salary; skill unobserved
    "gender_pay": [
        ("non-male gender", "no union membership"), ("male gender", "union membership"),
        ("non-male", "non-union"), ("gender", "union status"), ("male", "unionised"),
        ("high skill level", "high experience level"), ("low skill level", "low experience level"),
        ("skill", "experience"),
        ("occupation", "contract type"),
        ("white-collar job", "permanent contract"), ("blue-collar job", "temporary contract"),
        ("white-collar workers", "permanent workers"), ("blue-collar workers", "temporary workers"),
        ("salary", "bonus"), ("employee", "worker"),
    ],
    # fork: Alice waking up -> arriving <- traffic
    "getting_late": [
        ("Alice waking up", "Tom leaving home"),
        ("waking up late", "leaving home late"), ("waking up on time", "leaving home early"),
        ("wakes up late", "leaves home late"), ("wakes up on time", "leaves home early"),
        ("gotten up on time", "left home early"),
        ("Alice arriving to school", "Tom reaching the office"),
        ("arriving to school on time", "reaching the office on time"),
        ("arriving to school late", "reaching the office late"),
        ("arrives at school", "reaches the office"),
        ("Alice", "Tom"),
        ("heavy traffic", "a long queue"), ("no traffic", "no queue"), ("traffic", "queue"),
        ("on the road", "at the station"),
    ],
    # fork: tanning salon -> skin <- beach
    "getting_tanned": [
        ("no tanning salon treatment", "no tooth whitening treatment"),
        ("tanning salon treatments", "tooth whitening treatments"),
        ("tanning salon treatment", "tooth whitening treatment"),
        ("tanning salons", "whitening clinics"),
        ("not going to the beach", "not going to the dentist"),
        ("going to the beach", "going to the dentist"),
        ("goes to the beach", "goes to the dentist"), ("go to the beach", "go to the dentist"),
        ("tanned skin", "white teeth"), ("pale skin", "stained teeth"), ("skin", "teeth"),
    ],
    # collision: respiratory issues -> hospitalization <- broken bones
    "hospitalization": [
        ("hospitalization status", "emergency admission status"),
        ("non-hospitalization", "no emergency admission"),
        ("hospitalization", "emergency admission"),
        ("non-hospitalized individuals", "individuals not admitted to emergency"),
        ("hospitalized individuals", "individuals admitted to emergency"),
        ("no respiratory issues", "no high fever"), ("respiratory issues", "high fever"),
        ("no broken bones", "no sprained ankles"), ("broken bones", "sprained ankles"),
    ],
    # collision: personality -> relationship <- appearance
    "man_in_relationship": [
        ("attractive appearance", "fit physique"), ("unattractive appearance", "unfit physique"),
        ("appearance", "physique"),
        ("personality", "humour style"), ("kindness", "humour"), ("meanness", "seriousness"),
        ("relationship status", "dating status"), ("in a relationship", "dating someone"),
        ("single", "not dating"),
    ],
    # arrowhead: parents' intelligence -> social status -> child
    "nature_vs_nurture": [
        ("parents' intelligence", "parents' musicality"),
        ("unintelligent parents", "unmusical parents"), ("intelligent parents", "musical parents"),
        ("parents' social status", "household income"),
        ("high parental social status", "high household income"),
        ("low parental social status", "low household income"),
        ("child's intelligence", "child's musicality"),
        ("unintelligent child", "unmusical child"), ("intelligent child", "musical child"),
        ("children", "kids"),
    ],
    # mediation: smoking -> effort -> college admission
    "neg_mediation": [
        ("nonsmokers", "non-gamers"), ("nonsmoker", "non-gamer"), ("nonsmoking", "non-gaming"),
        ("smokers", "gamers"), ("smoking", "gaming"),
        ("are hard-working", "are well-rested"), ("are lazy", "are tired"),
        ("being hard-working", "being well-rested"), ("being lazy", "being tired"),
        ("effort", "rest"),
        ("college admission", "exam pass"), ("college rejection", "exam fail"),
        ("student", "pupil"),
    ],
    # arrowhead: obesity -> diabetes -> lifespan; smoking unobserved
    "obesity_mortality": [
        ("obesity", "inactivity"), ("obese", "inactive"), ("normal weight", "physically active"),
        ("nonsmokers", "clean-air residents"), ("nonsmoker", "clean-air resident"),
        ("smokers", "polluted-city residents"), ("smoker", "polluted-city resident"),
        ("smoking", "air quality"),
        ("having diabetes", "having heart disease"), ("absence of diabetes", "absence of heart disease"),
        ("diabetes", "heart disease"),
        ("long lifespan", "long life expectancy"), ("short lifespan", "short life expectancy"),
        ("lifespan", "life expectancy"),
    ],
    # chain: citrus -> vitamin C -> scurvy
    "orange_scurvy": [
        ("eating citrus", "eating dairy"), ("citrus intake", "dairy intake"),
        ("absence of citrus", "absence of dairy"), ("citrus", "dairy"),
        ("vitmain C", "calcium level"), ("sufficient vitamin C", "sufficient calcium"),
        ("vitamin C deficiency", "calcium deficiency"),
        ("no scurvy", "no brittle bones"), ("scurvy", "brittle bones"),
        ("patients", "residents"), ("patient", "resident"),
    ],
    # mediation: my decision -> penguin mood -> penguin survival
    "penguin": [
        ("my decision", "my travel choice"),
        ("taking the elevator", "taking the tram"), ("taking the stairs", "taking the ferry"),
        ("take the elevator", "take the tram"), ("take the stairs", "take the ferry"),
        ("taken the stairs", "taken the ferry"),
        ("penguin saddness", "parrot sadness"),
        ("penguins", "parrots"), ("penguin", "parrot"),
    ],
    # IV: yield per acre -> supply -> price, demand
    "price": [
        ("crop yield per acre", "driver availability"), ("yield per acre", "driver availability"),
        ("farms", "depots"), ("farm", "depot"),
        ("demand", "event activity"), ("supply", "bus frequency"), ("price", "ridership"),
    ],
    # confounding: gender -> treatment, recovery
    "simpson_drug": [
        ("non-male gender", "no diabetes"), ("male gender", "diabetes"),
        ("gender", "diabetes status"), ("male", "diabetic"),
        ("treatment", "therapy"),
        ("non-recovery", "non-healing"), ("recovery", "healing"), ("recover", "heal"),
    ],
    # confounding: age -> hospital costs, recovery
    "simpson_hospital": [
        ("old age", "severe illness"), ("youth", "mild illness"), ("age", "illness severity"),
        ("old", "severely ill"), ("young", "mildly ill"),
        ("hospital costs", "clinic fees"),
        ("high hospital bill", "high clinic fee"), ("low hospital bill", "low clinic fee"),
        ("non-recovery", "non-improvement"), ("recovery", "improvement"), ("recovers", "improves"),
    ],
    # confounding: kidney stone size -> treatment, recovery
    "simpson_kidneystone": [
        ("kidney stone size", "fracture severity"),
        ("large kidney stones", "complex fractures"), ("small kidney stones", "simple fractures"),
        ("large kidney stone", "complex fracture"), ("small kidney stone", "simple fracture"),
        ("treatment", "surgery"),
        ("non-recovery", "immobility"), ("recovery", "mobility"),
    ],
    # confounding: pre-conditions -> vaccination, disease
    "simpson_vaccine": [
        ("pre-conditions", "prior injuries"),
        ("vaccination", "knee bracing"),
        ("getting the vaccine", "getting the knee brace"),
        ("vaccine refusal", "knee brace refusal"),
        ("refusing the vaccine", "refusing the knee brace"),
        ("refused the vaccine", "refused the knee brace"),
        ("vaccine", "knee brace"),
        ("recovering from the disease", "recovering from the sprain"),
        ("dying from the disease", "worsening of the sprain"),
        ("disease", "sprain"),
    ],
    # arrowhead: maternal smoking -> birth weight -> infant mortality; health unobserved
    "smoke_birthWeight": [
        ("maternal smoking status", "cow's feeding status"),
        ("nonsmoking mothers", "well-fed cows"), ("smoking mothers", "underfed cows"),
        ("nonsmoking mother", "well-fed cow"), ("smoking mother", "underfed cow"),
        ("infant's birth weight", "calf's birth weight"),
        ("normal infant birth weight", "normal calf birth weight"),
        ("low infant birth weight", "low calf birth weight"),
        ("infant mortality", "calf mortality"),
        ("infants", "calves"), ("infant", "calf"),
    ],
    # frontdoor: smoking -> tar -> lung cancer; gender unobserved
    "smoking_frontdoor": [
        ("nonsmokers", "soda abstainers"), ("nonsmoker", "soda abstainer"),
        ("nonsmoking", "no soda drinking"),
        ("smokers", "soda drinkers"), ("smoking", "soda drinking"),
        ("tar deposit", "plaque buildup"),
        ("lung cancer", "tooth decay"),
        ("non-female", "not sweet-toothed"), ("female", "sweet-toothed"),
        ("gender", "taste preference"),
    ],
    # arrowhead: gene -> smoking -> lung cancer; pollution unobserved
    "smoking_gene_cancer": [
        ("nonsmoking genes", "non-craving genes"), ("smoking genes", "craving genes"),
        ("nonsmoking gene", "non-craving gene"), ("smoking gene", "craving gene"),
        ("nonsmokers", "non-gamblers"), ("smokers", "gamblers"),
        ("nonsmoking", "non-gambling"), ("smoking", "gambling"),
        ("lung cancer", "heavy debt"),
        ("pollution", "stress"),
    ],
    # chain: smoking -> tar -> lung cancer
    "smoking_tar_cancer": [
        ("nonsmokers", "non-sunbathers"), ("nonsmoker", "non-sunbather"),
        ("nonsmoking", "no sunbathing"),
        ("smokers", "sunbathers"), ("smoking", "sunbathing"),
        ("tar deposit", "UV damage"),
        ("lung cancer", "skin lesions"),
    ],
    # IV: cigarette tax -> maternal smoking -> birth weight
    "tax_smoke_birthWeight": [
        ("cigarette tax", "fuel tax"),
        ("maternal smoking status", "commuting mode"),
        ("nonsmoking mother", "cycling commuter"), ("smoking mother", "driving commuter"),
        ("infant's birth weight", "commuter's fitness"),
        ("normal infant birth weight", "normal commuter fitness"),
        ("low infant birth weight", "low commuter fitness"),
        ("is born with a normal weight", "ends up with normal fitness"),
        ("is born with low weight", "ends up with low fitness"),
        ("infant", "commuter"),
    ],
    # diamond: vaccination -> smallpox, reaction -> survival
    "vaccine_kills": [
        ("vaccination status", "helmet status"),
        ("severe vaccination reaction", "severe helmet discomfort"),
        ("no vaccine reaction", "no helmet discomfort"),
        ("vaccination reaction", "helmet discomfort"),
        ("lack of vaccination", "lack of a helmet"), ("vaccination", "helmet wearing"),
        ("unvaccinated individuals", "riders without helmets"),
        ("vaccinated individuals", "riders with helmets"),
        ("had been vaccinated", "had worn a helmet"),
        ("getting smallpox", "getting a concussion"), ("having smallpox", "having a concussion"),
        ("absence of smallpox", "absence of a concussion"),
        ("smallpox survival", "crash survival"), ("smallpox death", "crash death"),
        ("dies from smallpox", "dies from the crash"),
    ],
    # IV: water company -> water quality -> cholera, poverty
    "water_cholera": [
        ("global water company", "national power supplier"),
        ("local water company", "local power supplier"),
        ("water company", "power supplier"),
        ("water quality", "air quality"), ("clean water", "clean air"),
        ("polluted water", "polluted air"),
        ("cholera contraction", "asthma onset"), ("cholera prevention", "asthma prevention"),
        ("cholera", "asthma"),
        ("high poverty", "high housing density"), ("low poverty", "low housing density"),
        ("poverty", "housing density"),
    ],
}

# Generic words that may survive: grammar, template words of a few stories, and
# role words ("unobserved confounders") that are not a story's content.
KEEP_WORDS = set("""
don't doesn't by from at in out off on up down when where there those i my an per all can
cannot being been able ability like liking drink drinking speak speaking listen listening
have visited go goes going went take taken taking using get gets getting received receives
receiving experiences level levels high low long short large small normal good poor health
condition increased reduced sufficient deficiency absence intake eating consumed consuming
status individual individuals patient patients days time late situations affects between
correlation considered look mean choose decides decision sign signed signing order ordered
active inactive confounder confounders other gene genes weight birth mortality workers
accepted rejected admission acceptance happy sad habit score parents parents' child child's
lives living close far proximity higher treatment treatments decision sad happy eyes hair
""".split())

# Words a new story shares with its old one on purpose: generic words its new
# phrases reuse ("served by a local power supplier", "crash survival").
SHARED = {
    "water_cholera": {"served", "local", "region", "avoids", "quality", "clean", "polluted"},
    "tax_smoke_birthWeight": {"tax"},
    "simpson_hospital": {"pay", "paid"},
    "simpson_vaccine": {"recovering", "recovers", "refusing", "refused", "refusal"},
    "vaccine_kills": {"survival", "severe", "dies"},
    "penguin": {"survival", "death", "mood", "happiness"},
}

MIN_STORIES = 9          # a word in the prompts of this many stories is template
_W = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@lru_cache(maxsize=1)
def template_vocab() -> frozenset:
    """Words in the real-story prompts of at least MIN_STORIES CLadder stories."""
    import csv
    seen: dict[str, set] = {}
    with (ROOT / "data" / "cladder" / "full_v1.5_default.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("question_property") == "nonsense":
                continue
            for w in set(_W.findall(row["prompt"].lower())):
                seen.setdefault(w, set()).add(row["story_id"])
    return frozenset(w for w, s in seen.items() if len(s) >= MIN_STORIES)


@lru_cache(maxsize=64)
def _pattern(story: str):
    pairs = STORIES[story] + ANTI
    table = {}
    for old, new in pairs:
        table.setdefault(old.lower(), new)
    olds = sorted(table, key=len, reverse=True)
    rx = re.compile(r"(?<![A-Za-z'\-])(" + "|".join(re.escape(o) for o in olds) + r")(?![A-Za-z'\-])",
                    re.IGNORECASE)
    return rx, table


def _case(match: str, new: str) -> str:
    """Title-case matches give a title-case replacement; an acronym ("CEOs")
    does not, and sentence starts are capitalised afterwards."""
    if match[:1].isupper() and not match[:2].isupper():
        return new[:1].upper() + new[1:]
    return new


def rewrite(prompt: str, story: str) -> tuple[str, bool]:
    """The question told in its new story; clean is False when a word of the
    old story survives or nothing changed."""
    if story not in STORIES:
        return prompt, False
    rx, table = _pattern(story)
    out = rx.sub(lambda m: _case(m.group(0), table[m.group(0).lower()]), prompt)
    out = re.sub(r"(?<=[.!?:]\s)([a-z])", lambda m: m.group(1).upper(), out)
    out = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), out)
    return out, out != prompt and not leftovers(prompt, out, story)


def leftovers(original: str, rewritten: str, story: str = "") -> set:
    """Story words of the original that are still in the rewritten prompt."""
    tv = template_vocab()
    ok = KEEP_WORDS | SHARED.get(story, set())
    old = {w for w in _W.findall(original.lower()) if w not in tv and w not in ok}
    return old & set(_W.findall(rewritten.lower()))
