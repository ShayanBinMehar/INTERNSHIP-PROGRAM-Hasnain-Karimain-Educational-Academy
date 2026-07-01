"""Build expanded training and held-out test datasets."""

import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Existing 104 samples (kept as training seed)
EXISTING = """Text,Sentiment
"I love the design but hate the performance",Mixed
"This is amazing!",Positive
"It's okay, nothing special",Neutral
"Worst experience ever",Negative
"The product quality is excellent and I am very happy",Positive
"I absolutely hate this service",Negative
"The weather is normal today",Neutral
"Great features but terrible customer support",Mixed
"Best purchase I have ever made",Positive
"Completely disappointed with the results",Negative
"It works fine, nothing more nothing less",Neutral
"Love the price but hate the build quality",Mixed
"Outstanding performance and reliability",Positive
"Total waste of money",Negative
"The app is functional",Neutral
"Nice interface but slow loading times",Mixed
"Highly recommend this to everyone",Positive
"Never buying from them again",Negative
"Average product, meets basic expectations",Neutral
"Good battery life but poor camera quality",Mixed
"Exceeded all my expectations",Positive
"Broken on arrival, very upset",Negative
"The service was neither good nor bad",Neutral
"Beautiful design but expensive price tag",Mixed
"Five stars, absolutely perfect",Positive
"Horrible experience from start to finish",Negative
"Standard quality, nothing to complain about",Neutral
"Fast delivery but damaged packaging",Mixed
"Incredibly satisfied with my purchase",Positive
"Regret every penny spent on this",Negative
"Mediocre at best",Neutral
"Great taste but too much sugar",Mixed
"Life changing product, love it",Positive
"Awful quality and poor craftsmanship",Negative
"Acceptable for the price point",Neutral
"Friendly staff but long wait times",Mixed
"Pure joy using this every day",Positive
"Disgusting and unacceptable",Negative
"Neither impressed nor disappointed",Neutral
"Comfortable shoes but ugly color options",Mixed
"Brilliant innovation, well done",Positive
"Failed miserably at every task",Negative
"Just another ordinary item",Neutral
"Helpful tutorial but confusing interface",Mixed
"Thrilled with the outcome",Positive
"Angry and frustrated beyond words",Negative
"Fair enough, does what it says",Neutral
"Delicious food but rude waiter",Mixed
"Wonderful experience overall",Positive
"Pathetic and useless product",Negative
"The movie was average",Neutral
"Clean room but noisy neighbors",Mixed
"Absolutely delighted with the service",Positive
"Terrible decision to buy this",Negative
"Not bad, not great either",Neutral
"Smooth ride but uncomfortable seats",Mixed
"Perfect gift for my family",Positive
"Shocking lack of quality control",Negative
"Routine and unremarkable",Neutral
"Affordable price but cheap materials",Mixed
"Genuinely impressed by the results",Positive
"Disaster from day one",Negative
"Standard fare, nothing exciting",Neutral
"Quick response but unhelpful answers",Mixed
"Best in class performance",Positive
"Utterly useless and broken",Negative
"Middle of the road product",Neutral
"Stylish look but poor durability",Mixed
"Could not be happier with this",Positive
"Nightmare to use every single day",Negative
"Unremarkable but gets the job done",Neutral
"Fresh ingredients but small portions",Mixed
"Superb craftsmanship and attention to detail",Positive
"Ridiculously overpriced for what you get",Negative
"Plain and simple, no frills",Neutral
"Easy to use but lacks advanced features",Mixed
"Marvelous product, worth every cent",Positive
"Infuriating bugs ruin the experience",Negative
"Balanced and reasonable offering",Neutral
"Warm hospitality but cold food",Mixed
"Ecstatic about the new upgrade",Positive
"Deplorable service and attitude",Negative
"Indifferent to the whole experience",Neutral
"Powerful engine but high fuel consumption",Mixed
"Spectacular results every time",Positive
"Appalling condition upon delivery",Negative
"Run of the mill, nothing standout",Neutral
"Informative content but boring presentation",Mixed
"Over the moon with this purchase",Positive
"Despicable behavior from the company",Negative
"So so, could take it or leave it",Neutral
"Reliable performance but outdated design",Mixed
"Flawless execution and delivery",Positive
"Repulsive taste and texture",Negative
"Common and unexceptional",Neutral
"Convenient location but limited parking",Mixed
"Enthusiastic about the new features",Positive
"Detestable quality and finish",Negative
"Neither here nor there",Neutral
"Responsive team but missed deadlines",Mixed
"Radiant and beautiful design",Positive
"Loathsome experience throughout",Negative
"Passable quality for casual use",Neutral
"Crisp display but short battery life",Mixed"""

EXTRA_POSITIVE = [
    "I am thrilled with how well this works", "Customer support was friendly and helpful",
    "The update fixed every issue I had", "Delivery was fast and the packaging was perfect",
    "This app saves me hours every week", "The camera quality is stunning in daylight",
    "Battery life easily lasts the whole day", "Setup took less than five minutes",
    "The sound quality blew me away", "My team loves the new collaboration features",
    "Refund was processed quickly and fairly", "The tutorial made onboarding painless",
    "This course exceeded my learning goals", "The hotel room was spotless and comfortable",
    "Staff greeted us with genuine warmth", "The food tasted fresh and flavorful",
    "I would buy this again without hesitation", "Performance has been rock solid for months",
    "The interface feels polished and intuitive", "Shipping arrived two days early",
    "This laptop handles heavy workloads effortlessly", "The warranty coverage gives me peace of mind",
    "Every feature works exactly as advertised", "The concert was an unforgettable experience",
    "My kids enjoy using this every single day", "The workout plan produced visible results",
    "The documentation is clear and thorough", "I felt valued as a customer throughout",
    "The redesign looks modern and professional", "This restaurant never disappoints",
    "The plot kept me engaged from start to finish", "The support team resolved my ticket within an hour",
    "The fragrance is subtle and pleasant", "I appreciate how responsive the developers are",
    "The fabric feels soft and premium", "My presentation went smoothly thanks to this tool",
    "The trail was scenic and well maintained", "I wake up excited to use this product",
    "The coaching session was motivating and practical", "The software update added useful features",
    "The checkout process was quick and secure", "This book changed how I think about the topic",
    "The massage was relaxing and professional", "The team delivered ahead of schedule",
    "I trust this brand after years of use", "The colors on the display are vivid and accurate",
    "The recipe turned out perfectly on the first try", "The onboarding call answered all my questions",
    "The commute is much easier with this gadget", "The playlist recommendation was spot on",
    "The cabin was cozy and beautifully decorated", "The game runs smoothly on my older phone",
    "The flowers arrived fresh and on time", "The lecture was insightful and well organized",
    "The jacket keeps me warm in cold weather", "The podcast episode was informative and fun",
    "The repair shop did an excellent job", "The subscription is worth every penny",
    "The tour guide was knowledgeable and engaging", "The vacuum picks up pet hair effortlessly",
    "The webinar provided actionable advice", "The yoga class left me feeling energized",
    "This keyboard is a joy to type on", "This moisturizer improved my skin quickly",
    "This phone camera captures sharp night photos", "This tea blend tastes perfectly balanced",
    "This watch looks elegant and keeps accurate time", "We received compliments on the new decor",
    "Your product made our launch a success", "The training session was clear and engaging",
    "The new dashboard is clean and easy to navigate", "I finally found a shampoo that works for me",
    "The dentist explained everything clearly", "The puppy adapted to this food immediately",
    "The charity uses donations transparently", "The dentist office respects appointment times",
    "The suitcase survived a rough trip without damage", "This therapist helped me make real progress",
    "This umbrella held up in strong wind", "This vendor honored the quoted price",
    "The discount made this an unbeatable deal", "The garden looks vibrant after using this fertilizer",
]

EXTRA_NEGATIVE = [
    "The product stopped working after one week", "Customer service ignored my emails for days",
    "The app crashes every time I open settings", "Shipping took three weeks longer than promised",
    "The battery drains in under two hours", "The screen cracked without any drop",
    "I was charged twice and nobody fixed it", "The food arrived cold and undercooked",
    "The room smelled like mildew upon check in", "The instructor was rude and dismissive",
    "The software deleted my files during update", "The return policy is misleading and unfair",
    "The seat broke the first time I sat down", "The paint started peeling within a month",
    "The wifi drops constantly during meetings", "The zipper snapped on the first use",
    "They canceled my order without explanation", "The doctor rushed through the appointment",
    "The exam questions did not match the material", "The package arrived opened and incomplete",
    "The refund request was denied unfairly", "The noise level made sleep impossible",
    "The contract included hidden fees", "The map directions were completely wrong",
    "The printer jams on every other page", "The shoes gave me blisters on day one",
    "The subscription renewed without warning", "The ticket website crashed during checkout",
    "The tool broke under normal pressure", "The tour started an hour late with no apology",
    "The translation quality was embarrassingly bad", "The update made the app slower than before",
    "The video buffer loop ruined the live stream", "The waiter forgot our order twice",
    "The website login fails with correct credentials", "Their quality has declined sharply lately",
    "This airline lost my luggage again", "This blender leaks from the bottom every time",
    "This bus was overcrowded and unbearably hot", "This cable frayed after minimal bending",
    "This camera autofocus hunts endlessly", "This car repair cost far more than quoted",
    "This chair wobbles dangerously", "This coffee tastes burnt and bitter",
    "This course was outdated and poorly structured", "This dryer stopped heating after a month",
    "This fan is louder than my old one", "This game is pay to win and frustrating",
    "This gym equipment is constantly broken", "This headset mic picks up static constantly",
    "This heater failed during a cold night", "This hotel overbooked and moved us to a worse room",
    "This jacket leaked in light rain", "This keyboard misses keystrokes randomly",
    "This laptop overheats during basic tasks", "This lock jammed with the keys inside",
    "This mattress sags badly after six months", "This monitor flickers at lower brightness",
    "This mop left streaks on every surface", "This movie was boring and predictable",
    "This oven temperature is wildly inaccurate", "This park was littered and poorly maintained",
    "This pen runs out of ink immediately", "This phone freezes several times a day",
    "This plugin broke my entire website", "This policy change feels like a bait and switch",
    "This restaurant served stale bread", "This router drops connection every hour",
    "This seller shipped a counterfeit item", "This shampoo dried out my scalp",
    "This speaker distorts at medium volume", "This store refused a valid warranty claim",
    "This stroller wheel fell off in the park", "This tablet screen is unresponsive at the edges",
    "This tent leaked during a light drizzle", "This toaster burns one side every time",
    "This vacuum lost suction within weeks", "This washer shakes violently on spin cycle",
    "Waiting on hold for ninety minutes was unacceptable", "The medication caused worse side effects",
    "The taxi driver took an unnecessarily long route", "This gate agent was condescending and unhelpful",
    "This landlord never responds to maintenance requests", "This lawyer missed critical filing deadlines",
    "This nurse was dismissive of my symptoms", "This pharmacy gave me the wrong dosage",
    "This scanner produces blurry images", "This survey felt manipulative and pushy",
    "This taxi smelled terrible inside", "This thermostat reads the wrong temperature",
    "This ticket was impossible to redeem", "This trainer pushed unsafe exercises",
    "This trip insurance denied a valid claim", "This tutor canceled repeatedly at the last minute",
    "This vendor ghosted us after partial payment", "This wallet stitching came undone quickly",
    "This webinar was mostly sales pitch", "This wig tangled immediately and looked fake",
    "This workshop wasted an entire afternoon", "This zipper caught fabric and jammed shut",
]

EXTRA_NEUTRAL = [
    "The meeting lasted thirty minutes", "The bus arrived at the scheduled time",
    "The document is three pages long", "The store closes at nine pm",
    "The package weighs about two kilograms", "The event starts on Friday at noon",
    "The file uploaded without any message", "The form asks for name and address",
    "The lecture covered chapter four", "The menu includes pasta and salad options",
    "The office is located on the second floor", "The report was submitted before deadline",
    "The route takes roughly twenty minutes", "The software version is two point one",
    "The temperature today is fifteen degrees", "The ticket costs twelve dollars",
    "The train departs from platform three", "The update installed and restarted the app",
    "The website lists products by category", "There are four buttons on the home screen",
    "This app requires an account to sign in", "This book has twelve chapters in total",
    "This building was constructed in nineteen ninety", "This cable is one meter long",
    "This calendar shows appointments by week", "This chair has adjustable height",
    "This chart displays sales by region", "This class meets twice per week",
    "This coffee contains caffeine", "This color is labeled navy blue",
    "This container holds five hundred milliliters", "This course lasts eight weeks",
    "This dashboard shows the default layout", "This device supports bluetooth and wifi",
    "This folder contains archived invoices", "This gallery opens at ten in the morning",
    "This gate closes five minutes before departure", "This handbook explains company policies",
    "This invoice lists tax separately", "This jacket comes in three sizes",
    "This label lists ingredients in order", "This lamp uses a standard bulb",
    "This leaflet describes the museum hours", "This library card expires next year",
    "This listing shows square footage and price", "This log file records system events",
    "This magazine publishes monthly issues", "This map marks restrooms and exits",
    "This module covers basic statistics", "This notebook has two hundred pages",
    "This notice was posted in the lobby", "This page requires scrolling to continue",
    "This parking lot accepts card payment", "This passport expires in twenty twenty seven",
    "This photo was taken last summer", "This playlist has twenty songs",
    "This policy applies to all employees", "This postcard arrived without a stamp",
    "This product manual is written in English", "This program runs on Windows and Mac",
    "This receipt shows the transaction date", "This schedule lists classes by weekday",
    "This section describes return procedures", "This shelf holds about ten books",
    "This sign indicates one way traffic", "This spreadsheet tracks monthly expenses",
    "This stamp cost fifty cents", "This store accepts cash and cards",
    "This syllabus outlines weekly topics", "This table seats four people comfortably",
    "This tag shows the item code", "This template includes placeholder text",
    "This terminal serves domestic flights", "This ticket is valid for one entry",
    "This timetable lists departure intervals", "This toolbox contains basic tools",
    "This topic will be covered next week", "This tour lasts approximately one hour",
    "This tray holds standard letter paper", "This unit measures temperature in Celsius",
    "This user guide explains installation steps", "This version number appears in settings",
    "This voucher expires at the end of month", "This wall is painted light gray",
    "This warehouse stores seasonal inventory", "This warranty lasts twelve months",
    "This webpage loads in about two seconds", "This week includes a public holiday",
    "This widget displays current weather", "This workbook contains practice exercises",
    "This workshop is scheduled for Thursday", "This yard measures twenty meters wide",
    "Traffic was moderate during the commute", "The agenda includes three discussion items",
    "The archive stores records by year",     "The assignment is due next Monday",
]

BOOST_NEGATIVE = [
    "I hate this product with a passion", "This is the worst service I have encountered",
    "Absolutely disgusting and unacceptable quality", "I am furious about this purchase",
    "This item is completely broken and useless", "Terrible experience, never again",
    "The worst decision I have ever made", "I despise how poorly this works",
    "Horrible quality, total scam", "This made me angry and disappointed",
    "Awful taste, inedible and revolting", "I regret buying this immediately",
    "Pathetic support and rude staff", "This product failed within hours",
    "Disgusting smell and filthy condition", "I am outraged by the poor quality",
    "Unacceptable performance and constant crashes", "This company treats customers terribly",
    "Miserable experience from beginning to end", "I hate every minute using this",
]

BOOST_NEUTRAL = [
    "The file contains twelve rows of data", "The meeting room has eight chairs",
    "The bus route includes six stops", "This product weighs two hundred grams",
    "The class schedule lists five weekly sessions", "The form requires name and email",
    "The report covers quarterly sales figures", "This device uses a USB-C port",
    "The building has three floors and an elevator", "The ticket price is fifteen dollars",
    "The manual describes installation in four steps", "This page lists contact information",
    "The survey contains ten multiple choice questions", "The package dimensions are ten by five cm",
    "The software version is four point two", "The lecture notes cover chapter seven",
    "This account was registered in January", "The store hours are nine to six",
    "The document header shows the company logo", "The chart plots values over time",
]

BOOST_POSITIVE = [
    "I absolutely love this product", "This exceeded my highest expectations",
    "Fantastic quality and superb performance", "I am delighted with this purchase",
    "Outstanding service and friendly staff", "This works beautifully every day",
    "I highly recommend this to everyone", "Perfect solution for my needs",
    "Brilliant design and flawless execution", "I am thrilled with the results",
    "Excellent value and amazing features", "This made my life so much easier",
    "Impressive build quality and reliability", "I could not be more satisfied",
    "Wonderful experience, five stars", "This product is truly exceptional",
    "I love how intuitive and fast this is", "Superb customer care and quick delivery",
    "A joy to use from day one", "This is exactly what I hoped for",
]

BOOST_MIXED = [
    "Love the idea but hate the implementation", "Great quality but terrible price",
    "Nice design but poor battery performance", "Helpful staff but slow service",
    "Good features but buggy software", "Tasty food but very small portions",
    "Comfortable seat but noisy cabin", "Useful tool but steep learning curve",
    "Fast shipping but damaged box", "Beautiful view but long crowded queue",
    "Reliable car but expensive maintenance", "Clear sound but uncomfortable earbuds",
    "Great content but too many ads", "Handy app but confusing settings menu",
    "Nice hotel but far from downtown", "Good camera but slow autofocus",
    "Friendly guide but rushed tour", "Solid laptop but heavy to carry",
    "Fresh salad but oily dressing", "Smart features but privacy worries",
]

EXTRA_MIXED = [
    "I like the features but the app is very slow", "Great price but disappointing durability",
    "The staff was kind but the room was dirty", "Beautiful scenery but exhausting hike",
    "Interesting story but weak ending", "Fast shipping but wrong item sent",
    "Comfortable fit but colors fade quickly", "Useful content but poor audio quality",
    "Tasty meal but portions were too small", "Modern design but confusing navigation",
    "Reliable engine but uncomfortable seats", "Helpful guide but rushed explanations",
    "Affordable tickets but crowded venue", "Smooth checkout but limited payment options",
    "Bright screen but heavy to carry daily", "Good location but noisy at night",
    "Friendly host but apartment needs repairs", "Clear instructions but missing parts",
    "Efficient service but impersonal tone", "Fun gameplay but too many ads",
    "Solid build but outdated software", "Nice aroma but artificial aftertaste",
    "Quick install but frequent update prompts", "Great concept but sloppy execution",
    "Comfortable bed but thin walls", "Impressive specs but overheats under load",
    "Polite reception but long waiting time", "Fresh produce but high prices",
    "Easy booking but strict cancellation rules", "Stylish packaging but fragile contents",
    "Strong performance but loud fan noise", "Good chemistry but predictable plot",
    "Handy size but small storage capacity", "Warm welcome but cold main course",
    "Reliable alarm but complicated setup", "Nice photos but inaccurate description",
    "Smooth ride but bumpy suspension on hills", "Great tutorial but limited free content",
    "Attractive offer but hidden renewal fee", "Clean kitchen but slow service",
    "Responsive buttons but short battery life", "Good value but cheap accessories included",
    "Well organized event but poor signage", "Soft fabric but shrinks after washing",
    "Fast response but incomplete solution", "Delightful flavor but too much salt",
    "Compact design but cramped keyboard", "Trustworthy brand but declining support quality",
    "Excellent trainer but overcrowded classes", "Pretty cover but dull writing style",
    "Stable connection but weak signal at range", "Generous trial but aggressive upsell later",
    "Accurate tracking but bulky on the wrist", "Nice campus but expensive tuition",
    "Good warranty but slow repair turnaround", "Fun multiplayer but buggy solo mode",
    "Clear voice but robotic pronunciation", "Handy features but steep learning curve",
    "Great first impression but quality dropped later", "Nice gift wrap but arrived a day late",
    "Reliable backups but confusing restore process", "Good traction but wears out quickly",
    "Beautiful artwork but repetitive levels", "Helpful community but toxic comment sections",
    "Smooth animation but occasional frame drops", "Great starter kit but lacks advanced tools",
    "Comfortable grip but slippery when wet", "Informative tour but too much walking",
    "Nice hotel lobby but small bathroom", "Good interview but low salary offer",
    "Useful integrations but limited customization", "Tasty sauce but overcooked noodles",
    "Reliable schedule but old train cars", "Friendly nurse but long ER wait",
    "Clear video but echo in the audio", "Handy shortcuts but cluttered menu",
    "Great opening act but weak headliner", "Nice scent but irritated my skin",
    "Fast printer but expensive ink cartridges", "Good documentation but no sample projects",
    "Comfortable headphones but weak noise isolation", "Beautiful cover but pages stick together",
    "Reliable GPS but outdated map data", "Nice discount but minimum order too high",
    "Smooth onboarding but weak default settings", "Great camera but sluggish photo processing",
    "Helpful reminders but too many notifications", "Good flavor variety but inconsistent quality",
    "Stable app but ugly user interface", "Nice instructor but outdated course videos",
    "Quick delivery but dented outer box", "Good battery but slow charging speed",
    "Comfortable sandals but narrow toe box", "Useful dashboard but limited export formats",
    "Great voice assistant but privacy concerns", "Nice hotel breakfast but small dining area",
    "Reliable lock but tricky key alignment", "Good plot twist but rushed finale",
    "Handy travel size but leaks slightly", "Beautiful park but limited parking spaces",
    "Fast customer chat but generic copy paste answers", "Good fitness tracking but inaccurate sleep data",
]

TEST_POSITIVE = [
    "I am overjoyed with this purchase", "The service exceeded every expectation",
    "This is the finest product in its category", "Our team celebrates this tool daily",
    "The experience was seamless from start to finish", "I recommend this to all my colleagues",
    "The quality feels premium in every detail", "This made a difficult task surprisingly easy",
    "I received compliments on the results immediately", "The upgrade was absolutely worth it",
    "Everything arrived exactly as described", "The support agent solved my issue in minutes",
    "This gift brought genuine happiness", "The performance boost was immediately noticeable",
    "I feel confident choosing this brand again", "The flavor profile is rich and satisfying",
    "The layout is intuitive for new users", "This course helped me land a promotion",
    "The hotel exceeded our anniversary expectations", "The device feels sturdy and well engineered",
]

TEST_NEGATIVE = [
    "This is the worst product I have ever used", "I want a refund and will never return",
    "The experience was insulting and unprofessional", "Nothing about this worked as promised",
    "I feel cheated by misleading advertising", "The item broke before the first use",
    "Support hung up on me twice", "The smell alone made me return it",
    "This wasted an entire weekend of my time", "I am furious about the hidden charges",
    "The instructor mocked participants openly", "The software corrupted my project files",
    "This meal was inedible and overpriced", "The room had insects upon arrival",
    "They lost my reservation and blamed me", "The repair made the problem worse",
    "This policy is designed to trap customers", "The delivery driver threw the package",
    "I would rate this zero stars if I could", "The noise never stopped all night",
]

TEST_NEUTRAL = [
    "The seminar ended at four pm", "This bottle holds seven hundred fifty milliliters",
    "The bus stop is across the street", "The form has ten required fields",
    "The chapter discusses supply and demand", "This account was created in March",
    "The shelf is made of pine wood", "The event calendar shows six upcoming dates",
    "This route has four stops total", "The label indicates machine wash cold",
    "The invoice number appears at the top", "This keyboard layout is QWERTY",
    "The museum ticket allows one reentry", "This file format is compatible with PDF",
    "The meeting room fits twelve chairs", "This app version is three point four",
    "The package tracking shows in transit", "The course syllabus lists weekly readings",
    "This bridge connects the two districts", "The clock displays twenty four hour time",
]

TEST_MIXED = [
    "Love the concept but hate the execution", "Cheap price but questionable build quality",
    "Great speaker but awful venue acoustics", "Nice staff but the product failed quickly",
    "Beautiful packaging but mediocre contents", "Fast app but confusing account settings",
    "Good workout but painful knee strain", "Excellent coffee but rude barista",
    "Handy tool but missing essential attachments", "Fun event but terrible parking situation",
    "Reliable watch but uncomfortable band", "Interesting book but dragging middle section",
    "Great discount but strict no return policy", "Clean UI but missing key integrations",
    "Tasty appetizer but bland main dish", "Nice beach but overcrowded on weekends",
    "Strong wifi but uncomfortable desk chair", "Good acting but weak script overall",
    "Compact car but very little trunk space", "Helpful FAQ but no live support option",
]


def parse_existing():
    lines = EXISTING.strip().splitlines()[1:]
    rows = []
    for line in lines:
        if not line.strip():
            continue
        # CSV with quoted text
        text = line.split('",')[0][1:]
        sentiment = line.split(",")[-1].strip()
        rows.append((text, sentiment))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "Sentiment"])
        for text, sentiment in rows:
            writer.writerow([text, sentiment])


def generate_extra_samples() -> dict[str, list[str]]:
    """Generate additional template-based samples for each sentiment class."""
    generated: dict[str, list[str]] = {
        "Positive": [],
        "Negative": [],
        "Neutral": [],
        "Mixed": [],
    }

    pos_adj = [
        "excellent", "amazing", "outstanding", "wonderful", "fantastic",
        "superb", "brilliant", "delightful", "impressive", "perfect",
    ]
    pos_nouns = [
        "product", "service", "experience", "quality", "performance",
        "support", "design", "feature", "purchase", "upgrade",
    ]
    for adj in pos_adj:
        for noun in pos_nouns[:5]:
            generated["Positive"].append(f"This {noun} is {adj} and I love it")
            generated["Positive"].append(f"Truly {adj} {noun}, highly recommended")

    neg_adj = [
        "terrible", "awful", "horrible", "disgusting", "pathetic",
        "useless", "broken", "unacceptable", "dreadful", "atrocious",
    ]
    for adj in neg_adj:
        for noun in pos_nouns:
            generated["Negative"].append(f"This {noun} is {adj} and I hate it")
            generated["Negative"].append(f"Absolutely {adj} {noun}, never again")

    neutral_nouns = [
        "document", "report", "file", "module", "chart", "table", "form",
        "label", "manual", "schedule", "invoice", "ticket", "folder",
        "catalog", "brochure", "leaflet", "worksheet", "template", "guide",
    ]
    neutral_details = [
        "contains four sections", "includes basic information", "lists ten items",
        "shows the default view", "has three pages", "displays current settings",
        "records daily entries", "stores archived data", "uses standard format",
        "requires user login", "supports common file types", "opens in a new tab",
    ]
    for noun in neutral_nouns:
        for detail in neutral_details[:4]:
            generated["Neutral"].append(f"This {noun} {detail}")
            generated["Neutral"].append(f"The {noun} {detail}")

    mixed_pos = [
        "great", "good", "nice", "useful", "fast", "beautiful", "helpful", "solid",
        "clean", "smart", "compact", "fresh", "bright", "smooth", "stable",
    ]
    mixed_neg = [
        "bad", "slow", "ugly", "confusing", "expensive", "noisy", "fragile", "weak",
        "dated", "crowded", "rigid", "stale", "dim", "buggy", "unstable",
    ]
    mixed_nouns = [
        "app", "hotel", "phone", "course", "restaurant", "laptop", "camera", "tablet",
        "watch", "speaker", "keyboard", "monitor", "backpack", "jacket", "shoes",
    ]
    for pos in mixed_pos:
        for neg in mixed_neg[:8]:
            for noun in mixed_nouns[:6]:
                generated["Mixed"].append(f"{pos.capitalize()} {noun} but {neg} performance")
                generated["Mixed"].append(f"I like the {noun} but hate the {neg} quality")

    return generated


def main():
    existing = parse_existing()
    by_label = {label: [] for label in ("Positive", "Negative", "Neutral", "Mixed")}
    for text, label in existing:
        by_label[label].append(text)

    extras = {
        "Positive": EXTRA_POSITIVE + BOOST_POSITIVE,
        "Negative": EXTRA_NEGATIVE + BOOST_NEGATIVE,
        "Neutral": EXTRA_NEUTRAL + BOOST_NEUTRAL,
        "Mixed": EXTRA_MIXED + BOOST_MIXED,
    }
    for label, texts in extras.items():
        seen = set(by_label[label])
        for text in texts:
            if text not in seen:
                by_label[label].append(text)
                seen.add(text)

    for label, texts in generate_extra_samples().items():
        seen = set(by_label[label])
        for text in texts:
            if text not in seen:
                by_label[label].append(text)
                seen.add(text)

    # Target 200 per class for training
    train_rows = []
    for label in ("Positive", "Negative", "Neutral", "Mixed"):
        samples = by_label[label][:200]
        train_rows.extend((t, label) for t in samples)

    test_rows = []
    for label, texts in [
        ("Positive", TEST_POSITIVE),
        ("Negative", TEST_NEGATIVE),
        ("Neutral", TEST_NEUTRAL),
        ("Mixed", TEST_MIXED),
    ]:
        test_rows.extend((t, label) for t in texts)

    write_csv(os.path.join(BASE, "dataset.csv"), train_rows)
    write_csv(os.path.join(BASE, "test_samples.csv"), test_rows)

    print(f"dataset.csv: {len(train_rows)} samples")
    print(f"test_samples.csv: {len(test_rows)} samples (held-out)")


if __name__ == "__main__":
    main()
