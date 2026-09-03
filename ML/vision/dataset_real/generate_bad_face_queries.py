import json

queries = []
def add_q(cat, sub, q, desc):
    queries.append({
        "category": f"face_tier1_bad_{cat}",
        "subreddit": sub,
        "query": q,
        "sort": "top",
        "time": "all",
        "description": desc
    })

# Men - Hair/Beard Disasters
add_q("men_botched_haircut", "malehairadvice", "botched haircut", "Men - Botched/terrible haircut")
add_q("men_receding_balding", "malehairadvice", "receding hairline balding", "Men - Severe balding/receding")
add_q("men_patchy_neckbeard", "malegrooming", "neckbeard patchy", "Men - Terrible patchy neckbeard")
add_q("men_greasy_messy_hair", "malehairadvice", "greasy messy hair", "Men - Unwashed/messy hair")
add_q("men_shave_it_off", "malegrooming", "should i shave it off", "Men - Desperate hair/beard advice")
add_q("men_older_balding_bad", "malehairadvice", "older balding mess", "Men O50 - Bad balding")

# Women - Hair/Makeup Disasters
add_q("women_fried_damaged_hair", "femalehairadvice", "fried damaged hair", "Women - Severely damaged/fried hair")
add_q("women_botched_haircut", "femalehairadvice", "crying botched haircut", "Women - Terrible haircut disaster")
add_q("women_frizzy_mess", "femalehairadvice", "frizzy mess", "Women - Unmanageable frizzy hair")
add_q("women_cakey_bad_makeup", "MakeupAddiction", "cakey makeup help", "Women - Bad/cakey makeup execution")
add_q("women_awful_eyebrows", "awfuleyebrows", "", "Women - Terribly shaped/drawn eyebrows")
add_q("women_overplucked_brows", "Eyebrows", "overplucked ruined", "Women - Overplucked/ruined eyebrows")
add_q("women_older_damaged_hair", "femalehairadvice", "older thinning damaged", "Women O50 - Thinning damaged hair")

# Both - Severe Skin Issues
add_q("both_severe_cystic_acne", "acne", "severe cystic acne", "Both - Severe acne breakouts")
add_q("both_bad_breakout", "SkincareAddiction", "worst breakout", "Both - Terrible skin breakouts")
add_q("both_acne_scars", "acne", "acne scars texture", "Both - Bad skin texture and scarring")

# Both - Facial Structure / Candid Ugly
add_q("both_amiugly_brutal", "amiuglybrutallyhonest", "", "Both - Brutally honest ugly feedback")
add_q("both_asymmetrical_face", "amiugly", "asymmetrical ugly", "Both - Highly asymmetrical features")
add_q("both_ugly_features", "truerateme", "below average 3", "Both - Below average facial ratings")
add_q("both_lookyourbest_desperate", "lookyourbest", "desperate ugly", "Both - Desperate for facial improvement")

out_file = Path(__file__).resolve().parent / 'reddit_bad_faces.json'
with open(out_file, 'w') as f:
    json.dump(queries, f, indent=2)

print(f"Generated {len(queries)} specific Tier-1 'Needs Improvement' face categories.")
