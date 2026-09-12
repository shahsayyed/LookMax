#!/usr/bin/env python3
"""
generate_expansion_queries.py — Generates targeted query specifications for Reddit scraping.

Directly addresses the data deficits identified in the LookMax vision models:
1. Priority 1: Women Outfit Tier 1 ("Needs Improvement") acute deficit (only 137 images in current training set).
2. Priority 1b: Women Outfit Mature Demographics (35-50 and Over 50).
3. Priority 2: Men Outfit Over 50 and Tier 1 fit critiques.
4. Priority 3: Grooming / styling in mature age brackets.
5. Priority 4: Rotated uncapped feeds to fill remaining budget.

All queries default to a balanced limit (40 images) to guarantee breadth across
all missing cohorts within a 3,000-image session budget.

Outputs:
  ML/vision/dataset_real/reddit_expansion_queries.json
"""

import json
from pathlib import Path

queries = []


def add_q(
    cat_key: str,
    sub: str,
    query: str = None,
    sort: str = "top",
    time_filter: str = "all",
    listing: str = "hot",
    desc: str = "",
    limit: int = 40,
):
    item = {
        "category": cat_key,
        "subreddit": sub,
        "sort": sort,
        "time": time_filter,
        "description": desc,
        "limit": limit,
    }
    if query:
        item["query"] = query
    else:
        item["listing"] = listing
    queries.append(item)


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 1: High-Volume Real Fit Critiques (Targeting Tier 1 & 2 Women Outfits)
# Subreddits: r/weddingattireapproval, r/capsulewardrobe, r/oldhagfashion,
#             r/businesscasual, r/PlusSizeFashion, r/Midsizefashion, r/PetiteFashion
# ══════════════════════════════════════════════════════════════════════════════

# r/weddingattireapproval (massive stream of women 20s-60s asking for honest fit critiques)
add_q("women_fit_wedding_too_casual", "weddingattireapproval", query="too casual", sort="relevance", time_filter="all", desc="Women Outfit — Questions on underdressed / too casual outfits (Needs Improvement)")
add_q("women_fit_wedding_too_tight", "weddingattireapproval", query="too tight", sort="relevance", time_filter="all", desc="Women Outfit — Fit questions regarding overly tight dresses (Needs Improvement)")
add_q("women_fit_wedding_too_baggy", "weddingattireapproval", query="too baggy", sort="relevance", time_filter="all", desc="Women Outfit — Fit questions regarding oversized / ill-fitting dresses (Needs Improvement)")
add_q("women_fit_wedding_alterations", "weddingattireapproval", query="alterations needed", sort="relevance", time_filter="all", desc="Women Outfit — Garments needing alterations / tailoring flaws (Needs Improvement)")
add_q("women_fit_wedding_does_this_work", "weddingattireapproval", query="does this work", sort="new", time_filter="all", desc="Women Outfit — Candid outfit evaluation questions (Diverse tiers)")
add_q("women_fit_wedding_flattering", "weddingattireapproval", query="flattering or not", sort="relevance", time_filter="all", desc="Women Outfit — Candid silhouette and styling checks")
add_q("women_fit_wedding_mother_bride", "weddingattireapproval", query="mother of the bride", sort="top", time_filter="year", desc="Women Over 50 — Formal dress fit checks (Mature demographics)")
add_q("women_fit_wedding_mother_groom", "weddingattireapproval", query="mother of the groom", sort="top", time_filter="year", desc="Women Over 50 — Mature formal attire fit checks")
add_q("women_fit_wedding_mature_guest", "weddingattireapproval", query="mature guest dress", sort="relevance", time_filter="all", desc="Women 35-50 / Over 50 — Mature guest attire")
add_q("women_fit_wedding_hot_feed", "weddingattireapproval", listing="hot", desc="Women Outfit — Active daily wedding attire feedback feed")

# r/oldhagfashion (vibrant community with heavy representation of women 35+, 40+, 50+, 60+)
add_q("women_mature_oldhag_fit_check_new", "oldhagfashion", query="fit check", sort="new", time_filter="all", desc="Women 35+ / Over 50 — Daily fit checks (Diverse tiers)")
add_q("women_mature_oldhag_work_outfit", "oldhagfashion", query="work outfit", sort="top", time_filter="year", desc="Women 35+ / Over 50 — Work and office styling")
add_q("women_o50_oldhag_over_50", "oldhagfashion", query="over 50", sort="top", time_filter="all", desc="Women Over 50 — Self-identified 50+ fashion posters")
add_q("women_o50_oldhag_over_60", "oldhagfashion", query="over 60", sort="top", time_filter="all", desc="Women Over 50 — Self-identified 60+ fashion posters")
add_q("women_35_50_oldhag_over_40", "oldhagfashion", query="over 40", sort="top", time_filter="all", desc="Women 35-50 — Self-identified 40s styling")
add_q("women_mature_oldhag_awkward_fit", "oldhagfashion", query="awkward fit", sort="relevance", time_filter="all", desc="Women Outfit — Experimental / ill-proportioned styling (Needs Improvement)")
add_q("women_mature_oldhag_hot_feed", "oldhagfashion", listing="hot", desc="Women 35+ / Over 50 — Unfiltered daily outfit feed")

# r/capsulewardrobe (predominantly professional women in 30s-50s focusing on fit, basics, and workwear)
add_q("women_35_50_capsule_fit_check", "capsulewardrobe", query="fit check", sort="new", time_filter="all", desc="Women 35-50 — Minimalist and workwear fit checks")
add_q("women_35_50_capsule_workwear", "capsulewardrobe", query="office workwear", sort="top", time_filter="year", desc="Women 35-50 — Everyday office attire")
add_q("women_35_50_capsule_does_this_fit", "capsulewardrobe", query="does this fit", sort="relevance", time_filter="all", desc="Women 35-50 — Candid fit evaluation (Needs Improvement / Average)")
add_q("women_35_50_capsule_proportions", "capsulewardrobe", query="proportions critique", sort="relevance", time_filter="all", desc="Women 35-50 — Silhouette and proportion feedback")
add_q("women_35_50_capsule_top_year", "capsulewardrobe", listing="top", time_filter="year", desc="Women 35-50 — Top capsule wardrobe outfits past year")

# r/businesscasual (office and workplace attire across men and women 25-55)
add_q("women_35_50_bizcas_blazer_fit", "businesscasual", query="women blazer fit", sort="relevance", time_filter="all", desc="Women 35-50 — Blazer and jacket tailoring checks")
add_q("women_35_50_bizcas_trousers_fit", "businesscasual", query="women trousers pants fit", sort="relevance", time_filter="all", desc="Women 35-50 — Slacks and trouser fit checks")
add_q("women_35_50_bizcas_critique", "businesscasual", query="women work outfit critique", sort="relevance", time_filter="all", desc="Women 35-50 — Office outfit feedback")
add_q("women_35_50_bizcas_first_day", "businesscasual", query="first day outfit check", sort="relevance", time_filter="all", desc="Women 35-50 — First day / interview outfits")
add_q("men_35_50_bizcas_blazer_trousers", "businesscasual", query="men blazer chinos fit", sort="top", time_filter="year", desc="Men 35-50 — Business casual blazer chinos fit")
add_q("men_35_50_bizcas_slacks_check", "businesscasual", query="pants trousers fit check", sort="relevance", time_filter="all", desc="Men 35-50 — Pant length and tailoring feedback")

# r/PlusSizeFashion (active styling discussions and fit critiques)
add_q("women_plussize_fit_check_new", "PlusSizeFashion", query="fit check", sort="new", time_filter="all", desc="Women Outfit — Fit check posts (Diverse tiers)")
add_q("women_plussize_work_outfit", "PlusSizeFashion", query="work outfit critique", sort="top", time_filter="year", desc="Women Outfit — Work outfits and office fit")
add_q("women_plussize_does_this_flatter", "PlusSizeFashion", query="does this flatter", sort="relevance", time_filter="all", desc="Women Outfit — Candid fit evaluation (Needs Improvement / Average)")
add_q("women_plussize_honest_opinion", "PlusSizeFashion", query="honest opinion fit", sort="relevance", time_filter="all", desc="Women Outfit — Critical fit feedback")
add_q("women_35_50_plussize_over_40", "PlusSizeFashion", query="over 40 outfit", sort="top", time_filter="all", desc="Women 35-50 — Mature plus size styling")
add_q("women_o50_plussize_over_50", "PlusSizeFashion", query="over 50 outfit", sort="top", time_filter="all", desc="Women Over 50 — Mature plus size styling")
add_q("women_plussize_hot_feed", "PlusSizeFashion", listing="hot", desc="Women Outfit — Active daily plus size feed")

# r/Midsizefashion (everyday casual and work styling)
add_q("women_midsize_fit_check_new", "Midsizefashion", query="fit check", sort="new", time_filter="all", desc="Women Outfit — Fit checks")
add_q("women_midsize_workwear", "Midsizefashion", query="workwear critique", sort="top", time_filter="year", desc="Women Outfit — Workwear styling")
add_q("women_midsize_honest_thoughts", "Midsizefashion", query="honest thoughts outfit", sort="relevance", time_filter="all", desc="Women Outfit — Candid feedback")

# r/petitefashionadvice & r/PetiteFashion (tailoring, proportion issues, pant lengths)
add_q("women_petite_pants_too_long", "petitefashionadvice", query="pants too long fit check", sort="relevance", time_filter="all", desc="Women Outfit — Hemming / length issues (Needs Improvement)")
add_q("women_petite_tailoring_advice", "petitefashionadvice", query="tailoring advice blazer", sort="relevance", time_filter="all", desc="Women Outfit — Tailoring checks (Needs Improvement / Average)")
add_q("women_petite_proportions_critique", "petitefashionadvice", query="proportions critique outfit", sort="relevance", time_filter="all", desc="Women Outfit — Proportion critiques")
add_q("women_35_50_petite_over_40", "petitefashionadvice", query="over 40 fit check", sort="top", time_filter="all", desc="Women 35-50 — Petite mature styling")
add_q("women_o50_petite_over_50", "petitefashionadvice", query="over 50 style", sort="top", time_filter="all", desc="Women Over 50 — Petite mature styling")
add_q("women_petite_hot_feed", "petitefashionadvice", listing="hot", desc="Women Outfit — Petite feed")

# r/fashionadvice (general styling and fit questions across demographics)
add_q("women_fashionadv_critique", "fashionadvice", query="outfit critique honest", sort="relevance", time_filter="all", desc="Women Outfit — Honest outfit critique requests")
add_q("women_fashionadv_does_this_look_good", "fashionadvice", query="does this look good on me", sort="relevance", time_filter="all", desc="Women Outfit — Candid outfit questions")
add_q("women_fashionadv_how_to_improve", "fashionadvice", query="how can i improve this outfit", sort="relevance", time_filter="all", desc="Women Outfit — Improvement-seeking fit checks (Needs Improvement)")
add_q("women_35_50_fashionadv_over_40", "fashionadvice", query="over 40 style advice", sort="top", time_filter="all", desc="Women 35-50 — Style upgrade advice")
add_q("women_o50_fashionadv_over_50", "fashionadvice", query="over 50 outfit advice", sort="top", time_filter="all", desc="Women Over 50 — Mature style advice")


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 2: Mature & Diverse Menswear (Targeting Men Over 50 and 35-50)
# Subreddits: r/mensfashion, r/malefashionadvice, r/bigmenfashionadvice, r/vintagefashion
# ══════════════════════════════════════════════════════════════════════════════

# r/mensfashion (mature focus & natural fit critique terms)
add_q("men_o50_mensfashion_over_50", "mensfashion", query="over 50 fit check", sort="top", time_filter="all", desc="Men Over 50 — Fit check from older men")
add_q("men_o50_mensfashion_over_60", "mensfashion", query="over 60 style", sort="top", time_filter="all", desc="Men Over 50 — Senior style fit checks")
add_q("men_o50_mensfashion_older_man", "mensfashion", query="older man fit check", sort="relevance", time_filter="all", desc="Men Over 50 — Mature fit critiques")
add_q("men_35_50_mensfashion_over_40", "mensfashion", query="over 40 fit check", sort="top", time_filter="all", desc="Men 35-50 — Mid-career fit checks")
add_q("men_35_50_mensfashion_dad_upgrade", "mensfashion", query="dad style upgrade", sort="relevance", time_filter="all", desc="Men 35-50 — Style upgrade advice (Needs Improvement / Average)")
add_q("men_tier1_mensfashion_suit_too_big", "mensfashion", query="suit jacket too big", sort="relevance", time_filter="all", desc="Men Outfit — Ill-fitting suit jacket (Needs Improvement)")
add_q("men_tier1_mensfashion_pants_too_baggy", "mensfashion", query="pants too baggy fit check", sort="relevance", time_filter="all", desc="Men Outfit — Oversized pants fit check (Needs Improvement)")
add_q("men_tier1_mensfashion_alterations", "mensfashion", query="alterations advice fit check", sort="relevance", time_filter="all", desc="Men Outfit — Tailoring issues (Needs Improvement)")
add_q("men_mensfashion_hot_feed", "mensfashion", listing="hot", desc="Men Outfit — Active daily mensfashion feed")
add_q("men_mensfashion_new_feed", "mensfashion", listing="new", desc="Men Outfit — Fresh unfiltered mensfashion feed")

# r/malefashionadvice (uncapped natural queries)
add_q("men_mfa_fit_check_new", "malefashionadvice", query="fit check", sort="new", time_filter="all", desc="Men Outfit — Fresh fit checks")
add_q("men_mfa_first_suit_new", "malefashionadvice", query="first suit fit check", sort="new", time_filter="all", desc="Men Outfit — First suit trials (Needs Improvement)")
add_q("men_mfa_tailoring_critique", "malefashionadvice", query="tailoring critique suit", sort="relevance", time_filter="all", desc="Men Outfit — Tailoring critique")
add_q("men_o50_mfa_older_man", "malefashionadvice", query="older man advice fit check", sort="relevance", time_filter="all", desc="Men Over 50 — Mature MFA styling")

# r/bigmenfashionadvice (diverse builds and fit feedback)
add_q("men_bigmen_fit_check_new", "bigmenfashionadvice", query="fit check", sort="new", time_filter="all", desc="Men Outfit — Fresh big men fit checks")
add_q("men_bigmen_suit_fit", "bigmenfashionadvice", query="suit fit check", sort="top", time_filter="year", desc="Men Outfit — Big men suit fit checks")
add_q("men_bigmen_proportions", "bigmenfashionadvice", query="proportions advice fit", sort="relevance", time_filter="all", desc="Men Outfit — Big men proportions feedback")

# r/vintagefashion (classic tailoring for men & women)
add_q("men_vintage_suit_fit", "vintagefashion", query="mens suit vintage fit check", sort="top", time_filter="year", desc="Men Outfit — Classic tailored menswear")
add_q("women_vintage_dress_fit", "vintagefashion", query="vintage dress fit check", sort="top", time_filter="year", desc="Women Outfit — Vintage dress fit check")
add_q("women_o50_vintage_mature", "vintagefashion", query="vintage mature woman outfit", sort="relevance", time_filter="all", desc="Women Over 50 — Mature vintage styling")


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 3: Hair, Beard & Grooming for Underrepresented Demographics
# Subreddits: r/femalehairadvice, r/malehairadvice, r/malegrooming, r/MakeupAddiction
# ══════════════════════════════════════════════════════════════════════════════

# r/femalehairadvice (mature haircuts & styling improvements)
add_q("face_women_35_50_hair_40s", "femalehairadvice", query="haircut 40s", sort="new", time_filter="all", desc="Women 35-50 — Haircut advice in 40s")
add_q("face_women_o50_hair_50s", "femalehairadvice", query="haircut 50s", sort="top", time_filter="all", desc="Women Over 50 — Haircut advice in 50s")
add_q("face_women_o50_hair_60s", "femalehairadvice", query="haircut 60s", sort="top", time_filter="all", desc="Women Over 50 — Haircut advice in 60s")
add_q("face_women_mature_hair_advice", "femalehairadvice", query="mature hairstyle advice", sort="relevance", time_filter="all", desc="Women 35+ / Over 50 — Mature hair advice")
add_q("face_women_tier1_bad_haircut", "femalehairadvice", query="bad haircut how to fix", sort="relevance", time_filter="all", desc="Women Grooming — Haircut fixes / unstyled hair (Needs Improvement)")
add_q("face_women_tier1_messy_styling", "femalehairadvice", query="messy unstyled hair advice", sort="relevance", time_filter="all", desc="Women Grooming — Unstyled hair styling advice (Needs Improvement)")

# r/malehairadvice (mature hairstyles)
add_q("face_men_35_50_hair_40s", "malehairadvice", query="haircut in 40s", sort="relevance", time_filter="all", desc="Men 35-50 — Haircut advice in 40s")
add_q("face_men_o50_hair_50s", "malehairadvice", query="haircut in 50s", sort="relevance", time_filter="all", desc="Men Over 50 — Haircut advice in 50s")
add_q("face_men_o50_mature_hair", "malehairadvice", query="mature mens haircut", sort="top", time_filter="all", desc="Men Over 50 — Mature hairstyles")
add_q("face_men_tier1_haircut_fix", "malehairadvice", query="bad haircut need advice", sort="relevance", time_filter="all", desc="Men Grooming — Unstyled / flawed haircut (Needs Improvement)")

# r/malegrooming (mature beard, lineup & grooming checks)
add_q("face_men_35_50_grooming_beard", "malegrooming", query="beard trim advice 40s", sort="relevance", time_filter="all", desc="Men 35-50 — Beard trimming in 40s")
add_q("face_men_o50_grooming_silver", "malegrooming", query="older man beard trim silver", sort="top", time_filter="all", desc="Men Over 50 — Silver beard styling (Polished)")
add_q("face_men_tier1_neckbeard_cleanup", "malegrooming", query="neckbeard lineup cleanup", sort="relevance", time_filter="all", desc="Men Grooming — Untrimmed neckbeard cleanup (Needs Improvement)")
add_q("face_men_tier1_stubble_advice", "malegrooming", query="patchy stubble advice", sort="relevance", time_filter="all", desc="Men Grooming — Patchy stubble advice (Needs Improvement)")

# r/MakeupAddiction (mature styling & makeup execution)
add_q("face_women_35_50_makeup_40s", "MakeupAddiction", query="everyday makeup 40s", sort="relevance", time_filter="all", desc="Women 35-50 — Everyday makeup in 40s")
add_q("face_women_o50_makeup_50s", "MakeupAddiction", query="makeup routine 50s", sort="top", time_filter="all", desc="Women Over 50 — Makeup routine in 50s")
add_q("face_women_o50_hooded_eyes", "MakeupAddiction", query="mature hooded eyes makeup", sort="top", time_filter="year", desc="Women Over 50 — Mature eye makeup")
add_q("face_women_tier1_cakey_foundation", "MakeupAddiction", query="cakey foundation fix", sort="relevance", time_filter="all", desc="Women Grooming — Cakey makeup execution (Needs Improvement)")


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 4: Rotated Uncapped Queries on Core Subreddits (OUTFITS, femalefashionadvice)
# Replacing top/all (which was exhausted) with 'new', 'hot', and 'top/year'
# ══════════════════════════════════════════════════════════════════════════════

# r/femalefashionadvice (natural phrasing + rotated sort)
add_q("women_ffa_fit_check_new", "femalefashionadvice", query="fit check", sort="new", time_filter="all", desc="Women Outfit — Fresh FFA fit checks")
add_q("women_ffa_how_does_this_fit", "femalefashionadvice", query="how does this fit", sort="relevance", time_filter="all", desc="Women Outfit — Natural fit inquiries")
add_q("women_ffa_tailoring_critique", "femalefashionadvice", query="tailoring critique advice", sort="relevance", time_filter="all", desc="Women Outfit — Tailoring critique")
add_q("women_35_50_ffa_in_my_40s", "femalefashionadvice", query="in my 40s style", sort="top", time_filter="all", desc="Women 35-50 — FFA 40s style discussions")
add_q("women_o50_ffa_in_my_50s", "femalefashionadvice", query="in my 50s style", sort="top", time_filter="all", desc="Women Over 50 — FFA 50s style discussions")
add_q("women_o50_ffa_over_60", "femalefashionadvice", query="over 60 style", sort="top", time_filter="all", desc="Women Over 50 — FFA 60s style discussions")
add_q("feed_ffa_top_year", "femalefashionadvice", listing="top", time_filter="year", desc="Women Outfit — Top FFA past year")

# r/OUTFITS (mass volume, uncapped)
add_q("feed_outfits_new_rot", "OUTFITS", listing="new", desc="General Outfits — Fresh unfiltered feed (High Tier 1 & 2 content)")
add_q("feed_outfits_hot_rot", "OUTFITS", listing="hot", desc="General Outfits — Hot trending feed")
add_q("women_outfits_rate_my_fit", "OUTFITS", query="rate my outfit", sort="new", time_filter="all", desc="Women Outfit — Direct rating requests")
add_q("women_outfits_honest_opinion", "OUTFITS", query="honest opinion fit", sort="relevance", time_filter="all", desc="Women Outfit — Honest fit feedback")
add_q("women_35_50_outfits_over_40", "OUTFITS", query="over 40 outfit", sort="top", time_filter="all", desc="Women 35-50 — Over 40 outfit checks")
add_q("women_o50_outfits_over_50", "OUTFITS", query="over 50 outfit", sort="top", time_filter="all", desc="Women Over 50 — Over 50 outfit checks")

# r/femalefashion (visual feed)
add_q("feed_femalefashion_hot", "femalefashion", listing="hot", desc="Women Outfit — Hot feed femalefashion")
add_q("feed_femalefashion_new", "femalefashion", listing="new", desc="Women Outfit — New feed femalefashion")

# r/outfitoftheday
add_q("feed_outfitoftheday_new", "outfitoftheday", listing="new", desc="General Outfits — New feed outfitoftheday")


def main():
    out_path = Path(__file__).resolve().parent / "reddit_expansion_queries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    print(f"✅ Generated {len(queries)} expansion query configurations.")
    print(f"📁 Saved to: {out_path}")

    # Print distribution by target domain
    domain_counts = {}
    for q in queries:
        cat = q["category"]
        parts = cat.split("_")
        prefix = parts[0] + "_" + (parts[1] if len(parts) > 1 else "")
        domain_counts[prefix] = domain_counts.get(prefix, 0) + 1

    print("\nBreakdown by Category Prefix:")
    for k, v in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:30s}: {v:3d} queries")


if __name__ == "__main__":
    main()
