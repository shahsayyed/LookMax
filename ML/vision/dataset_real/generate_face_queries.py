import json

face_queries = []

def add_q(cat, sub, q, desc):
    face_queries.append({
        "category": f"face_{cat}",
        "subreddit": sub,
        "query": q,
        "sort": "top",
        "time": "all",
        "description": desc
    })

# ==========================================
# MEN UNDER 35
# ==========================================
# Needs Improvement
add_q("men_u35_hair_messy", "malehairadvice", "messy hair fix", "Men Under 35 - Unstyled/messy hair (Needs Improvement)")
add_q("men_u35_hair_thinning", "malehairadvice", "thinning hair advice", "Men Under 35 - Thinning hair/receding hairline (Needs Improvement)")
add_q("men_u35_beard_unkempt", "malegrooming", "unkempt beard help", "Men Under 35 - Unkempt patchy beard (Needs Improvement)")
add_q("men_u35_skin_acne", "SkincareAddiction", "acne help male", "Men Under 35 - Acne prone skin (Needs Improvement)")
add_q("men_u35_ugly_advice", "amiugly", "M18 am i ugly", "Men Under 35 - Candid face advice (Needs Improvement/Average)")
# Average
add_q("men_u35_hair_average", "malehairadvice", "everyday haircut", "Men Under 35 - Basic everyday haircut (Average)")
add_q("men_u35_beard_trim", "malegrooming", "beard trim advice", "Men Under 35 - Standard beard trim (Average)")
add_q("men_u35_skin_basic", "SkincareAddiction", "basic routine male", "Men Under 35 - Normal skin (Average)")
# Polished
add_q("men_u35_hair_fade", "malehairadvice", "fresh fade", "Men Under 35 - Sharp styled haircut (Polished)")
add_q("men_u35_beard_perfect", "malegrooming", "perfect beard shape", "Men Under 35 - Well groomed beard (Polished)")
add_q("men_u35_glowup", "GlowUps", "M20 to M25 glow up", "Men Under 35 - Facial glow up (Polished)")

# ==========================================
# MEN 35-50
# ==========================================
# Needs Improvement
add_q("men_35_50_hair_balding", "malehairadvice", "balding 40s advice", "Men 35-50 - Balding hair (Needs Improvement)")
add_q("men_35_50_beard_patchy", "malegrooming", "patchy beard 40s", "Men 35-50 - Poor beard growth (Needs Improvement)")
add_q("men_35_50_skin_tired", "SkincareAddiction", "tired eyes dark circles men", "Men 35-50 - Tired looking skin/eyes (Needs Improvement)")
# Average
add_q("men_35_50_hair_normal", "malehairadvice", "haircut 40s", "Men 35-50 - Standard haircut (Average)")
add_q("men_35_50_beard_average", "beards", "short beard 40s", "Men 35-50 - Average beard (Average)")
add_q("men_35_50_skin_aging", "30PlusSkinCare", "anti aging men", "Men 35-50 - Normal aging skin (Average)")
# Polished
add_q("men_35_50_hair_styled", "malehairadvice", "stylish haircut 40s", "Men 35-50 - Styled sophisticated hair (Polished)")
add_q("men_35_50_beard_majestic", "beards", "epic beard trim", "Men 35-50 - Epic perfectly shaped beard (Polished)")
add_q("men_35_50_glowup", "GlowUps", "M40 glow up", "Men 35-50 - Middle aged glow up (Polished)")

# ==========================================
# MEN OVER 50
# ==========================================
# Needs Improvement
add_q("men_o50_hair_messy", "malehairadvice", "messy hair older man", "Men Over 50 - Unkempt hair (Needs Improvement)")
add_q("men_o50_beard_wild", "beards", "wild grey beard", "Men Over 50 - Wild unshaped beard (Needs Improvement)")
add_q("men_o50_skin_neglected", "30PlusSkinCare", "sun damage men 60s", "Men Over 50 - Sun damaged skin (Needs Improvement)")
# Average
add_q("men_o50_hair_short", "malehairadvice", "short haircut older guy", "Men Over 50 - Simple short hair (Average)")
add_q("men_o50_beard_trim", "malegrooming", "trimmed grey beard", "Men Over 50 - Trimmed beard (Average)")
add_q("men_o50_skin_normal", "30PlusSkinCare", "skincare men 50+", "Men Over 50 - Normal skin (Average)")
# Polished
add_q("men_o50_hair_distinguished", "malehairadvice", "distinguished grey hair", "Men Over 50 - Stylish grey hair (Polished)")
add_q("men_o50_beard_sharp", "beards", "sharp grey beard", "Men Over 50 - Perfectly shaped grey beard (Polished)")
add_q("men_o50_glowup", "GlowUps", "M50 glow up", "Men Over 50 - Senior glow up (Polished)")


# ==========================================
# WOMEN UNDER 35
# ==========================================
# Needs Improvement
add_q("women_u35_hair_frizzy", "femalehairadvice", "frizzy messy hair help", "Women Under 35 - Messy/frizzy hair (Needs Improvement)")
add_q("women_u35_brows_messy", "Eyebrows", "messy eyebrows fix", "Women Under 35 - Unshaped eyebrows (Needs Improvement)")
add_q("women_u35_skin_acne", "SkincareAddiction", "cystic acne help female", "Women Under 35 - Acne prone skin (Needs Improvement)")
add_q("women_u35_ugly_advice", "amiugly", "F19 am i ugly", "Women Under 35 - Candid face advice (Needs Improvement/Average)")
# Average
add_q("women_u35_hair_everyday", "femalehairadvice", "everyday hairstyle", "Women Under 35 - Normal everyday hair (Average)")
add_q("women_u35_makeup_basic", "MakeupAddiction", "no makeup makeup look", "Women Under 35 - Basic makeup (Average)")
add_q("women_u35_skin_clear", "SkincareAddiction", "basic routine clear skin", "Women Under 35 - Clear healthy skin (Average)")
# Polished
add_q("women_u35_hair_salon", "femalehairadvice", "fresh salon blowout", "Women Under 35 - Perfect salon hair (Polished)")
add_q("women_u35_brows_perfect", "Eyebrows", "perfectly shaped brows", "Women Under 35 - Sculpted eyebrows (Polished)")
add_q("women_u35_makeup_glam", "MakeupAddiction", "glam makeup look", "Women Under 35 - Glamorous makeup (Polished)")
add_q("women_u35_glowup", "GlowUps", "F20 to F25 glow up", "Women Under 35 - Facial glow up (Polished)")

# ==========================================
# WOMEN 35-50
# ==========================================
# Needs Improvement
add_q("women_35_50_hair_damaged", "femalehairadvice", "damaged hair 40s", "Women 35-50 - Damaged hair (Needs Improvement)")
add_q("women_35_50_skin_tired", "30PlusSkinCare", "tired looking skin 40s", "Women 35-50 - Tired skin (Needs Improvement)")
add_q("women_35_50_makeup_dated", "MakeupAddiction", "outdated makeup advice", "Women 35-50 - Dated makeup style (Needs Improvement)")
# Average
add_q("women_35_50_hair_normal", "femalehairadvice", "low maintenance haircut 40s", "Women 35-50 - Low maintenance hair (Average)")
add_q("women_35_50_skin_routine", "30PlusSkinCare", "skincare routine 40s", "Women 35-50 - Normal skin (Average)")
add_q("women_35_50_makeup_work", "MakeupAddiction", "work makeup look 40s", "Women 35-50 - Professional makeup (Average)")
# Polished
add_q("women_35_50_hair_chic", "femalehairadvice", "chic haircut 40s", "Women 35-50 - Stylish chic hair (Polished)")
add_q("women_35_50_skin_glowing", "30PlusSkinCare", "glowing skin success 40s", "Women 35-50 - Glowing skin (Polished)")
add_q("women_35_50_glowup", "GlowUps", "F40 glow up", "Women 35-50 - Glow up success (Polished)")

# ==========================================
# WOMEN OVER 50
# ==========================================
# Needs Improvement
add_q("women_o50_hair_thinning", "femalehairadvice", "thinning hair 60s", "Women Over 50 - Thinning hair (Needs Improvement)")
add_q("women_o50_skin_dry", "30PlusSkinCare", "dry mature skin help", "Women Over 50 - Dry mature skin (Needs Improvement)")
add_q("women_o50_brows_sparse", "Eyebrows", "sparse eyebrows older", "Women Over 50 - Sparse eyebrows (Needs Improvement)")
# Average
add_q("women_o50_hair_grey", "femalehairadvice", "natural grey hair transition", "Women Over 50 - Normal grey hair (Average)")
add_q("women_o50_skin_normal", "30PlusSkinCare", "routine for 60s", "Women Over 50 - Standard mature skincare (Average)")
add_q("women_o50_makeup_light", "MakeupAddiction", "light makeup mature skin", "Women Over 50 - Light makeup (Average)")
# Polished
add_q("women_o50_hair_elegant", "femalehairadvice", "elegant mature haircut", "Women Over 50 - Beautifully styled mature hair (Polished)")
add_q("women_o50_skin_radiant", "30PlusSkinCare", "radiant skin 60s", "Women Over 50 - Radiant healthy mature skin (Polished)")
add_q("women_o50_glowup", "GlowUps", "F50 glow up", "Women Over 50 - Beautiful senior glow up (Polished)")

out_file = Path(__file__).resolve().parent / 'reddit_face_queries.json'
with open(out_file, 'w') as f:
    json.dump(face_queries, f, indent=2)

print(f"Generated {len(face_queries)} comprehensive face grooming categories.")
