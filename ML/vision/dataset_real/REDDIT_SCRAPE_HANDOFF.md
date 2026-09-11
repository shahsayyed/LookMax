# LookMax — Reddit Image Scrape Handoff (Context Transfer for another AI)

## What this project is
LookMax trains vision models to score a person's outfit / grooming / posture into
3 tiers (`1_Needs_Improvement`, `2_Average`, `3_Polished`) across 6 demographic
buckets (Men/Women × Under 35 / 35-50 / Over 50). Real (non-synthetic) training
photos come from Reddit, scraped via `ML/vision/dataset_real/reddit_scraper.py`
— an authenticated Playwright session hitting Reddit's `.json` listing/search
endpoints (not the paid official API). That script and its query catalog
(`reddit_outfit_queries.json`, `reddit_face_queries.json`, `reddit_bad_faces.json`)
are the ground truth for "what we already tried."

## The problem
Some demographic/tier combinations — mostly **women 35+/50+** and a few
cross-demographic categories — came back with almost no images. This is a
**content-scarcity problem, not a scraper bug**: r/femalefashionadvice and
r/Outfits are the only genuinely active subreddits for this kind of content and
their posters skew young, so searching harder with the same terms on the same
subs won't produce more. See counts below.

## Categories already tried and confirmed thin (DO NOT re-run these exact
subreddit+query combos expecting new results — they're likely near-exhausted)

Format: `category | results found | subreddit | search query | sort | time`

```
men_35_50_blazer_trousers_combo | 0 | businesscasual | blazer trousers fit check | top | all
men_35_50_styleboard_grid | 0 | styleboards | men smart casual outfit grid | top | all
women_35_50_slouching_tired_posture | 0 | femalefashionadvice | slouching posture fit check | relevance | all
women_35_50_business_casual_blazer | 0 | businesscasual | women business casual blazer pants | top | all
face_men_u35_glowup | 0 | GlowUps | M20 to M25 glow up | top | all
face_men_35_50_glowup | 0 | GlowUps | M40 glow up | top | all
face_men_o50_glowup | 0 | GlowUps | M50 glow up | top | all
face_women_35_50_glowup | 0 | GlowUps | F40 glow up | top | all
face_women_o50_glowup | 0 | GlowUps | F50 glow up | top | all
face_tier1_bad_men_patchy_neckbeard | 0 | malegrooming | neckbeard patchy | top | all
face_tier1_bad_women_cakey_bad_makeup | 0 | MakeupAddiction | cakey makeup help | top | all
women_35_50_tailored_monochrome | 1 | femalefashion | tailored monochrome outfit | top | all
women_o50_mature_slouch_posture | 1 | femalefashionadvice | mature woman posture outfit | relevance | all
face_women_u35_glowup | 1 | GlowUps | F20 to F25 glow up | top | all
posture_full_body_standing_check | 2 | Posture | standing posture check full body | top | all
posture_desk_slouch_seated_standing | 2 | Posture | desk posture standing check | top | all
face_tier1_bad_both_ugly_features | 2 | truerateme | below average 3 | top | all
face_men_35_50_hair_normal | 3 | malehairadvice | haircut 40s | top | all
face_women_o50_skin_normal | 3 | 30PlusSkinCare | routine for 60s | top | all
face_tier1_bad_women_overplucked_brows | 3 | Eyebrows | overplucked ruined | top | all
face_tier1_bad_both_lookyourbest_desperate | 4 | lookyourbest | desperate ugly | top | all
women_u35_minimalist_aesthetic | 5 | femalefashion | minimalist aesthetic outfit | top | all
women_u35_clean_girl_aesthetic | 5 | femalefashion | clean girl aesthetic street | top | all
women_35_50_mature_effortless_chic | 5 | femalefashion | mature women effortless chic | top | all
face_women_35_50_hair_chic | 5 | femalehairadvice | chic haircut 40s | top | all
women_o50_cashmere_blazer_mature | 6 | femalefashionadvice | mature woman cashmere blazer | top | all
posture_good_upright_alignment | 6 | Posture | good posture upright standing | top | all
women_o50_sophisticated_tailored | 7 | femalefashionadvice | sophisticated older woman tailored | top | all
men_o50_classic_menswear_over50 | 8 | mensfashion | classic menswear over 50 | top | all
women_o50_senior_casual_clothes | 10 | femalefashionadvice | senior woman casual clothes | top | all
women_u35_cocktail_elegant_dress | 11 | femalefashionadvice | cocktail dress elegant fit | top | all
women_35_50_unflattering_office | 11 | femalefashionadvice | unflattering office clothes advice | relevance | all
men_u35_techwear_aesthetic | 14 | streetwear | techwear outfit | top | all
men_o50_mature_posture_standing | 14 | mensfashion | mature man standing fit | relevance | all
women_u35_date_night_chic | 15 | femalefashionadvice | date night chic outfit | top | all
women_o50_elderly_dress_fit | 15 | femalefashionadvice | elderly woman dress fit critique | relevance | all
women_o50_chic_over_60 | 18 | femalefashionadvice | chic over 60 outfit fit | top | all
posture_correction_before_after | 18 | Posture | posture correction before after | top | all
posture_confident_standing_portrait | 20 | Posture | confident standing posture portrait | top | all
men_35_50_business_casual_work | 21 | businesscasual | men business casual office | top | all
women_35_50_sophisticated_dinner | 21 | femalefashionadvice | sophisticated dinner dress woman | top | all
women_35_50_smart_linen_blazer | 22 | femalefashionadvice | smart casual linen blazer 40s | top | all
posture_rounded_shoulders_slouch | 22 | Posture | rounded shoulders slouching standing | top | all
women_u35_evening_gala_gown | 24 | femalefashionadvice | evening gala gown portrait | top | all
women_35_50_corporate_executive | 24 | femalefashionadvice | corporate executive business formal woman | top | all
men_u35_summer_casual_fit | 26 | malefashionadvice | summer casual outfit | top | all
men_u35_bespoke_tailored_suit | 27 | malefashionadvice | tailored suit bespoke fit | top | all
men_u35_skater_street_casual | 28 | streetwear | skater casual street | top | all
women_u35_casual_summer_sundress | 28 | femalefashionadvice | casual summer sundress outfit | top | all
```

## Categories that hit the scraper's `--limit 100` cap (DIFFERENT situation —
these stopped because we told them to, not because Reddit ran out; re-running
with a higher `--limit` and a different `sort`/`time` on these SAME subreddits
is likely to surface genuinely new images)

```
body_bigmen_fit_check | bigmenfashionadvice | outfit fit check
body_malegrooming_style_check | malegrooming | full body style advice
face_tier1_bad_both_acne_scars | acne | acne scars texture
face_tier1_bad_both_amiugly_brutal | amiuglybrutallyhonest | (feed)
face_tier1_bad_both_asymmetrical_face | amiugly | asymmetrical ugly
face_tier1_bad_both_bad_breakout | SkincareAddiction | worst breakout
face_tier1_bad_both_severe_cystic_acne | acne | severe cystic acne
face_tier1_bad_men_greasy_messy_hair | malehairadvice | greasy messy hair
face_tier1_bad_men_older_balding_bad | malehairadvice | older balding mess
face_tier1_bad_men_receding_balding | malehairadvice | receding hairline balding
face_tier1_bad_men_shave_it_off | malegrooming | should i shave it off
face_tier1_bad_women_awful_eyebrows | awfuleyebrows | (feed)
face_tier1_bad_women_botched_haircut | femalehairadvice | crying botched haircut
face_tier1_bad_women_fried_damaged_hair | femalehairadvice | fried damaged hair
face_tier1_bad_women_frizzy_mess | femalehairadvice | frizzy mess
face_tier1_bad_women_older_damaged_hair | femalehairadvice | older thinning damaged
feed_femalefashion_top_all | femalefashion | (feed: top/all)
feed_femalefashionadvice_top_all | femalefashionadvice | (feed: top/all)
feed_malefashionadvice_top_all | malefashionadvice | (feed: top/all)
feed_mensfashion_top_all | mensfashion | (feed: top/all)
feed_outfitoftheday_hot | outfitoftheday | (feed: hot)
feed_outfitoftheday_top_all | outfitoftheday | (feed: top/all)
feed_outfits_hot | OUTFITS | (feed: hot)
feed_outfits_top_all | OUTFITS | (feed: top/all)
feed_streetwear_top_all | streetwear | (feed: top/all)
feed_throwingfits_hot | ThrowingFits | (feed: hot)
men_35_50_40s_casual_advice | mensfashion | 40s casual clothes advice
men_35_50_classic_navy_suit | mensfashion | classic navy suit tailoring
men_35_50_dad_outfit_critique | mensfashion | dad outfit advice fit check
men_35_50_executive_business_formal | mensfashion | executive formal business suit
men_35_50_fitted_blazer_jeans | mensfashion | blazer with jeans fit
men_35_50_office_business_casual | mensfashion | business casual office fit
men_35_50_sartorial_double_breasted | mensfashion | double breasted suit sartorial
men_35_50_selvedge_denim_boots | rawdenim | fit check denim boots
men_35_50_slouch_posture_fit | mensfashion | posture slouching fit check
men_35_50_smart_casual_dinner | mensfashion | smart casual dinner blazer
men_35_50_weekend_polo_chinos | mensfashion | polo chinos weekend fit
men_o50_elderly_suit_fitting | mensfashion | elderly man suit fit
men_o50_linen_summer_suit | mensfashion | older man elegant linen suit
men_o50_mature_sartorial_overcoat | mensfashion | mature sartorial coat tailored
men_o50_older_man_fit_critique | mensfashion | older man fit check
men_o50_senior_casual_outfit | mensfashion | senior casual everyday outfit
men_o50_silver_fox_sharp_suit | mensfashion | silver fox sharp suit
men_o50_tweed_jacket_gentleman | mensfashion | tweed jacket mature gentleman
men_o50_vintage_classic_menswear | vintagefashion | vintage menswear suit jacket
men_u35_outfits_daily_look | OUTFITS | guy casual daily outfit
men_u35_outfits_party_event | OUTFITS | guy party nightlife outfit
men_u35_runway_high_fashion | streetwear | high fashion street runway
men_u35_slouch_and_fit_check | malefashionadvice | fit check slouch
men_u35_streetwear_oversized_hoodie | streetwear | hoodie fit check
men_u35_vintage_japanese_streetwear | streetwear | japanese streetwear fit
men_u35_wedding_guest_suit | malefashionadvice | wedding guest suit
posture_progress_standing_full | progresspics | posture standing full body
women_35_50_creative_mature_outfit | oldhagfashion | colorful creative mature outfit
women_o50_vintage_elegant_dress | vintagefashion | vintage elegant dress mature
women_u35_awkward_dress_critique | femalefashionadvice | awkward dress fit critique
women_u35_outfits_casual_daily | OUTFITS | casual cute daily outfit
women_u35_outfits_evening_dress | OUTFITS | evening dress fit check
women_u35_petite_casual_outfit | petitefashionadvice | petite casual everyday outfit
women_u35_petite_tailored_suit | petitefashionadvice | petite tailored professional suit
women_u35_plussize_casual_cute | PlusSizeFashion | plus size casual cute outfit
women_u35_plussize_formal_dress | PlusSizeFashion | plus size elegant formal dress
women_u35_trendy_brunch_look | OUTFITS | trendy brunch outfit
```

## Where the full "already scraped" URL list lives (use this to dedupe)

- **`ML/data/vision_real/2_VLM_Processing/metadata_logs/reddit_images.json`** —
  the canonical record. A JSON dict of `{category_name: [image_url, ...]}`,
  194 categories, **11,563 total URLs already scraped**. This is exactly the
  file `reddit_scraper.py` loads on startup to skip anything already collected
  (`scrape_categories()` in `reddit_scraper.py`, loads this path via
  `--output`, unions every category's URL list into one `global_seen_urls`
  set before scraping anything new). **Any new scrape should load this same
  file and check candidate URLs against the full union of its values, not
  just its own category's list** — the same photo can appear in more than one
  category/query and we don't want it twice anywhere in the dataset.
- `ML/data/vision_real/2_VLM_Processing/metadata_logs/scraped_urls.txt` — a
  flat URL log shared by the non-Reddit loaders (Unsplash/CelebA/FairFace),
  639 lines. Less relevant for a Reddit-only run but check it too if a
  candidate image's source URL happens to match.
- **Content-level dedup**: `reddit_scraper.py` also hashes every downloaded
  image's raw bytes (SHA256) and skips re-saving identical files — this
  catches the same photo appearing twice even under a different URL. If the
  new scrape writes into a fresh location instead of the existing
  `RAW_SCRAPES_DIR`, replicate this: hash-check against every file already
  under `ML/data/vision_real/1_Raw_Scrapes/` and
  `ML/data/vision_real/3_CoreML_Training_Data/` before saving anything new.
  A SHA256 exact-match check is the minimum bar; note it will **not** catch
  the same photo re-encoded/resized/cropped elsewhere — only true byte-identical
  duplicates.

## What "success" looks like for a new scrape run
1. Load `reddit_images.json`, build the global seen-URL set from **all**
   category values (not just the target category).
2. For the capped-list categories: re-query the **same subreddit**, same or
   broadened search term, but a higher result cap and a different `sort`
   (`new`, `relevance`, `controversial`) / `time` (`year`, `month`) than what
   was used before — top/all was already exhausted at the old limit for these.
3. For the scarce-list categories: don't retry the identical subreddit+query —
   it's very likely genuinely near-empty. Instead widen to adjacent
   subreddits (e.g. `frugalfemalefashion`, unisex `fashionadvice`) or drop the
   quality-judgment language from the query and pull a broader unfiltered feed,
   since the existing VLM classifier step handles quality tiering separately
   from scraping.
4. Any newly saved image file should be content-hashed and checked against
   the existing corpus before being written to disk, and its source URL
   appended back into `reddit_images.json` under the right category so future
   runs (by any tool) don't repeat this work.
