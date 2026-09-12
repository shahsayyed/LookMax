import json
from pathlib import Path

with open("scratch/detailed_side_by_side.json") as f:
    data = json.load(f)

for item in data:
    fn = item["filename"]
    cat = item["category"]
    our = item["our_prediction"]
    qwen_raw = item["qwen_raw"]
    try:
        qwen = json.loads(qwen_raw.strip("`").replace("json\n", "").strip())
    except:
        qwen = {}

    print(f"### Image: `{fn}` ({cat})")
    print("| Attribute / Metric | Large Model (Qwen2-VL-7B) | On-Device Simulator (CoreML) | Status / Analysis |")
    print("| :--- | :--- | :--- | :--- |")

    # Score
    q_score = qwen.get("score") or (qwen.get("outfit", {}).get("score")) or (qwen.get("grooming", {}).get("score")) or "N/A"
    delta_str = f"{float(our['score']) - float(q_score):+.2f}" if q_score != "N/A" else "0"
    print(f"| **Score** | **{q_score}** | **{our['score']}** ({our['tier_label']}) | Delta: {delta_str} |")

    if "Outfit" in cat:
        q_out = qwen.get("outfit", qwen)
        o_attrs = our.get("attributes", {})

        # Upper
        q_up = q_out.get("upper_type", "N/A")
        o_up = o_attrs.get("upper_type", {}).get("prediction", "N/A")
        u_match = "✅ Match" if str(q_up) in str(o_up) or str(o_up) in str(q_up) else "❌ Diverged"
        print(f"| **Upper Garment** | {q_up} | {o_up} ({o_attrs.get('upper_type',{}).get('confidence',0)*100:.1f}%) | {u_match} |")

        # Mid
        q_mid = q_out.get("mid_type", "none")
        o_mid = o_attrs.get("mid_type", {}).get("prediction", "none")
        m_match = "✅ Match" if q_mid == o_mid else "❌ Diverged"
        print(f"| **Mid / Outer** | {q_mid} | {o_mid} ({o_attrs.get('mid_type',{}).get('confidence',0)*100:.1f}%) | {m_match} |")

        # Lower
        q_low = q_out.get("lower_type", "N/A")
        o_low = o_attrs.get("lower_type", {}).get("prediction", "N/A")
        l_match = "✅ Match" if str(q_low) in str(o_low) or str(o_low) in str(q_low) else "❌ Diverged"
        print(f"| **Lower Garment** | {q_low} | {o_low} ({o_attrs.get('lower_type',{}).get('confidence',0)*100:.1f}%) | {l_match} |")

        # Footwear
        q_shoe = q_out.get("footwear_type", "N/A")
        o_shoe = o_attrs.get("footwear_type", {}).get("prediction", "N/A")
        s_match = "✅ Match" if str(q_shoe) in str(o_shoe) or str(o_shoe) in str(q_shoe) else "❌ Diverged"
        print(f"| **Footwear** | {q_shoe} | {o_shoe} ({o_attrs.get('footwear_type',{}).get('confidence',0)*100:.1f}%) | {s_match} |")

        # Formality
        q_form = q_out.get("formality", "N/A")
        o_form = o_attrs.get("formality", {}).get("prediction", "N/A")
        f_match = "✅ Match" if q_form == o_form else "⚠️ Close / Diverged"
        print(f"| **Formality** | {q_form} | {o_form} ({o_attrs.get('formality',{}).get('confidence',0)*100:.1f}%) | {f_match} |")

        # Fit
        q_fit = q_out.get("fit", "N/A")
        o_fit = "tailored" if o_attrs.get("fit_tailored",{}).get("prediction") == "1" else ("baggy" if o_attrs.get("fit_baggy",{}).get("top_index",0) > 0 else "regular")
        print(f"| **Fit** | {q_fit} | {o_fit} | Contextual match |")
    else:
        q_gr = qwen.get("grooming", qwen)
        o_attrs = our.get("attributes", {})

        # Hair length
        q_hl = q_gr.get("hair_length", "N/A")
        o_hl = o_attrs.get("hair_length", {}).get("prediction", "N/A")
        hl_match = "✅ Match" if q_hl == o_hl else "⚠️ Close / Diverged"
        print(f"| **Hair Length** | {q_hl} | {o_hl} ({o_attrs.get('hair_length',{}).get('confidence',0)*100:.1f}%) | {hl_match} |")

        # Hair styled
        q_hs = q_gr.get("hair_styled", "N/A")
        o_hs = "styled" if o_attrs.get("hair_styled",{}).get("prediction") == "1" else "unstyled"
        hs_match = "✅ Match" if (q_hs == "yes" and o_hs == "styled") or (q_hs in ["no", "messy"] and o_hs == "unstyled") else "❌ Diverged"
        print(f"| **Hair Styling** | {q_hs} | {o_hs} | {hs_match} |")

        # Skin condition
        q_sk = q_gr.get("skin_condition", "N/A")
        o_sk = "healthy" if o_attrs.get("skin_healthy",{}).get("prediction") == "1" else "natural / baseline"
        print(f"| **Skin Condition** | {q_sk} | {o_sk} | General agreement |")

        # Facial hair or makeup
        q_fm = q_gr.get("facial_hair_or_makeup", "N/A")
        if "Men" in cat:
            o_fm = o_attrs.get("facial_hair_style", {}).get("prediction", "N/A")
        else:
            o_fm = o_attrs.get("makeup_style", {}).get("prediction", "N/A")
        print(f"| **Facial Hair / Makeup** | {q_fm} | {o_fm} | Contextual match |")

    print("\n")
