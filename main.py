import pandas as pd
import random

# 1. Your Dictionary Data (The 1-2 word entries)
# "apple" -> "تفاحة"
# "run" -> "جري"
# ...

# 2. THE MEGA-LIST of Templates (High Variance)
# We mix: Questions, Commands, Statements, and casual chatting.
templates = [
    # Direct Questions
    ("How do you say '{}' in Tunisian?", "كيفاش تقول '{}' بالتونسي؟"),
    ("What is the word for '{}'?", "شنوة كلمة '{}'؟"),
    ("What does '{}' mean?", "شنوة معناها '{}'؟"),
    
    # Casual / Chat
    ("I think the word is '{}'.", "نظن الكلمة هي '{}'."),
    ("They call it '{}' here.", "يسميوها لهنا '{}'."),
    ("I need a '{}'.", "حشتي ب '{}'."),
    ("Do you have '{}'?", "عندك '{}'؟"),
    
    # Commands
    ("Translate '{}' please.", "ترجم '{}' بربي."),
    ("Write '{}' in Derja.", "اكتب '{}' بالدارجة."),
    
    # Contextual (Simulated)
    ("He said '{}' to me.", "قالي '{}'."),
    ("I saw a '{}' yesterday.", "ريت '{}' البارح."),
    ("Can you give me the '{}'?", "تنجم تعطيني '{}'؟"),
    ("Where is the '{}'?", "وين '{}'؟"),
    
    # Short / Fragments (Critical for robustness)
    ("'{}'", "'{}'"), 
    ("The '{}'", "'{}' ال"), # "The apple" -> "التفاحة" (Rough approximation)
    ("A '{}'", "'{}'"),
]

def smart_inflate(df):
    processed_data = []
    
    for _, row in df.iterrows():
        en_text = row['input']
        tn_text = row['output']
        
        # A. If it's already a sentence (3+ words), KEEP IT RAW.
        # Don't touch real data.
        if len(en_text.split()) > 2:
            processed_data.append({"input": en_text, "output": tn_text})
            # Add it again to weight real sentences higher than synthetic ones
            processed_data.append({"input": en_text, "output": tn_text}) 
            continue
            
        # B. If it's a Dictionary Word, create 3 UNIQUE variations
        # We sample 3 random templates so every word looks different.
        selected_templates = random.sample(templates, 3)
        
        for temp_en, temp_tn in selected_templates:
            # Handle the template filling
            processed_data.append({
                "input": temp_en.format(en_text),
                "output": temp_tn.format(tn_text)
            })
            
    return pd.DataFrame(processed_data)

# Usage
# df = pd.DataFrame(your_raw_data)
data = pd.read_json("drejja_perfect_instruct.jsonl", lines=True)
df = pd.DataFrame(data)
final_df = smart_inflate(df)
final_df = final_df.drop_duplicates()
final_df.to_json("drejja_perfect_instruct_inflated.jsonl", orient="records", lines=True, force_ascii=False)
print(f"Diversity Score: {final_df['input'].nunique() / len(final_df)}")