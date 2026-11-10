import argparse
import re
from pathlib import Path
 
import pandas as pd

PROJECT_PATH = Path(__file__).resolve().parents[1]


POS = set("""great good excellent amazing wonderful perfect lovely fantastic comfortable clean spotless cozy cosy
beautiful friendly helpful nice superb brilliant recommend recommended recommending enjoyed enjoy loved love
stunning spacious tidy convenient welcoming responsive kind pleasant fabulous awesome delightful immaculate
charming gem happy easy smooth seamless quick fast flawless outstanding gorgeous bright quiet peaceful safe
super genial parfait propre agréable magnifique excelente perfecto limpio genial encantador sehr schön sauber""".split())
NEG = set("""bad terrible awful horrible dirty filthy disappointing disappointed poor worst uncomfortable noisy
smelly smell mouldy mold mould broken cold damp rude unhelpful overpriced misleading cramped small old
unsafe sketchy dangerous scary problem problems issue issues unfortunately avoid rubbish disgusting
stain stains dusty leak leaking unresponsive cancelled canceled complaint annoying uncomfortable loud
bugs cockroach cockroaches mice infested thinwalls sale sucio horrible terrible schmutzig laut""".split())
NEGATORS = {"not", "no", "never", "without", "hardly", "barely", "nothing", "none", "neither", "nor", "pas", "sin", "nicht", "kein"}
 
# Từ vựng chuyên biệt theo khía cạnh nhạy cảm (nghĩa phụ thuộc phủ định)
NOISE_POS = {"quiet", "peaceful", "silent", "tranquil", "calm", "soundproof", "soundproofed"}
NOISE_NEG = {"noisy", "noise", "noises", "loud", "thinwalls", "disturbance", "disturbed", "sirens", "siren", "construction"}
SAFE_POS = {"safe", "secure", "securely", "safely"}
SAFE_NEG = {"unsafe", "sketchy", "dangerous", "crime", "robbed", "robbery", "burglary", "theft", "scary", "shady", "stolen"}