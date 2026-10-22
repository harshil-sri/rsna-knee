import argparse
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

LABEL_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

MEDIAL_RE = re.compile(
    r'\bmedial|\bmediale|\bmedialer|\bmedialen|\bmedialem|\bmedialis|\bmedyal|\bmedijaln|\bmedijalna|\bmedijalnog|\bmedijalnom'
    r'|\binner|\binnere|\binnen|\binnenmeniskus|\bmedius|\binterno|\binterna|\bmédial|\bmédiale|έσω|медиал|内侧',
    re.I
)

LATERAL_RE = re.compile(
    r'\blateral|\blaterale|\blateraler|\blateralen|\blateralem|\blateralis|\blateryal|\blateraln|\blateralna|\blateralnog|\blateralnom'
    r'|\bouter|\bäussere|\baussen|\baussenmeniskus|\baußenmeniskus|\bexterno|\bexterna|\blatéral|\blatérale|έξω|латерал|外侧',
    re.I
)

PF_RE = re.compile(
    r'patellofemoral|\bpf\b|femoro-patell|femoropatell|patellofémoral|patellofemorale|patellaire|rétropatellaire|retropatellar'
    r'|patellar|trochle|troclea|rótula|rotulian|patel|patellofemoralni|femoropatelarni|\bfp\b|patelofemoral|пател|трохле|трохлея|patellofemoralt'
    r'|επιγονατιδ|τροχιλ|πετραντιληπιακ',
    re.I
)

MENISCUS_KW = re.compile(
    r'menisc|menisk|ménis|μηνίσκ|мениск|半月板|innenmeniskus|aussenmeniskus|außenmeniskus',
    re.I
)

MENISCUS_PATHOLOGY = re.compile(
    r'tear|torn|ruptur|rupture|extrusion|extru|horizontal|vertical|complex|komplex|bucket[- ]?handle|kova sap|radial|radijaln|radiale'
    r'|ekstrud|скъсван|разрыв|scheur|desgarro|riss|flap|amputación'
    r'|grade\s*3|grad\s*3|stupnj?\s*3|степен\s*3'
    r'|grade\s*iii|grad\s*iii|stupnj?\s*iii|степен\s*iii'
    r'|grade\s*2b|grad\s*2b|stupnj?\s*2b|степен\s*2b'
    r'|lesion|lésion|lezij|увред',
    re.I
)

MENISCUS_NEG = re.compile(
    r'no\s+(?:evidence|sign|definite|obvious)?\s*(?:of\s+)?(?:medial\s+|lateral\s+)?(?:menisc\w+\s+)?(?:tear|rupture|lesion)'
    r'|no\s+(?:medial|lateral)?\s*menisc\w*\s+tear'
    r'|without\s+(?:evidence|sign)?\s*(?:of)?\s*(?:tear|rupture)'
    r'|not\s+torn|intact'
    r'|sin\s+(?:signos|criterios|evidencia)?\s*(?:de)?\s*rotura|sin\s+desgarro'
    r'|sans\s+(?:signe|présence)?\s*(?:de)?\s*(?:déchirure|rupture)|pas\s+de\s+déchirure'
    r'|kein\s*(?:e)?\s*(?:meniskus)?(?:riss|ruptur)|ohne\s+(?:hinweis|nachweis|anzeichen)|intakt'
    r'|geen\s+(?:aanwijzingen|tekenen|scheur)|zonder\s+scheur'
    r'|senza\s+(?:segni|evidenza)?\s*di\s*rottura|sem\s+(?:sinais|evidência)?\s*de\s*rotura'
    r'|yırtık\s+(?:saptanmadı|izlenmedi|yoktur|yok)|yoktur'
    r'|bez\s+(?:znakova|evidencije)?\s*(?:degeneracije\s+ili\s+)?rupture|nema\s+(?:znakova)?\s*rupture|uredan|uredna|uredno'
    r'|χωρίς\s+(?:ρήξη|σημεία\s+ρήξης)|δεν\s+παρατηρείται\s+ρήξη'
    r'|б\.о\.|без\s+(?:данни\s+за)?\s*(?:руптура|разрив)|няма\s+(?:данни\s+за)?\s*(?:руптура|разрив)|не\s+се\s+вижда\s+руптура'
    r'|normaldir|normal\b|normale\b|normalen\b'
    r'|без\s+особености|без\s+патологи|sin\s+alteraciones|without\s+surfacing\s+tear|no\s+tear',
    re.I
)

ACL_KW = re.compile(
    r'\bacl\b|\blca\b|\bvkb\b|voorste kruisband|vorderes kreuzband|ligamento cruzado anterior|ligament croisé antérieur'
    r'|legamento crociato anteriore|ön çapraz|предн\w+ кръст\w+ връзка|предн\w+ крестообразн\w+ связка|prednji križni|prednjeg križnog'
    r'|πρόσθι\w+ χιαστ\w+|πρόσθιου\s+χιαστού|前交叉韧带|前十字靭帯',
    re.I
)
ACL_TEAR = re.compile(
    r'tear|torn|ruptur|rupture|insufficien|sprain|lesion|lésion|lezij|laxit|discontinu|devamsızlık|ausriss'
    r'|разрыв|руптура|скъсван|povred|scheur|rotura|desgarro|ρήξη|断裂|撕裂|injury',
    re.I
)
ACL_NEG = re.compile(
    r'intact|normal|preserved|intakt|regelm|sin\s+alteraciones|bez\s+osobitosti|uredan|uredna|održan|интакт|няма|б\.о\.|geen|pas\s+de|sans|no\s+tear|not\s+torn|kein',
    re.I
)

MCL_KW = re.compile(
    r'\bmcl\b|\blcm\b|medial collateral|ligamento colateral medial|ligament collatéral médial|legamento collaterale mediale'
    r'|mediales kollateralband|medialen kollateralband|mediale band|medijaln\w+ kolateraln\w+|medial kollateral'
    r'|έσω πλάγι\w+|медиален колатерален|медиальная коллатеральная|内侧副韧带',
    re.I
)
MCL_TEAR = re.compile(
    r'tear|torn|ruptur|rupture|sprain|lesion|lésion|lezij|ausriss|разрыв|руптура|скъсван|povred|scheur|rotura|desgarro|ρήξη|断裂|撕裂|injury|thickening|zadebljan|ödematöz',
    re.I
)
MCL_NEG = re.compile(
    r'intact|normal|preserved|intakt|regelm|sin\s+alteraciones|bez\s+osobitosti|uredan|uredna|održan|интакт|няма|б\.о\.|geen|pas\s+de|sans|no\s+tear|not\s+torn|kein',
    re.I
)

TRICOMPARTMENTAL_RE = re.compile(
    r'tricompartment|tri-compartment|all\s+three\s+compartments|all\s+3\s+compartments|trigompartmental|pan-compartment',
    re.I
)

OA_KW = re.compile(
    r'osteoarthr|arthros|artros|gonarthros|gonartros|goanrtrot|chondropath|chondromalac|hondromalac|kraakbeenlijden|condropat|condromalacia'
    r'|cartilage\s+loss|chondral\s+(?:defect|lesion|injury|damage|thinning|fissur)|cartilage\s+(?:defect|thinning|loss|heterogeneity|fissur)|dégradation\s+cartilagineuse'
    r'|изтъняване\s+на\s+хрущял|остеоартрит|denudacija\s+hrskavic|hrskavic\w*\s+(?:reduciran|oštećen|defekt|erozij)|femorotibial|tibiofemoral'
    r'|οστεοαρθρίτιδ|οστεοαρθριτικ|αρθρικό\w*\s+χόνδρ\w*|εξάλειψη\s+του\s+αρθρικού\s+χόνδρου|διάβρωση\s+του\s+αρθρικού\s+χόνδρου|λέπτυνση\s+και\s+ανωμαλία'
    r'|artrotsk|kıkırdak\s+(?:kaybı|defekt|erozyon)|úlcera\w*\s+condral\w*|chondrosis|osteophyte|osteofit|остеофит|chondral\w*|kraakbeenverlies'
    r'|eklem\s+aralığı\s+(?:belirgin\s+)?daral|fissuring|reduciran\w*\s+zglobn\w*\s+prostor',
    re.I
)

OA_NEG = re.compile(
    r'no\s+(?:focal\s+)?chondrosis|no\s+chondral|cartilage.*(?:is\s+)?normal|cartilages\s+normal|sin\s+alteraciones|intact|intakt|uredan|б\.о\.|няма'
    r'|χωρίς\s+(?:αλλοιώσεις|βλάβη|οστεοαρθρίτιδα)|δεν\s+παρατηρούνται\s+(?:αξιόλογα\s+ευρήματα|αλλοιώσεις)'
    r'|preserved\s+(?:medial\s+and\s+lateral\s+)?(?:femorotibial\s+)?(?:articular\s+)?cartilage'
    r'|normaal\s+voorkomen\s+kraakbeen|cartilage.*appears\s+congruent\s+without\s+chondral\s+lesion'
    r'|cartilage\s+appears\s+intact',
    re.I
)

EFFUSION_KW = re.compile(
    r'effusion|épanchement|versamento|derrame|erguss|gelenkerguss|articular\s+fluid|joint\s+fluid|\bfluid\b|vocht|sıvı|mayii|izljev|излив'
    r'|синовиална\s+течност|синовиальная\s+жидкость|выпот|накопление\s+жидкости|关节积液|関節液|überschüssige\s+flüssigkeit'
    r'|συλλογή|υγρού|υγρό|υδράρθρωση|hydrops|efüzyon|artmış\s+efüzyon|izliv',
    re.I
)

EFFUSION_NEG = re.compile(
    r'no\s+(?:significant|trace|definite|gross|large|moderate|focal|joint|articular)?\s*(?:amount\s+of\s+)?(?:joint\s+|knee\s+)?effusion'
    r'|no\s+(?:significant|abnormal|excessive)?\s*(?:articular\s+|joint\s+)?fluid'
    r'|without\s+(?:joint\s+|knee\s+)?(?:effusion|fluid)'
    r'|pas\s+d\'épanchement|sin\s+(?:signos\s+de\s+)?derrame|kein\s*(?:e)?\s*(?:relevanter\s+|gelenk)?erguss|geen\s+(?:hydrops|vocht)'
    r'|sıvı\s+artışı\s+yok|sıvı\s+miktarı\s+normal|efüzyon\s+saptanmadı|efüzyon\s+izlenmedi|efüzyon\s+yok'
    r'|bez\s+(?:zglobnog\s+|slobodnog\s+)?izljeva|nema\s+izljeva|няма\s+излив|без\s+излив|б\.о\.'
    r'|χωρίς\s+(?:παρουσία\s+)?(?:συλλογής|υγρού)|δεν\s+(?:παρατηρείται|διαπιστώνεται)\s+(?:συλλογή|υγρό)'
    r'|senza\s+(?:significativo\s+)?versamento|sem\s+derrame',
    re.I
)

EFFUSION_MODERATE_OR_HIGH = re.compile(
    r'moderate|large|gross|marked|abundant|massive|prominent|tens\w+|opsežan|значим|σημαντικ|ικανή|yaygın|artmış\s+efüzyon|ausgeprägt',
    re.I
)

EFFUSION_MILD_OR_SMALL = re.compile(
    r'mild|small|minimal|trace|slight|some\s+amount|leve|discreto|discreta|gering|spuren|pequeño|pequeña|mínim|μικρ|ήπι|blag|manj|малк|минимал|hafif|kısm',
    re.I
)

SYNOVITIS_DIRECT = re.compile(
    r'synovit|sinovit|synovial\s+(?:thickening|hypertrophy|tissue|pannus|proliferation)|proliferat\w*\s+sinovij|hipertrofi\w*\s+sinovij|синовит|синовиална\s+хипертрофия|滑膜炎|synovialitis'
    r'|υμενίτιδ|υμενίτις|υμενικ\w*\s+πάχυνση|πάχυνση\s+του\s+υµένα|reizsynovialitis|verdikkingen\s+van\s+van\s+het\s+synovium',
    re.I
)

SYNOVITIS_INDIRECT = re.compile(
    r'bursitis|burzitis|ορογονοθυλακίτιδ|suprapatellar\s+bursa|suprapatellar\s+recess|suprapatellaire\s+recessus|suprapatelarno|suprapatellar\w*\s+(?:bursa|plica)',
    re.I
)

SYNOVITIS_NEG = re.compile(
    r'no\s+synovitis|sin\s+sinovitis|keine\s+synovitis|sans\s+synovite|bez\s+sinovitisa|няма\s+синовит|χωρίς\s+υμενίτιδα',
    re.I
)

BAKER_KW = re.compile(
    r'baker|popliteal\s+cyst|kyste\s+poplité|quiste\s+poplíteo|cista\s+poplite|popliteakyste|poplitealzyste|киста\s+на\s+бейкер|бекерова\s+киста|киста\s+бейкера'
    r'|bakerova\s+cista|bakerova\s+kista|κύστη\s+baker|συνοβιακή\s+κύστη|贝克囊肿',
    re.I
)
BAKER_NEG = re.compile(
    r'no\s+baker|no\s+popliteal\s+cyst|sin\s+quiste|keine\s+baker|geen\s+baker|geen\s+poplitea|sans\s+kyste|bez\s+poplitealne\s+ciste|bez\s+baker|nema\s+baker|няма\s+бекерова'
    r'|χωρίς\s+κύστη|δεν\s+παρατηρείται\s+κύστη',
    re.I
)

CONTUSION_KW = re.compile(
    r'contusion|bruise|bone\s+bruise|edema\s+óseo|koštani\s+edem|kemik\s+iliği\s+ödem|костно[- ]мозъчен\s+едем|костномозговой\s+отеκ|botoedeem|knochenödem'
    r'|marrow\s+edema|bone\s+marrow\s+edema|trabecular\s+microfracture|ост\w+ контузия|trabekul\w+ edem|οστεομυελικό\s+οίδημα|οστικό\s+μώλωπα|οστικό\s+οίδημα|挫伤|骨挫傷',
    re.I
)
CONTUSION_NEG = re.compile(
    r'no\s+(?:significant\s+)?bone\s+(?:marrow\s+)?(?:edema|contusion|bruise)|kein\s*(?:e)?\s*bone\s*bruise|kein\s*knochenödem|sin\s+(?:edema|alteraciones\s+en\s+médula)|bez\s+koštanog\s+edema'
    r'|kemik\s+iliği\s+ödemi\s+saptanmadı|няма\s+костно-мозъчен|б\.о\.|χωρίς\s+οστικό\s+οίδημα|χωρίς\s+οστεομυελικό\s+οίδημα',
    re.I
)

FRACTURE_KW = re.compile(
    r'fractur|fraktur|fracture|frattura|rotura\s+ósea|kırık|fraktura|фрактур|перелом|impaction|subchondral\s+collapse|avulsion|ausriss|subchondral\s+fracture|骨折',
    re.I
)
FRACTURE_NEG = re.compile(
    r'no\s+fracture|keine\s+fraktur|geen\s+fractuur|sin\s+fractura|sans\s+fracture|bez\s+frakture|kırık\s+saptanmadı|kırık\s+izlenmedi|kırık\s+yok|няма\s+фрактура|без\s+фрактура|б\.о\.',
    re.I
)

def split_sentences(text):
    if not isinstance(text, str):
        return []
    return [s.strip() for s in re.split(r'[\n\.\?!;•\t]+', text.lower()) if s.strip()]

def parse_report(report):
    labels = {col: 0.0 for col in LABEL_COLS}
    sents = split_sentences(report)

    med_men = 0.0
    lat_men = 0.0
    for s in sents:
        if MENISCUS_KW.search(s) and MENISCUS_PATHOLOGY.search(s):
            if MENISCUS_NEG.search(s):
                continue
            has_m = bool(MEDIAL_RE.search(s))
            has_l = bool(LATERAL_RE.search(s))
            if has_m and not has_l:
                med_men = 1.0
            elif has_l and not has_m:
                lat_men = 1.0
            elif has_m and has_l:
                med_men = 1.0
                lat_men = 1.0
            else:
                med_men = max(med_men, 0.5)
                lat_men = max(lat_men, 0.5)
    labels['Medial Meniscus'] = med_men
    labels['Lateral Meniscus'] = lat_men

    med_oa = 0.0
    lat_oa = 0.0
    pf_oa = 0.0
    if TRICOMPARTMENTAL_RE.search(report):
        med_oa = max(med_oa, 0.9)
        lat_oa = max(lat_oa, 0.9)
        pf_oa = max(pf_oa, 0.9)

    for s in sents:
        if OA_KW.search(s):
            if OA_NEG.search(s):
                continue
            has_m = bool(MEDIAL_RE.search(s))
            has_l = bool(LATERAL_RE.search(s))
            has_pf = bool(PF_RE.search(s))
            if has_pf:
                pf_oa = max(pf_oa, 1.0)
            if has_m and not has_l:
                med_oa = max(med_oa, 1.0)
            elif has_l and not has_m:
                lat_oa = max(lat_oa, 1.0)
            elif has_m and has_l:
                med_oa = max(med_oa, 1.0)
                lat_oa = max(lat_oa, 1.0)
            elif not has_pf:
                med_oa = max(med_oa, 0.5)
                lat_oa = max(lat_oa, 0.5)
    labels['Medial OA'] = med_oa
    labels['Lateral OA'] = lat_oa
    labels['PF OA'] = pf_oa

    for s in sents:
        if ACL_KW.search(s) and ACL_TEAR.search(s):
            if not ACL_NEG.search(s):
                labels['ACL'] = 1.0
                break

    for s in sents:
        if MCL_KW.search(s) and MCL_TEAR.search(s):
            if not MCL_NEG.search(s):
                labels['MCL'] = 1.0
                break

    eff_val = 0.0
    for s in sents:
        if EFFUSION_KW.search(s):
            if EFFUSION_NEG.search(s):
                continue
            if EFFUSION_MODERATE_OR_HIGH.search(s):
                eff_val = max(eff_val, 1.0)
            elif EFFUSION_MILD_OR_SMALL.search(s):
                eff_val = max(eff_val, 0.6)
            else:
                eff_val = max(eff_val, 0.8)
    labels['Effusion'] = eff_val

    syn_val = 0.0
    for s in sents:
        if SYNOVITIS_DIRECT.search(s):
            if not SYNOVITIS_NEG.search(s):
                syn_val = max(syn_val, 1.0)
        elif SYNOVITIS_INDIRECT.search(s):
            syn_val = max(syn_val, 0.6)
    labels['Synovitis'] = syn_val

    for s in sents:
        if BAKER_KW.search(s):
            if not BAKER_NEG.search(s):
                labels["Baker's"] = 1.0
                break

    for s in sents:
        if CONTUSION_KW.search(s):
            if not CONTUSION_NEG.search(s):
                labels['Contusion'] = 1.0
                break

    for s in sents:
        if FRACTURE_KW.search(s):
            if not FRACTURE_NEG.search(s):
                labels['Fracture'] = 1.0
                break

    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv', default='/mnt/ssd/rsna-knee/data/train.csv')
    parser.add_argument('--out', default='/mnt/ssd/rsna-knee/data/train_weak.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    parsed_rows = []
    for _, row in df.iterrows():
        parsed_labels = parse_report(row['Report'])
        is_gold = not all(pd.isna(row[label]) for label in LABEL_COLS)
        if is_gold:
            for label in LABEL_COLS:
                if not pd.isna(row[label]):
                    parsed_labels[label] = float(row[label])
        parsed_rows.append(parsed_labels)

    weak_df = pd.DataFrame(parsed_rows)
    result_df = pd.concat([df[['StudyInstanceUID', 'Report']], weak_df], axis=1)
    result_df.to_csv(args.out, index=False)

    gold_mask = ~df[LABEL_COLS].isna().all(axis=1)
    gold_rows = df[gold_mask]

    auc_scores = {}
    accuracy_scores = {}

    print(f"{'Label':<20} {'AUC':<10} {'Accuracy':<10}")
    for label in LABEL_COLS:
        y_true = gold_rows[label].values
        if len(np.unique(y_true)) < 2:
            continue
        fresh_preds = [parse_report(r)[label] for r in gold_rows['Report'].values]
        auc = roc_auc_score(y_true, fresh_preds)
        acc = accuracy_score(y_true, [1 if v > 0.5 else 0 for v in fresh_preds])
        auc_scores[label] = auc
        accuracy_scores[label] = acc
        print(f"{label:<20} {auc:<10.4f} {acc:<10.4f}")

    macro_auc = np.mean(list(auc_scores.values()))
    macro_accuracy = np.mean(list(accuracy_scores.values()))
    print(f"{'Macro':<20} {macro_auc:<10.4f} {macro_accuracy:<10.4f}")

    weak_mask = df[LABEL_COLS].isna().all(axis=1)
    weak_reports = df.loc[weak_mask, 'Report'].values
    weak_preds = [parse_report(r) for r in weak_reports]
    print("\nWeak Positive Rates:")
    for label in LABEL_COLS:
        cnt = sum(wp[label] > 0 for wp in weak_preds)
        rate = cnt / len(weak_reports)
        print(f"{label:<20} {rate*100:6.2f}% ({cnt}/{len(weak_reports)})")

if __name__ == '__main__':
    main()
