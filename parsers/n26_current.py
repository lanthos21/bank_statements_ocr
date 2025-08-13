# n26_current.py — robust N26 extractor (header-first; cluster fallback close-to-amount)
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

# ---------- regexes ----------
DATE_RE   = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b")
AMOUNT_RE = re.compile(r"(?P<sign>[+\-−])?\s*(?P<body>\d{1,3}(?:[.\s\u00A0\u2009\u202F]\d{3})*[.,]\d{2})\s*€")
FURNITURE = re.compile(r"^(?:value\s*date\b|mastercard\b)$", re.I)

# ---------- utils ----------
def _med_height(df: pd.DataFrame) -> float:
    return float(df["height"].median()) if ("height" in df and not df["height"].empty) else 18.0

def _norm(s: str) -> str:
    if not s: return ""
    return (s.replace("\u00A0"," ")
             .replace("\u2009"," ")
             .replace("\u202F"," ")
             .replace("−","-")
             .strip())

def _page_words_to_df(page: dict) -> pd.DataFrame:
    rows = []
    for ln in page.get("lines", []) or []:
        for w in ln.get("words", []) or []:
            rows.append(w)
    if not rows:
        return pd.DataFrame(columns=[
            "text","left","top","width","height","right","bottom","cy",
            "block_num","par_num","line_num","word_num"
        ])
    df = pd.DataFrame(rows).copy()
    for c in ("left","top","width","height"):
        if c in df.columns: df[c] = df[c].astype(int)
    for c in ("block_num","par_num","line_num","word_num"):
        if c not in df.columns: df[c] = -1
    df["right"]  = df["left"] + df["width"]
    df["bottom"] = df["top"]  + df["height"]
    df["cy"]     = df["top"] + df["height"] * 0.5
    df["text"]   = df["text"].astype(str)
    return df

# ---------- header-by-words (preferred) ----------
def _find_header_columns_by_words(pages: List[dict], debug=False) -> Optional[Dict[str,int]]:
    """
    Robust header finder:
      1) collect only header words: description / booking / date / amount
      2) cluster by Y (same line)
      3) pick the line that contains 'amount' and ('booking' or 'date')
      4) compute column windows from *that* line only
    """
    header_words = {"description", "booking", "date", "amount"}

    for page_idx, page in enumerate(pages[:2]):
        df = _page_words_to_df(page)
        if df.empty:
            continue

        # keep only possible header tokens
        tok = df.assign(t=df["text"].str.strip().str.lower())
        hdr = tok[tok["t"].isin(header_words)].copy()
        if hdr.empty:
            continue

        # cluster by Y (tight: same printed line)
        med_h = _med_height(hdr)
        gap = max(6.0, med_h * 0.60)
        hdr = hdr.sort_values("cy").reset_index(drop=True)

        lines: List[pd.DataFrame] = []
        start = 0
        for i in range(1, len(hdr)):
            if hdr.loc[i, "cy"] - hdr.loc[i-1, "cy"] > gap:
                lines.append(hdr.iloc[start:i].sort_values("left"))
                start = i
        lines.append(hdr.iloc[start:].sort_values("left"))

        # score lines; must have AMOUNT and (BOOKING or DATE)
        candidates = []
        for ln in lines:
            words = set(ln["t"].tolist())
            if "amount" in words and (("booking" in words) or ("date" in words)):
                left_min  = int(ln["left"].min())
                right_max = int((ln["left"] + ln["width"]).max())
                cy_med    = float(ln["cy"].median())
                width     = right_max - left_min
                candidates.append((ln, width, left_min, right_max, cy_med))

        if not candidates:
            # no good line on this page, try next page
            continue

        # choose widest header line (it spans Description .. Amount)
        candidates.sort(key=lambda x: (-x[1], x[2]))
        ln, _, _, _, cy_med = candidates[0]

        # derive column windows from this line only
        amount_tokens = ln[ln["t"].eq("amount")]
        booking_tokens = ln[ln["t"].eq("booking")]
        date_tokens = ln[ln["t"].eq("date")]
        desc_tokens = ln[ln["t"].eq("description")]

        amount_left  = int(amount_tokens["left"].median())
        amount_right = int((amount_tokens["left"] + amount_tokens["width"]).median())

        # date column spans "Booking"+"Date" block; take min left / max right from those two
        if not booking_tokens.empty and not date_tokens.empty:
            date_left_hard = int(min(booking_tokens["left"].median(), date_tokens["left"].median()))
            date_right_hdr = int(max((booking_tokens["left"] + booking_tokens["width"]).median(),
                                     (date_tokens["left"] + date_tokens["width"]).median()))
        else:
            src = booking_tokens if not booking_tokens.empty else date_tokens
            date_left_hard = int(src["left"].median())
            date_right_hdr = int((src["left"] + src["width"]).median())

        # pad windows
        date_left  = date_left_hard - 80
        date_right = date_right_hdr + 80

        amount_left_pad  = amount_left - 140
        amount_right_pad = amount_right + 160

        # enforce separation: date must be strictly left of amount
        if date_right >= amount_left_pad:
            date_right = amount_left_pad - 12

        desc_max_right = date_left_hard - 40

        if debug:
            hdr_line_text = " | ".join(ln.sort_values("left")["text"].astype(str))
            print(f"[COLS] (by header words) page={page_idx+1} line_y≈{int(cy_med)} "
                  f"date_left_hard={date_left_hard}  "
                  f"date_win=({date_left},{date_right})  amount_win=({amount_left_pad},{amount_right_pad})")
            print(f"      header-line: {hdr_line_text}")

        return dict(
            date_left=date_left, date_right=date_right,
            amount_left=amount_left_pad, amount_right=amount_right_pad,
            desc_max_right=desc_max_right, date_left_hard=date_left_hard,
            source="header", header_text="Description | Booking Date | Amount"
        )

    return None

# ---------- clustering helpers ----------
def _cluster_by_left(xs: List[int], gap: int) -> List[List[int]]:
    if not xs: return []
    xs = sorted(xs)
    clusters, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= gap:
            cur.append(x)
        else:
            clusters.append(cur); cur = [x]
    clusters.append(cur)
    return clusters

# ---------- cluster fallback (close-to-amount) ----------
def _infer_columns_by_clusters(pages: List[dict], debug=False) -> Optional[Dict[str,int]]:
    for page_idx, page in enumerate(pages[:2]):
        df = _page_words_to_df(page)
        if df.empty:
            continue

        # candidates
        is_date = df["text"].apply(lambda t: bool(DATE_RE.search(t)))
        dates = df.loc[is_date, ["left","right","top","bottom","cy","text"]].copy()

        has_euro = df["text"].str.contains("€", regex=False, na=False)
        amts = df.loc[has_euro, ["left","right","top","bottom","cy","text"]].copy()

        if dates.empty or amts.empty:
            continue

        # clusters
        date_clusters = _cluster_by_left(dates["left"].astype(int).tolist(), gap=140)
        amt_clusters  = _cluster_by_left(amts["left"].astype(int).tolist(), gap=180)

        d_scored, a_scored = [], []
        for cl in date_clusters:
            g = dates[dates["left"].isin(cl)]
            d_scored.append({
                "size": int(len(g)),
                "left_med": int(g["left"].median()),
                "right_med": int(g["right"].median()),
                "cy_med": float(g["cy"].median()),
            })
        for cl in amt_clusters:
            g = amts[amts["left"].isin(cl)]
            a_scored.append({
                "size": int(len(g)),
                "left_med": int(g["left"].median()),
                "right_med": int(g["right"].median()),
                "cy_med": float(g["cy"].median()),
            })
        if not d_scored or not a_scored:
            continue

        # amount = most hits, then far-right
        a_scored.sort(key=lambda s: (-s["size"], s["left_med"]))
        a_pick = a_scored[-1]
        amount_left  = a_pick["left_med"] - 140
        amount_right = a_pick["right_med"] + 160

        # date: must be left of amount; choose **closest** to amount on the left
        left_of_amount = [s for s in d_scored if s["right_med"] < amount_left - 20]
        if not left_of_amount:
            left_of_amount = d_scored  # last resort

        for s in left_of_amount:
            s["_dx"] = max(1, a_pick["left_med"] - s["left_med"])  # smaller is better

        # primary: min distance to amount; secondary: more rows; tertiary: right-most
        left_of_amount.sort(key=lambda s: (s["_dx"], -s["size"], -s["left_med"]))
        d_pick = left_of_amount[0]

        date_left_hard = d_pick["left_med"]
        date_left  = date_left_hard - 80
        date_right = d_pick["right_med"] + 80
        if date_right >= amount_left:
            date_right = amount_left - 10

        if debug:
            print(f"[COLS] (by clusters) page={page_idx+1} date_left_hard={date_left_hard}  "
                  f"date_win=({date_left},{date_right})  amount_win=({amount_left},{amount_right})")
            for s in sorted(d_scored, key=lambda k:k["left_med"]):
                flag = "  <-- pick" if s is d_pick else ""
                extra = f" dx={a_pick['left_med']-s['left_med']}" if "left_med" in s else ""
                print(f"      date cluster: x≈{s['left_med']} size={s['size']} cy≈{int(s['cy_med'])}{extra}{flag}")
            for s in sorted(a_scored, key=lambda k:k["left_med"]):
                print(f"      amt  cluster: x≈{s['left_med']} size={s['size']} cy≈{int(s['cy_med'])}"
                      f"{'  <-- pick' if s is a_pick else ''}")

        return dict(
            date_left=date_left, date_right=date_right,
            amount_left=amount_left, amount_right=amount_right,
            desc_max_right=date_left_hard - 40,
            date_left_hard=date_left_hard,
            source="clusters", header_text="Description | Booking Date | Amount"
        )
    return None

# ---------- rows ----------
def _yclusters(df: pd.DataFrame, gap: Optional[float]=None) -> List[pd.DataFrame]:
    if df.empty: return []
    med_h = _med_height(df)
    if gap is None: gap = max(6.0, med_h * 0.45)
    df = df.sort_values("cy").reset_index(drop=True)
    out, start = [], 0
    for i in range(1, len(df)):
        if df.loc[i, "cy"] - df.loc[i-1, "cy"] > gap:
            out.append(df.iloc[start:i].sort_values("left")); start = i
    out.append(df.iloc[start:].sort_values("left"))
    return out

# ---------- field extractors ----------
def _extract_topmost_date_in_window(row_df: pd.DataFrame, left: int, right: int) -> tuple[Optional[str], Optional[int]]:
    if row_df.empty: return None, None
    m = (row_df["right"] >= left) & (row_df["left"] <= right)
    cand = row_df.loc[m, ["text","top","height","cy","left","right"]].copy()
    if cand.empty: return None, None

    toks = []
    for _, r in cand.iterrows():
        for t in str(r["text"]).split():
            mdt = DATE_RE.search(t)
            if mdt:
                toks.append((t, int(r["top"]), float(r.get("cy", r["top"] + r["height"] * 0.5))))
    if toks:
        tmin, _, cy = min(toks, key=lambda x: x[1])
        d, mth, y = DATE_RE.search(tmin).groups()
        if len(y) == 2: y = "20" + y
        return f"{d.zfill(2)}.{mth.zfill(2)}.{y}", int(round(cy))

    txt = _norm(" ".join(cand["text"].tolist())).replace(" ", "")
    m2 = DATE_RE.search(txt)
    if not m2: return None, None
    d, mth, y = m2.groups()
    if len(y) == 2: y = "20" + y
    return f"{d.zfill(2)}.{mth.zfill(2)}.{y}", int(round(float(cand["cy"].min())))

def _extract_amount_text(row_df: pd.DataFrame, left: int, right: int) -> Optional[str]:
    """Strict per-row amount parsing: require '€' inside the window."""
    if row_df.empty: return None
    m = (row_df["right"] >= left) & (row_df["left"] <= right)
    if not m.any(): return None
    txt = _norm(" ".join(row_df.loc[m, "text"].tolist()))
    if "€" not in txt:
        return None
    mt = AMOUNT_RE.search(txt)
    if not mt: return None
    sign = mt.group("sign") or ""
    sign = "-" if sign in ("-","−") else ("+" if sign == "+" else "")
    body = (mt.group("body")
            .replace("\u00A0"," ").replace("\u2009"," ").replace("\u202F"," ").replace(" ",""))
    if "," not in body and "." in body:
        i = body.rfind("."); body = body[:i] + "," + body[i+1:]
    return f"{sign}{body}€"

# ---------- description (seed line + baseline) ----------
def _pick_seed_token(left_df: pd.DataFrame, anchor_y: float) -> Optional[pd.Series]:
    if left_df is None or left_df.empty: return None
    cand = left_df.copy()
    avoid = cand["text"].str.strip().str.lower().str.match(r"(value|date|mastercard)\b", na=False)
    good  = cand.loc[~avoid]
    target = good if not good.empty else cand
    target = target.assign(dy=(target["cy"] - float(anchor_y)).abs())
    above  = target[target["cy"] <= float(anchor_y) + 0.5]
    if not above.empty:
        return above.sort_values(["dy","left"]).iloc[0]
    return target.sort_values(["dy","left"]).iloc[0]

def _merge_same_tess_line(page_df: pd.DataFrame, seed: pd.Series, date_left_hard: int) -> str:
    b, p, l = int(seed["block_num"]), int(seed["par_num"]), int(seed["line_num"])
    same = page_df[(page_df["block_num"]==b)&(page_df["par_num"]==p)&(page_df["line_num"]==l)]
    if same.empty:
        return _norm(str(seed["text"]))
    left_cut = int(date_left_hard) - 2
    same = same[same["left"] < left_cut].sort_values("left")
    txt = " ".join(_norm(t) for t in same["text"].astype(str)).strip()
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt or _norm(str(seed["text"]))

def _merge_same_baseline(page_df: pd.DataFrame, seed: pd.Series, date_left_hard: int) -> str:
    left_cut = int(date_left_hard) - 2
    left = page_df[page_df["left"] < left_cut].copy()
    if left.empty:
        return _norm(str(seed["text"]))
    med_h = _med_height(left)
    tol = max(6.0, med_h * 0.85)
    band = left[left["cy"].between(float(seed["cy"]) - tol, float(seed["cy"]) + tol)].sort_values("left")
    if band.empty:
        return _norm(str(seed["text"]))
    txt = " ".join(_norm(t) for t in band["text"].astype(str)).strip()
    txt = re.sub(r"\s{2,}", " ", txt)
    if FURNITURE.match(txt):
        upper = left[left["cy"].between(float(seed["cy"]) - tol*1.1, float(seed["cy"]) - tol*0.2)].sort_values("left")
        txt2 = " ".join(_norm(t) for t in upper["text"].astype(str)).strip()
        if txt2:
            txt = re.sub(r"\s{2,}", " ", txt2)
    return txt or _norm(str(seed["text"]))

def _description_seedline(page_df: pd.DataFrame, date_left_hard: int, anchor_y: float, debug=False) -> Optional[str]:
    if page_df is None or page_df.empty:
        return None
    left_cut = int(date_left_hard) - 2
    left_all = page_df[page_df["left"] < left_cut].copy()
    if left_all.empty:
        return None

    med_h = _med_height(left_all)
    seed_band = left_all[left_all["cy"].between(float(anchor_y) - 1.8*med_h, float(anchor_y) + 1.0*med_h)]
    if seed_band.empty:
        seed_band = left_all

    seed = _pick_seed_token(seed_band, anchor_y)
    if seed is None:
        return None

    line_txt = _merge_same_tess_line(page_df, seed, date_left_hard)
    need_fallback = (len(line_txt.split()) <= 1) or bool(FURNITURE.match(line_txt))
    desc = line_txt if not need_fallback else _merge_same_baseline(page_df, seed, date_left_hard)

    if debug:
        b, p, l = int(seed["block_num"]), int(seed["par_num"]), int(seed["line_num"])
        same = page_df[(page_df["block_num"]==b)&(page_df["par_num"]==p)&(page_df["line_num"]==l)]
        same = same[same["left"] < left_cut].sort_values("left")
        dbg_line = " | ".join(same["text"].astype(str))
        print(f"  [DBG] seed='{seed['text']}' cy={round(float(seed['cy']),1)} line={b}/{p}/{l}")
        print(f"  [DBG] tess-line tokens: {dbg_line if dbg_line else '(none left of date)'}")
        print(f"  [DBG] desc='{desc}' (line merge{' + baseline' if need_fallback else ''})")

    return desc or None

# ---------- main ----------
def parse_and_preview_n26(raw_ocr: dict, max_rows: int = 50, debug: bool = False) -> List[Tuple[str,str,str]]:
    pages = raw_ocr.get("pages", []) or []
    if not pages:
        print("No pages in OCR.")
        return []

    # try header first, then cluster fallback
    cols = _find_header_columns_by_words(pages, debug=debug)
    if not cols:
        cols = _infer_columns_by_clusters(pages, debug=debug)
    if cols:
        print(f"📑 Header: {cols['header_text']}")
    else:
        print("📑 Header not found; using crude column guesses.")
        cols = {
            "date_left": 900, "date_right": 1150,
            "amount_left": 1500, "amount_right": 1850,
            "desc_max_right": 850, "date_left_hard": 940,
            "source":"fallback", "header_text":""
        }

    out: List[Tuple[str,str,str]] = []
    shown = 0
    seen = set()

    for pi, page in enumerate(pages):
        df = _page_words_to_df(page)
        if df.empty: continue

        med_h_page = _med_height(df)
        ybin_h = max(10.0, med_h_page * 0.9)

        for idx, row in enumerate(_yclusters(df)):
            amount = _extract_amount_text(row, cols["amount_left"], cols["amount_right"])
            if not amount:
                continue
            date, anchor_y = _extract_topmost_date_in_window(row, cols["date_left"], cols["date_right"])
            if not date:
                continue

            # dedupe by (page, ybin, amount)
            rowkey = (pi, int(round(float(anchor_y) / ybin_h)), amount)
            if rowkey in seen:
                if debug:
                    print(f"  [DBG] skip duplicate p{pi+1} idx={idx} y~{round(float(anchor_y),1)}")
                continue
            seen.add(rowkey)

            if debug:
                print(f"[ROW] p{pi+1} idx={idx} y~{round(float(anchor_y),1)} date={date} amt={amount}")

            desc = _description_seedline(df, cols["date_left_hard"], float(anchor_y), debug=debug) or ""

            out.append((desc, date, amount))
            if shown < max_rows:
                print(f"{desc}   {date}   {amount}")
                shown += 1

        if shown >= max_rows:
            break

    if not out:
        print("No transactions detected.")
    else:
        print(f"…showing {min(len(out), max_rows)} of {len(out)} matches")
    return out
