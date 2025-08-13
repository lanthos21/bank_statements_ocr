import os, pandas as pd

def save_ocr_words_csv(raw_ocr, out_path="results/ocr_words.csv"):
    rows = []
    for p in raw_ocr.get("pages", []):
        page_no = p.get("page_number")
        for line in p.get("lines", []):
            line_y = line.get("y")
            for w in line.get("words", []):
                left   = int(w["left"]);  top = int(w["top"])
                width  = int(w["width"]); height = int(w["height"])
                right  = left + width;    bottom = top + height
                rows.append({
                    "page": page_no,
                    "block_num": w.get("block_num"),
                    "par_num":   w.get("par_num"),
                    "line_num":  w.get("line_num"),
                    "word_num":  w.get("word_num"),
                    "conf":      w.get("conf"),
                    "text":      str(w.get("text", "")),
                    "left": left, "top": top, "width": width, "height": height,
                    "right": right, "bottom": bottom,
                    "cx": left + width/2.0, "cy": top + height/2.0,
                    "line_y": line_y,
                })
    df = pd.DataFrame(rows)
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.sort_values(["page", "line_num", "left"], inplace=True)
    df.to_csv(out_path, index=False)
    print(f"🔎 OCR words CSV saved to {out_path}")


def save_ocr_pretty_txt(raw_ocr, out_path="results/ocr_words_pretty.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        for p in raw_ocr.get("pages", []):
            f.write(f"=== Page {p.get('page_number')} ===\n")
            for ln in p.get("lines", []):
                words = ln.get("words", [])
                if not words:
                    continue
                cy = int(round(sum(w["top"] + w["height"] / 2 for w in words) / len(words)))
                line_text = " ".join(w["text"] for w in sorted(words, key=lambda w: w["left"]))
                f.write(f"y≈{cy:>5}  {line_text}\n")
                for w in sorted(words, key=lambda w: w["left"]):
                    left, top = int(w["left"]), int(w["top"])
                    right, bottom = left + int(w["width"]), top + int(w["height"])
                    f.write(
                        f"    {w['text']:<22} "
                        f"[{left:>4},{top:>4}]–[{right:>4},{bottom:>4}]  "
                        f"b/p/l/w={w.get('block_num')}/{w.get('par_num')}/{w.get('line_num')}/{w.get('word_num')}  "
                        f"conf={w.get('conf')}\n"
                    )
            f.write("\n")
    print(f"📝 Pretty OCR dump saved to {out_path}")
