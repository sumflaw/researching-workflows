#!/usr/bin/env python3
"""Machine-translate empty strings in Qt Linguist .ts files using NLLB.

Environment:
  TARGET_LANG    optional filter, e.g. "fr" or "pt_BR" (empty = every language found)
  MODEL_NAME     Hugging Face model id
  MARK_FINISHED  "true" to drop type="unfinished" (default: keep it for human review)
  BATCH_SIZE     strings per generate() call (default 16)
"""
import os
import re
import sys
from pathlib import Path

from lxml import etree

# ISO / Qt locale  ->  NLLB (FLORES-200) code
NLLB = {
    "ar": "arb_Arab", "bg": "bul_Cyrl", "bn": "ben_Beng", "ca": "cat_Latn",
    "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn", "eu": "eus_Latn",
    "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "gl": "glg_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "nb": "nob_Latn", "nl": "nld_Latn",
    "no": "nob_Latn", "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn", "sr": "srp_Cyrl",
    "sv": "swe_Latn", "th": "tha_Thai", "tr": "tur_Latn", "uk": "ukr_Cyrl",
    "vi": "vie_Latn", "zh": "zho_Hans", "zh_CN": "zho_Hans", "zh_TW": "zho_Hant",
}

EXCLUDE_DIRS = {".git", "node_modules", "build", "3rdparty", "third_party", "external"}
# printf-style / Qt placeholders, format braces, markup, newlines, literal "&&", entities
PROTECT = re.compile(r"%%|%L?\d+|%n|\{[^{}]*\}|<[^>]+>|&&|\n|&[#\w]+;")

SUMMARY = os.getenv("GITHUB_STEP_SUMMARY")


def log(msg):
    print(msg, flush=True)
    if SUMMARY:
        with open(SUMMARY, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def norm(lang):
    return (lang or "").replace("-", "_").lower()


def nllb_code(lang):
    if not lang:
        return None
    lang = lang.replace("-", "_")
    return NLLB.get(lang) or NLLB.get(lang.split("_")[0])


def matches_filter(lang, flt):
    if not flt:
        return True
    lang, flt = norm(lang), norm(flt)
    return lang == flt or lang.split("_")[0] == flt


def find_ts_files():
    """Only real Qt Linguist files (root <TS>), not TypeScript sources."""
    for p in sorted(Path(".").rglob("*.ts")):
        if EXCLUDE_DIRS & set(p.parts):
            continue
        try:
            head = p.read_bytes()[:2048]
        except OSError:
            continue
        if b"<TS" in head:
            yield p


def file_language(root, path):
    lang = root.get("language")
    if lang:
        return lang
    m = re.search(r"[_.-]([a-z]{2,3}(?:[_-][A-Za-z]{2,4})?)\.ts$", path.name)
    return m.group(1).replace("-", "_") if m else None


def needs_work(msg):
    tr = msg.find("translation")
    if not (msg.findtext("source") or "").strip():
        return False
    if tr is None:
        return True
    if tr.get("type") in ("vanished", "obsolete"):
        return False
    if msg.get("numerus") == "yes":
        forms = tr.findall("numerusform")
        return bool(forms) and any(not (f.text or "").strip() for f in forms)
    return not (tr.text or "").strip()


def prepare(text):
    """-> (masked_text, protected_list) or None if there is nothing to translate."""
    lead = re.match(r"\s*", text).group()
    trail = re.search(r"\s*$", text).group()
    core = text.strip()
    protected = []

    def mask(m):
        protected.append(m.group(0))
        return "{%d}" % (len(protected) - 1)

    masked = PROTECT.sub(mask, core)
    masked = re.sub(r"&(?=[^\W\d_])", "", masked)  # drop accelerator marks (&File)
    if not re.search(r"[^\W\d_]", re.sub(r"\{\d+\}", "", masked)):
        return None
    return masked, protected, lead, trail


def restore(result, protected, lead, trail):
    result = re.sub(r"\{\s*(\d+)\s*\}", r"{\1}", result.strip())
    found = [int(i) for i in re.findall(r"\{(\d+)\}", result)]
    if sorted(found) != list(range(len(protected))):
        return None  # model dropped/duplicated a placeholder -> don't trust it
    result = re.sub(r"\{(\d+)\}", lambda m: protected[int(m.group(1))], result)
    return lead + result + trail


class Translator:
    def __init__(self, model_name, batch_size):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        self.batch_size = batch_size
        self.cache = {}

    def translate_many(self, texts, src, tgt):
        tgt_id = self.tok.convert_tokens_to_ids(tgt)
        if tgt_id == self.tok.unk_token_id:
            raise ValueError(f"Model has no language token {tgt}")
        self.tok.src_lang = src

        todo = []
        for t in texts:
            key = (src, tgt, t)
            if key in self.cache:
                continue
            prep = prepare(t)
            if prep is None:
                self.cache[key] = t  # numbers/symbols only: copy as-is
            else:
                todo.append((t, prep))
        todo.sort(key=lambda x: len(x[1][0]))

        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            enc = self.tok([c[1][0] for c in chunk], return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
            with self.torch.inference_mode():
                out = self.model.generate(
                    **enc, forced_bos_token_id=tgt_id, num_beams=4,
                    max_new_tokens=int(enc["input_ids"].shape[1] * 2 + 10))
            for (orig, (_, prot, lead, trail)), res in zip(
                    chunk, self.tok.batch_decode(out, skip_special_tokens=True)):
                self.cache[(src, tgt, orig)] = restore(res, prot, lead, trail)
            print(f"  {min(i + self.batch_size, len(todo))}/{len(todo)}", flush=True)

        return {t: self.cache.get((src, tgt, t)) for t in texts}


def apply(msg, text, mark_finished):
    tr = msg.find("translation")
    if tr is None:
        tr = etree.SubElement(msg, "translation")
    if msg.get("numerus") == "yes":
        for f in tr.findall("numerusform"):
            if not (f.text or "").strip():
                f.text = text
        tr.set("type", "unfinished")  # plural forms always need a human check
        return
    tr.text = text
    if mark_finished:
        tr.attrib.pop("type", None)  # Qt "finished" == no type attribute
    else:
        tr.set("type", "unfinished")


def write_ts(tree, path):
    body = etree.tostring(tree, encoding="utf-8", xml_declaration=False)
    path.write_bytes(b'<?xml version="1.0" encoding="utf-8"?>\n' + body + b"\n")


def main():
    flt = os.getenv("TARGET_LANG", "").strip()
    model_name = os.getenv("MODEL_NAME") or "facebook/nllb-200-distilled-600M"
    mark_finished = os.getenv("MARK_FINISHED") == "true"
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)

    plan = []
    for path in find_ts_files():
        tree = etree.parse(str(path), parser)
        root = tree.getroot()
        lang = file_language(root, path)
        if not matches_filter(lang, flt):
            continue
        src_lang = root.get("sourcelanguage") or "en"
        if not lang or norm(lang).split("_")[0] == norm(src_lang).split("_")[0]:
            continue
        tgt, src = nllb_code(lang), nllb_code(src_lang)
        if not tgt or not src:
            log(f"- `{path}`: skipped, no NLLB mapping for `{lang}`/`{src_lang}`")
            continue
        pending = [m for m in root.iter("message") if needs_work(m)]
        if not pending:
            log(f"- `{path}` ({lang}): nothing to translate")
            continue
        plan.append((path, tree, pending, src, tgt, lang))

    changed = False
    if plan:
        translator = Translator(model_name, batch_size)
        total = failed = 0
        for path, tree, pending, src, tgt, lang in plan:
            print(f"::group::{path} ({lang}, {len(pending)} strings)")
            texts = sorted({m.findtext("source") for m in pending})
            results = translator.translate_many(texts, src, tgt)
            done = 0
            for m in pending:
                res = results.get(m.findtext("source"))
                if res is None:
                    failed += 1
                    continue
                apply(m, res, mark_finished)
                done += 1
            if done:
                write_ts(tree, path)
                changed = True
            print("::endgroup::")
            log(f"- `{path}` ({lang}): {done}/{len(pending)} translated")
            total += done
        log(f"\n**Total translated: {total}, skipped (placeholder mismatch): {failed}**")
    else:
        log("No .ts files need translation.")

    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
