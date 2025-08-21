import argparse
from pathlib import Path

import spacy
from TTS.api import TTS


def extract_keywords(text: str, nlp, num_keywords=5):
    doc = nlp(text)
    # Chọn danh từ & danh từ riêng
    keywords = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
    return keywords[:num_keywords]


def add_emphasis(text: str, keywords, emphasis_level=1.5):
    # LƯU Ý: Coqui TTS không hỗ trợ SSML <emphasis> chính tắc.
    # Đoạn này chỉ giữ nguyên logic từ file gốc của bạn.
    for kw in keywords:
        text = text.replace(kw, f"<emphasis level='{emphasis_level}'>{kw}</emphasis>")
    return text


def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech (Coqui TTS) CLI")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Chuỗi văn bản đầu vào")
    g.add_argument("--text-file", type=str, help="Đường dẫn file .txt input")

    parser.add_argument("--out", type=str, default="/out/output.wav",
                        help="Đường dẫn file WAV output (mặc định: /out/output.wav)")
    parser.add_argument("--model", type=str,
                        default="tts_models/en/ljspeech/tacotron2-DDC",
                        help="Tên model Coqui TTS")
    parser.add_argument("--no-emphasis", action="store_true",
                        help="Bỏ bước chèn <emphasis> (vì đa số model không hiểu SSML)")
    args = parser.parse_args()

    # Load text
    if args.text:
        text = args.text
    else:
        text = Path(args.text_file).read_text(encoding="utf-8")

    # spaCy
    nlp = spacy.load("en_core_web_sm")
    keywords = extract_keywords(text, nlp, num_keywords=5)
    

    # Emphasis (tùy chọn)
    final_text = text if args.no_emphasis else add_emphasis(text, keywords)

    # Coqui TTS
    tts = TTS(model_name=args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("🔎 Extracted keywords:", keywords)
    tts.tts_to_file(text=final_text, file_path=str(out_path))

    print(f"✅ Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
