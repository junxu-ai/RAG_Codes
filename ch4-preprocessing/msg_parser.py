import os
import extract_msg
import datetime

def msg_to_txt(msg_path: str, txt_dir: str):
    msg = extract_msg.Message(msg_path)
    try:
        # Read core properties
        sender = msg.sender or ""
        to = msg.to or ""
        cc = msg.cc or ""
        d = msg.date or ""
        if isinstance(d, datetime.datetime):
            date = d.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            date = str(d)    
        subject = msg.subject or ""
        body = msg.body or ""
    finally:
        msg.close()

    # Sanitize subject for safe filename
    safe_subj = "".join(c for c in subject if c.isalnum() or c in (' ', '_')).strip()
    base = safe_subj or os.path.splitext(os.path.basename(msg_path))[0]
    filename = f"{date[:10]}_{base}.txt"
    out_path = os.path.join(txt_dir, filename)

    with open(out_path, "w", encoding="utf‑8") as f:
        f.write(f"From: {sender}\n")
        f.write(f"To: {to}\n")
        f.write(f"Cc: {cc}\n")
        f.write(f"Date: {date}\n")
        f.write(f"Subject: {subject}\n")
        f.write("\n---\n\n")
        f.write(body)

    print(f"Wrote: {out_path}")

def batch_convert(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(".msg"):
            path = os.path.join(input_dir, filename)
            try:
                msg_to_txt(path, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .msg files to .txt with metadata.")
    parser.add_argument("input_dir", help="Directory containing .msg files")
    parser.add_argument("output_dir", help="Directory to write .txt files")
    args = parser.parse_args()

    batch_convert(args.input_dir, args.output_dir)
