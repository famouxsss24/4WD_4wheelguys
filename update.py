from pathlib import Path

label_root = Path("/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/finaldata/train/labels")

for txt in label_root.rglob("*final*.txt"):   # final 파일만
    lines = txt.read_text().splitlines()
    new_lines = []

    for line in lines:
        parts = line.split()
        if parts[0] == "0":
            parts[0] = "8"   #brr-> final
        new_lines.append(" ".join(parts))

    txt.write_text("\n".join(new_lines))

print("완료")