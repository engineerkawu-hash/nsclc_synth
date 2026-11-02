"""One-off repair of outputs/experiments/missing.csv (written before append_rows aligned columns).

Chunks whose first row was a 'not estimable' complete-case row carried an extra 'note' column
at position 5 (index 4), so every line in those chunks has one field more than the header.
Removing that field restores alignment; its text is kept in a proper 'note' column.
"""
import csv
import shutil

P = "outputs/experiments/missing.csv"
shutil.copy(P, P + ".bak")
with open(P + ".bak", newline="") as fh:
    rows = list(csv.reader(fh))
header, body = rows[0], rows[1:]
fixed, notes, n_shift = [], [], 0
for r in body:
    note = ""
    if len(r) == len(header) + 1:
        note = r.pop(4)
        n_shift += 1
    assert len(r) == len(header), f"unexpected width {len(r)}"
    assert r[2] and not r[2].replace(".", "").isdigit(), "model column misaligned"
    fixed.append(r + [note])
with open(P, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header + ["note"])
    w.writerows(fixed)
print(f"repaired {n_shift} shifted lines out of {len(body)}; notes: {sum(1 for f in fixed if f[-1])}")
