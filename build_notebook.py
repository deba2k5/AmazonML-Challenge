"""Convert kaggle_entity_resolution.py ("# %%" cells) into a Jupyter notebook for Kaggle."""
import json, re, sys

src = sys.argv[1] if len(sys.argv) > 1 else "kaggle_entity_resolution.py"
dst = src.replace(".py", ".ipynb")
text = open(src, encoding="utf-8").read()

cells = []
for block in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if header.strip() == "[markdown]":
        lines = [re.sub(r"^# ?", "", l) for l in body.splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"wrote {dst} ({len(cells)} cells)")
