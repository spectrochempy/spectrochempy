from pathlib import Path


# ..........................................................................
def convert_py_to_pct_py():
    DOCS = Path(__file__).parent

    # clean notebooks output
    for py in DOCS.rglob("**/*.py"):
        if "userguide" not in py.parts:
            continue
        if "pct" in py.name:
            pyx = py.parent / py.name.replace(".pct", "")

        pctpy = pyx.with_suffix(".pct.py")
        print(py, "converted to ", pctpy)
        py.rename(pctpy)


if __name__ == "__main__":
    convert_py_to_pct_py()
