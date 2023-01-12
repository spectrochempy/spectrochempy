import pathlib

REFERENCE = pathlib.Path("../docs/userguide/reference")

api = REFERENCE / "api.rst"
index = REFERENCE / "index.rst"

apitext = api.read_text()
content = apitext.split("\n")

apicontent = []
for line in content[:]:
    if line.startswith("    "):
        apicontent.append(line.strip())

# print('\n'.join(apicontent))

####

indextext = index.read_text()
content = indextext.split("\n")

idxcontent = []
for i, line in enumerate(content[:]):
    if line.startswith(".. autosummary"):
        j = i + 4
        while True:
            linej = content[j].strip()
            if linej != "":
                idxcontent.append(linej)
                j += 1
            else:
                break

# print('\n'.join(idxcontent))

# Now compare
for item in apicontent:
    if item not in idxcontent:
        if "." in item and not item.startswith("NDDataset"):
            continue
        print(f"    {item} : missing definition of  in index.rst")
