lines = []
with open('requirements.txt', 'r+') as f:
    for l in f.readlines():
        lines.append(l.split('@')[0].strip())

with open('requirements2.txt', 'w+') as f:
    for l in lines:
        f.write(l + '\n')
