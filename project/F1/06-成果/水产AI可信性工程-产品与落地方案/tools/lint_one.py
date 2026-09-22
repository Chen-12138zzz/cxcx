import json, os, subprocess, sys
NODE = r"C:/Users/Chen/.workbuddy/binaries/node/versions/22.22.2-3/node.exe"
ENTRY = r"C:/Users/Chen/.workbuddy/binaries/node/versions/22.22.2-3/node_modules/@tencent/slidep/dist/index.js"
BASE = r"E:/BJTU/CXCY/CXCY-PAPER/project/F1/06-成果/水产AI可信性工程-产品与落地方案"
PPTX = os.path.join(BASE, "水产AI可信性工程-产品与落地方案.pptx")
i = int(sys.argv[1])
p = subprocess.run([NODE, ENTRY, 'lint', PPTX, '--dsl-file', os.path.join(BASE,'slides',f'{i:02d}.slide')],
                   capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=BASE)
out = (p.stdout or '') + (p.stderr or '')
for line in out.splitlines():
    line = line.strip()
    if line.startswith('{'):
        r = json.loads(line)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        break
else:
    print(out)
