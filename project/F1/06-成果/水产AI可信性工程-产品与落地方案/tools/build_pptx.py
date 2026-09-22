import json, os, subprocess, sys

NODE = r"C:/Users/Chen/.workbuddy/binaries/node/versions/22.22.2-3/node.exe"
ENTRY = r"C:/Users/Chen/.workbuddy/binaries/node/versions/22.22.2-3/node_modules/@tencent/slidep/dist/index.js"
BASE = r"E:/BJTU/CXCY/CXCY-PAPER/project/F1/06-成果/水产AI可信性工程-产品与落地方案"
PPTX = os.path.join(BASE, "水产AI可信性工程-产品与落地方案.pptx")
N = 14

def run(args):
    p = subprocess.run([NODE, ENTRY] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', cwd=BASE)
    return (p.stdout or '') + (p.stderr or '')

# 1) 备份旧文件（若存在）
if os.path.exists(PPTX):
    bak = PPTX + '.bak'
    if os.path.exists(bak):
        os.remove(bak)
    os.rename(PPTX, bak)
    print('[backup] ->', os.path.basename(bak))

# 2) create 新建
print('[create]', run(['create', PPTX])[-500:])

# 3) 逐页 upsert，显式 0-based 索引
fails = []
for i in range(1, N + 1):
    f = os.path.join(BASE, 'slides', f'{i:02d}.slide')
    out = run(['upsert-dsl', PPTX, '--dsl-file', f, '--page-index', str(i - 1)])
    ok = '"failures":[]' in out.replace(' ', '') or '"ok":true' in out.replace(' ', '')
    mark = 'OK ' if ok else 'FAIL'
    print(f'[upsert P{i:02d} idx={i-1}] {mark} :: {out.strip().splitlines()[-1][:200]}')
    if not ok:
        fails.append((i, out[-400:]))

print('=== FAILS:', len(fails))
for i, o in fails:
    print(i, o)
