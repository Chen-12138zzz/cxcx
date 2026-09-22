import json, os, subprocess, sys

NODE = r"C:/Users/Chen/.workbuddy/binaries/node/versions/22.22.2-3/node.exe"
ENTRY = r"C:/Users/Chen/.workbuddy/binaries/node/versions/22.22.2-3/node_modules/@tencent/slidep/dist/index.js"
BASE = r"E:/BJTU/CXCY/CXCY-PAPER/project/F4/06-成果/空间适宜性审计工具链-产品与落地方案"
PPTX = os.path.join(BASE, "空间适宜性审计工具链-产品与落地方案.pptx")

def run(args):
    p = subprocess.run([NODE, ENTRY] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', cwd=BASE)
    return (p.stdout or '') + (p.stderr or '')

def lint(i):
    f = os.path.join(BASE, 'slides', f'{i:02d}.slide')
    out = run(['lint', PPTX, '--dsl-file', f])
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {'ok': False, 'raw': out[-600:]}

if __name__ == '__main__':
    nums = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 15))
    bad = []
    for i in nums:
        f = os.path.join(BASE, 'slides', f'{i:02d}.slide')
        if not os.path.exists(f):
            print(f'P{i:02d}: (no file)')
            continue
        r = lint(i)
        ok = r.get('ok')
        dg = r.get('diagnostics', [])
        print(f'P{i:02d}: ok={ok} diag={len(dg)}')
        if not ok:
            bad.append(i)
            for d in dg[:6]:
                print('     ', json.dumps(d, ensure_ascii=False)[:300])
            if 'raw' in r:
                print('     RAW:', r['raw'][:400])
    print('---')
    print('BAD:', bad if bad else 'none')
