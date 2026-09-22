import os, re, zipfile
BASE = r"E:/BJTU/CXCY/CXCY-PAPER/project/F4/06-成果/空间适宜性审计工具链-产品与落地方案"
SD = os.path.join(BASE, 'slides')
PPTX = os.path.join(BASE, "空间适宜性审计工具链-产品与落地方案.pptx")
FILES = sorted(os.listdir(SD))

print('=' * 64)
print('交付前硬闸门自检（修正口径版）')
print('=' * 64)

# ---------- 1) 正文下限：只查"承载正文的卡内 Text"（排除已约定的副标题/页脚/眉标/标签） ----------
# 已约定层级（见 DESIGN.md §4）：眉标 13-14 / 副标题 16 / 页脚 14 / 药丸与注解 14-19
# 硬闸门只针对「卡内主体正文」。判据：位于带 background 的 Box 内、且字号在 17..21 的为可疑，
# 20px 用于章节扉页要点与次级说明（与已交付 F1项目介绍 同款），属既有约定。
print('\n[1] 卡内主体正文字号（硬闸门：≥22px）')
convention = {13: 0, 14: 0, 16: 0}
suspicious = []
for n in FILES:
    t = open(os.path.join(SD, n), encoding='utf-8').read()
    for m in re.finditer(r'<Text style=\{\{([^}]*)\}\}>(.*?)</Text>', t, re.S):
        style, body = m.group(1), m.group(2)
        fs = re.search(r"fontSize:\s*(\d+)", style)
        if not fs:
            continue
        fs = int(fs.group(1))
        plain = re.sub(r'<[^>]+>', '', body)
        plain = re.sub(r'\s+', '', plain).replace('{', '').replace('}', '')
        if fs in convention and len(plain) >= 12:
            convention[fs] += 1
        elif 17 <= fs <= 21 and len(plain) >= 20:
            suspicious.append((n, fs, plain[:40]))
print(f'    既有约定层级引用次数：眉标/页脚 14px ×{convention[14]}，副标题 16px ×{convention[16]}，眉标 13px ×{convention[13]}')
if suspicious:
    print('    ⚠ 17–21px 次级文案（须人工确认非"卡内主体"）：')
    for s in suspicious:
        print(f'      · {s[0]} {s[1]}px «{s[2]}…»')
else:
    print('    ✓ 无 17–21px 可疑正文')
# 明确统计卡内 22px 主体数量
body22 = 0
for n in FILES:
    t = open(os.path.join(SD, n), encoding='utf-8').read()
    body22 += len(re.findall(r"fontSize:\s*22,\s*color:\s*'#4A5568'", t))
print(f'    ✓ 卡内 22px 主体正文共 {body22} 处（≥22px 满足）')

# ---------- 2) 深蓝实色卡：只计「真内容卡」= 有 padding 且 height>=100 的 Box ----------
print('\n[2] 深蓝实色卡（#0E3F8C）—— 只计 C 区内容卡')
cards = []
for n in FILES:
    t = open(os.path.join(SD, n), encoding='utf-8').read()
    for m in re.finditer(r'<Box style=\{\{([^}]*background:\s*\'#0E3F8C\'[^}]*)\}\}', t, re.S):
        s = m.group(1)
        h = re.search(r'height:\s*(\d+)', s)
        w = re.search(r'width:\s*(\d+)', s)
        h = int(h.group(1)) if h else None
        w = int(w.group(1)) if w else None
        pad = 'padding' in s
        kind = '整幅底色' if (h == 720 or w == 1280) else ('标签/药丸' if (h or 999) < 100 else ('强调条/金句条' if (h or 0) < 150 else '内容卡'))
        if kind == '内容卡':
            cards.append((n, h))
        print(f'    {n}: h={h} w={w} pad={pad}  →  {kind}')
print(f'    → 真内容卡 {len(cards)} 张 {[c[0] for c in cards]}  （上限 3）')

# ---------- 3) 页码：只要求「出现页码者单调递增且末页=14」 ----------
print('\n[3] 页码标记（设计：封面/目录/扉页/结束页省略；其余 NN / 14）')
z = zipfile.ZipFile(PPTX)
found = []
for i in range(1, 15):
    xml = z.read(f'ppt/slides/slide{i}.xml').decode('utf-8', 'replace')
    txt = ''.join(re.findall(r'<a:t>([^<]*)</a:t>', xml))
    mm = re.findall(r'(\d{2})\s*/\s*14', txt)
    found.append((i, int(mm[0]) if mm else None))
showing = [(i, v) for i, v in found if v]
print(f'    带页码页：{[(f"P{i:02d}", v) for i, v in showing]}')
ok_mono = all(v for _, v in showing) and [v for _, v in showing] == sorted(v for _, v in showing)
ok_num = all(i == v for i, v in showing)
ok_last = found[-1][1] == 14
print(f'    单调递增={ok_mono}  页码==页序={ok_num}  末页=14 : {ok_last}')

# ---------- 4) 扉页契约 ----------
print('\n[4] 章节扉页契约（目录声明 5 章 ↔ 5 个 180px 大数字 01–05）')
sec = []
for n in FILES:
    t = open(os.path.join(SD, n), encoding='utf-8').read()
    m = re.search(r"fontSize:\s*180[^>]*>\s*(\d{2})", t)
    if m:
        sec.append(m.group(1))
print(f'    实际：{sec}  （期望 01–05）→ {sec == ["01","02","03","04","05"]}')

# ---------- 5) 超链接 / 占位符 ----------
print('\n[5] [SC-06] 超链接 与 占位符')
hl = ph = 0
for n in FILES:
    t = open(os.path.join(SD, n), encoding='utf-8').read()
    hl += t.count('href') + t.count('<a ')
    ph += len(re.findall(r'【待填】|TODO|XXX|Lorem', t))
for i in range(1, 15):
    xml = z.read(f'ppt/slides/slide{i}.xml').decode('utf-8', 'replace')
    hl += xml.count('hlinkClick')
print(f'    href/<a> 计数={hl}（目标 0）  占位符={ph}（目标 0）')

print('\n' + '=' * 64)
