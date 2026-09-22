import os, re, zipfile, json

BASE = r"E:/BJTU/CXCY/CXCY-PAPER/project/F4/06-成果/空间适宜性审计工具链-产品与落地方案"
PPTX = os.path.join(BASE, "空间适宜性审计工具链-产品与落地方案.pptx")

z = zipfile.ZipFile(PPTX)
names = z.namelist()
pres = z.read('ppt/presentation.xml').decode('utf-8', 'replace')
rels = z.read('ppt/_rels/presentation.xml.rels').decode('utf-8', 'replace')

sld_ids = re.findall(r'<p:sldId id="(\d+)" r:id="([^"]+)"', pres)
rel_map = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))

print('slide_count(file) =', len(sld_ids))
print('zip ppt/slides/*.xml =', len([n for n in names if re.match(r'ppt/slides/slide\d+\.xml$', n)]))
print()
for i, (sid, rid) in enumerate(sld_ids, 1):
    tgt = rel_map.get(rid, '?')
    path = 'ppt/' + tgt.lstrip('/').replace('slides/', 'slides/') if not tgt.startswith('slides') else 'ppt/' + tgt
    path = 'ppt/' + tgt if not tgt.startswith('/') else tgt.lstrip('/')
    print(f'  order {i:2d}  sldId={sid:>4}  {rid:>8}  -> {tgt}')

# 逐页提取可见文本，做「页码标记」检查
print('\n--- 页脚页码标记检查 ---')
for i, (sid, rid) in enumerate(sld_ids, 1):
    tgt = rel_map.get(rid, '')
    p = 'ppt/' + tgt
    if p not in names:
        print(f'  P{i:02d}: (xml missing)')
        continue
    xml = z.read(p).decode('utf-8', 'replace')
    txt = ''.join(re.findall(r'<a:t>([^<]*)</a:t>', xml))
    m = re.findall(r'(\d{2})\s*/\s*14', txt)
    has_link = '<a:hlinkClick' in xml
    print(f'  P{i:02d}: page_mark={m}  hlink={has_link}  textlen={len(txt)}')

# 占位符 / 违规词扫描（从 DSL 源扫，更可靠）
print('\n--- DSL 源违规扫描 ---')
# 片段拼接：源码中不出现完整词形，避免本脚本自身被一致性校验命中
_FRAG = ['完全' + '空白', '尚属' + '空白', '未检索到' + '同类', '无人' + '研究']
bad_words = ['已落地', '已上线', '已服务', '已合作', '已产生收入',
             '国内' + '首个', '首创', '【待填】', 'TODO', 'XXX', 'href'] + _FRAG
sd = os.path.join(BASE, 'slides')
for n in sorted(os.listdir(sd)):
    t = open(os.path.join(sd, n), encoding='utf-8').read()
    hits = [w for w in bad_words if w in t]
    if hits:
        print(f'  {n}: {hits}')
print('  (无输出 = 干净)')
