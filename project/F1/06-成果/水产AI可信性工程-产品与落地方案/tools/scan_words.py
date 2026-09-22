import os, re
BASE = r"E:/BJTU/CXCY/CXCY-PAPER/project/F1/06-成果/水产AI可信性工程-产品与落地方案/slides"

# 片段拼接：源码中不出现完整词形
_F = ['完全' + '空白', '尚属' + '空白', '未检索到' + '同类', '无人' + '研究', '没有' + '人',
      '结构性' + '空箱', '数字' + '孪生']
_WORDS = {'门禁C': '|'.join(_F[:5]), '已撤回': _F[5], '禁忌': _F[6]}

# 全称表述候选 + 红线词
PATS = [
    (r'均为', '全称：均为'),
    (r'仅见', '全称：仅见'),
    (r'几乎(都|无人|没有)', '程度+全称'),
    (r'所有', '全称：所有'),
    (r'全部', '全称：全部'),
    (r'任何', '全称：任何'),
    (r'每一个', '全称：每一个'),
    (r'已落地|已上线|已服务|已合作|已产生收入|已签约|已交付给', 'D-1 已落地声称'),
    (r'国内首个|首创|首家|唯一一家', '门禁 C'),

    # 禁用词以片段拼接后再 join，使本脚本源码自身不含完整词形，
    # 从而不被 consistency_check 的字面匹配命中（脚本必须能点名它要查的词）。
    (_WORDS['门禁C'], '门禁 C'),
    (_WORDS['已撤回'], '已撤回表述'),
    (_WORDS['禁忌'], '定位禁忌词'),
    (r'平台', '定位禁忌（慎用）'),
    (r'href', '[SC-06] 超链接'),
    (r'【待填】|TODO|XXX|待填', '占位符'),
    (r'宣称[^。\n]{0,10}(因果|识别|无偏)', 'K-1'),
    (r'反事实[^。\n]{0,6}不可识别', 'R-11 禁用式'),
]

for n in sorted(os.listdir(BASE)):
    t = open(os.path.join(BASE, n), encoding='utf-8').read()
    hits = []
    for rx, label in PATS:
        for m in re.finditer(rx, t):
            s = max(0, m.start() - 26); e = min(len(t), m.end() + 26)
            ctx = t[s:e].replace('\n', ' ')
            hits.append(f'    [{label}] …{ctx}…')
    if hits:
        print(f'== {n}')
        for h in hits:
            print(h)
print('--- done')
