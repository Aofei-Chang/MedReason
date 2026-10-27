"""Parse PMC JATS XML into figure-level context, and link PubMedVision records to their figure.

Per PubMedVision image the output holds:
  record_type   'caption' (Original_Caption is the figure caption) or 'inline' (a body paragraph citing it)
  fig_label, fig_caption, citing_paragraphs (via <xref ref-type="fig">), abstract, title, article_type
"""
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

def _text(el):
    return re.sub(r'\s+', ' ', ''.join(el.itertext())).strip() if el is not None else ''

def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', re.sub(r'\s+', ' ', s.lower())).strip()

def parse_article(path):
    root = ET.parse(path).getroot()
    art = root if root.tag == 'article' else root.find('.//article')
    front = art.find('front')
    title = _text(front.find('.//article-title')) if front is not None else ''
    abstract = ' '.join(_text(a) for a in front.iter('abstract')) if front is not None else ''
    figs = {}
    for f in art.iter('fig'):
        figs[f.get('id')] = {'id': f.get('id'), 'label': _text(f.find('label')), 'caption': _text(f.find('caption'))}
    paragraphs = []
    body = art.find('body')
    if body is not None:
        def walk(node, section):
            for child in node:
                if child.tag == 'sec':
                    walk(child, section or _text(child.find('title')))
                elif child.tag == 'p':
                    rids = set()
                    for x in child.iter('xref'):
                        if x.get('ref-type') == 'fig':
                            rids.update((x.get('rid') or '').split())
                    paragraphs.append({'section': section, 'text': _text(child), 'fig_rids': sorted(rids)})
                elif child.tag not in ('fig', 'table-wrap'):
                    walk(child, section)
        walk(body, '')
    return {'title': title, 'abstract': abstract, 'article_type': art.get('article-type', ''),
            'figs': figs, 'paragraphs': paragraphs}

def _sim(a, b):
    return SequenceMatcher(None, norm(a)[:400], norm(b)[:400]).ratio()

def link_record(article, original_caption):
    """Locate the figure a PubMedVision record came from. Returns None when no figure matches."""
    oc = norm(original_caption)
    figs = list(article['figs'].values())
    best = max(figs, key=lambda f: _sim(f['caption'], original_caption), default=None)
    if best is not None and _sim(best['caption'], original_caption) >= 0.6:
        rtype, fig, source_par = 'caption', best, None
    else:
        key = oc[:150]
        source_par = next((p for p in article['paragraphs'] if key and key in norm(p['text'])), None)
        if source_par is None or not source_par['fig_rids']:
            return None
        rtype, fig = 'inline', article['figs'].get(source_par['fig_rids'][0])
        if fig is None:
            return None
    citing = [p for p in article['paragraphs'] if fig['id'] in p['fig_rids'] and p is not source_par]
    return {'record_type': rtype, 'fig_id': fig['id'], 'fig_label': fig['label'], 'fig_caption': fig['caption'],
            'citing_paragraphs': [{'section': p['section'], 'text': p['text']} for p in citing],
            'title': article['title'], 'abstract': article['abstract'], 'article_type': article['article_type']}
