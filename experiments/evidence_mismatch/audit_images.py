"""Image integrity checks and contact sheets for externally flagged targets."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

OUT=Path(__file__).resolve().parent/'results'

def main():
    rows=[json.loads(l) for l in (OUT/'manifest.jsonl').read_text().splitlines()]
    checks=[]
    for r in rows:
        images=[]
        for p in [r['image_path']]+[d['image_path'] for d in r['donors']]:
            im=Image.open(p).convert('RGB')
            assert min(im.size)>0
            images.append(dict(path=p,size=im.size,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
        assert len({x['pixel_sha256'] for x in images})==3, 'Duplicate donor/target pixels'
        checks.append(dict(id=r['id'],images=images))
    (OUT/'image_integrity.json').write_text(json.dumps(checks,indent=2))
    cases=[r for r in rows if r['category']=='external_context']
    for page in range((len(cases)+7)//8):
        canvas=Image.new('RGB',(1600,920),'white')
        draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(cases[page*8:page*8+8]):
            im=Image.open(r['image_path']).convert('RGB')
            im=ImageOps.contain(im,(390,410))
            x=(j%4)*400;y=(j//4)*460
            canvas.paste(im,(x+(400-im.width)//2,y+25+(410-im.height)//2))
            draw.text((x+5,y+5),r['id'],fill='black')
            draw.text((x+5,y+440),r['subset'],fill='black')
        canvas.save(OUT/f'external_contact_{page+1}.jpg',quality=90)
    print(f'Checked {len(checks)} records / {len(checks)*3} images; no identical-pixel donors.')

if __name__=='__main__':main()
